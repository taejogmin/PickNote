"""부분 정보 상태에서 추천하기.

밴픽은 늘 정보가 덜 나온 상태다. 상대 1명만 나왔거나, 내가 선픽이라 아무것도
안 나왔거나. 그래서 "10명이 다 찬 조합의 승률"이 아니라
**"지금까지 나온 것으로 다음 픽"**이 실제로 필요한 계산이다.

빈 자리는 그 포지션의 **픽률 가중 평균 챔피언**으로 채운다. 그냥 비워 두면
상호작용 항이 전부 사라져서 추천이 "챔피언 자체 승률" 순서로 무너진다.

평균 벡터를 넣는 게 왜 정확한가: 궁합·상성 항은 각 벡터에 대해 **선형**이다.
⟨v_빈칸, v_아군⟩의 기댓값은 ⟨E[v], v_아군⟩과 정확히 같다. 근사가 아니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.i18n import L
from src.region import usable_model_dir
from src.model.fm import DraftModel, sigmoid

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = usable_model_dir()   # 그 서버 모델이 없으면 들어 있는 한국 모델

# 표본이 적은 챔피언은 추정이 요동쳐서 엉뚱하게 1등으로 올라온다.
# 데이터 추천 후보에서는 빼고, 내가 등록한 챔피언은 보여주되 "표본 부족"을 단다.
MIN_GAMES = 300
DATA_TOP_N = 5
# 픽률 0.5% 미만(21만 경기 기준 약 1,050경기)인 슬롯에는 '비주류' 배지를 단다.
# 미드 질리언처럼 흔치 않은 픽은 장인이 고르는 경우가 많아 승률에 숙련도가 섞였을 수 있다.
# 경기 예측은 이 추정을 쓰는 게 더 정확해서 빼지는 않고, 판단은 사용자에게 맡긴다.
NICHE_SHARE = 0.005
# 그 챔피언 경기의 이만큼이 이 자리여야 '이 자리 픽'으로 본다.
# 비주류 배지와 묻는 것이 다르다 — 배지는 "이 자리에서 드문가", 이건 "이 챔피언에게 낯선 자리인가".
# 브랜드 원딜은 자기 경기의 7%, 스웨인 원딜은 13%다. 15%면 원딜에 베이가(19%)가 남고,
# 35%면 그라가스 정글(21%) 같은 진짜 플렉스 픽까지 잃는다.
OFF_ROLE_SHARE = 0.25
# 추천 이유에 궁합·상성 한 줄로 올릴 최소 크기(%p). 개별 쌍 값은 흔들려서
# 학습에 안 쓴 경기로 믿을 수 있는 크기를 재서 정한다.
# 기준은 배율을 넣기 전 값으로 잰 것이라, 짝 항을 키운 모델에서는 기준도 같은 배율로 올린다
# — 그래야 같은 쌍만 이유에 오른다.
REASON_MIN_PP = 0.1


def extend_with_unknown(
    model: DraftModel, slot_position: np.ndarray, pick_weight: np.ndarray
) -> tuple[DraftModel, np.ndarray]:
    """'아직 안 나온 자리' 슬롯 5개(포지션당 하나)를 모델 뒤에 붙인다.

    돌려주는 것: (확장된 모델, 포지션별 미지 슬롯 번호)
    """
    n = model.n_slots
    unknown = np.arange(n, n + 5)

    out = DraftModel(
        n_slots=n + 5, n_tiers=model.n_tiers, k=model.k,
        use_pairs=model.use_pairs
    )
    out.side = model.side.copy()
    out.cw = model.cw.copy()
    out.cx = model.cx.copy()

    for name in ("b", "v", "a", "d"):
        src = getattr(model, name)
        dst = np.zeros((n + 5,) + src.shape[1:])
        dst[:n] = src
        for p in range(5):
            mask = slot_position == p
            w = pick_weight[mask]
            total = w.sum()
            dst[n + p] = (src[mask] * w[:, None]).sum(0) / total if src.ndim == 2 \
                else (src[mask] * w).sum() / total
        setattr(out, name, dst)

    tb = np.zeros((model.n_tiers, n + 5))
    tb[:, :n] = model.tier_b
    for p in range(5):
        mask = slot_position == p
        w = pick_weight[mask]
        tb[:, n + p] = (model.tier_b[:, mask] * w).sum(1) / w.sum()
    out.tier_b = tb
    return out, unknown


def build_draft(
    ally: dict[str, int],
    enemy: dict[str, int],
    unknown: np.ndarray,
    slot_of: dict[tuple[int, int], int],
) -> np.ndarray:
    """밴픽 상태 → (1, 10) 배열. 안 나온 자리는 '평균 챔피언'으로 채운다."""
    row = np.empty(10, dtype=np.int64)
    for p, position in enumerate(POSITIONS):
        for offset, side in ((0, ally), (5, enemy)):
            champion = side.get(position)
            slot = slot_of.get((champion, p)) if champion is not None else None
            row[offset + p] = unknown[p] if slot is None else slot
    return row[None, :]


def rank_candidates(
    model: DraftModel,
    draft: np.ndarray,
    my_position: int,
    candidates: np.ndarray,
    tier: int = 0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """내 자리에 후보를 하나씩 넣어 본 승률. (후보 순서 그대로의 승률, 정렬 인덱스)

    scale은 과신 보정 기울기다. 모델은 끝으로 갈수록 과하게 확신해서 차이를 줄여 읽는다.
    사이드 이점은 뺀다 — 두 팀이 같으면 50%가 나와야 한다.
    """
    grid = np.repeat(draft, len(candidates), axis=0)
    grid[:, my_position] = candidates
    tiers = np.full(len(grid), tier, dtype=np.int64)
    win = sigmoid(scale * (model.logits(grid, tiers) - model.side[0]))
    return win, np.argsort(-win)


# ---------------------------------------------------------------- 화면에 답하기

class Recommender:
    """저장된 모델(`python -m src.model.train`의 결과)로 화면이 쓰는 답을 만든다."""

    def __init__(self, model: DraftModel, slots, games, meta: dict, curve=None):
        self.base = model
        self.slots = [(int(c), int(p)) for c, p in slots]
        self.slot_of = {s: i for i, s in enumerate(self.slots)}
        self.games = np.asarray(games, dtype=np.int64)
        self.meta = meta
        self.slot_position = np.array([p for _, p in self.slots])
        # 챔피언마다 다섯 자리를 합친 경기 수 — "이 챔피언에게 이 자리가 얼마나 제 자리인가"를 잰다
        self.champion_games: dict[int, int] = {}
        for (champion, _), n in zip(self.slots, self.games):
            self.champion_games[champion] = self.champion_games.get(champion, 0) + int(n)
        # 과신 보정 기울기 (train.py가 튜닝 기간에서 맞춰 저장). 옛 파일에는 없어서 1.
        self.scale = float(meta.get("calibration", {}).get("a", 1.0))
        self.reason_min_pp = REASON_MIN_PP * float(meta.get("pair_scale") or 1.0)
        # 강함은 포지션마다 기준점이 떠 있다 — 한 팀에 포지션이 하나씩이라 포지션별 상수는
        # 예측에서 상쇄되기 때문이다. 화면에 "+5%p"라고 쓰려면 그 포지션 평균(픽률 가중)을 빼야 한다.
        # 티어 보정이 켜져 있으면 티어마다 강함이 다르므로 기준점도 티어마다 잡는다
        self.tier_names = list(meta.get("tiers", [])) if model.n_tiers > 1 else []
        self.strength = np.stack([model.b + model.tier_b[t] for t in range(model.n_tiers)])
        center = np.zeros_like(self.strength)
        for t in range(model.n_tiers):
            for q in range(len(POSITIONS)):
                k = self.slot_position == q
                if self.games[k].sum():
                    center[t, k] = np.average(self.strength[t, k], weights=self.games[k])
        self.center = center
        # 빈 자리는 픽률(=경기 수) 가중 평균 챔피언으로 채운다
        self.model, self.unknown = extend_with_unknown(
            model, self.slot_position, self.games.astype(np.float64)
        )
        # 경기 길이 구간별 차이값. 안 나온 자리는 그 포지션 평균으로 채운다.
        self.curve = None
        if curve is not None and len(curve):
            ext = np.zeros((len(curve), self.model.n_slots))
            ext[:, :len(self.slots)] = curve
            for q in range(len(POSITIONS)):
                k = self.slot_position == q
                w = self.games[k].astype(np.float64)
                if w.sum():
                    ext[:, self.unknown[q]] = (curve[:, k] * w).sum(1) / w.sum()
            self.curve = ext
        self.curve_labels = list(meta.get("curve", {}).get("labels", []))

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "Recommender":
        arrays = np.load(directory / "draft.npz")
        meta = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
        model = DraftModel(
            n_slots=len(arrays["b"]), n_tiers=meta["n_tiers"], k=meta["k"],
            use_pairs=meta["use_pairs"],
        )
        for name in ("b", "tier_b", "v", "a", "d", "cw", "cx", "side"):
            setattr(model, name, arrays[name].copy())
        curve = arrays["curve"].copy() if "curve" in arrays.files else None
        # 옛 Adam 모델(2026-09-11 이전 파일)은 티어가 하나여도 tier_b가 b의 복사본이라
        # 실제 강함이 합이다. 합쳐 둔다. 예측은 바뀌지 않는다.
        if model.n_tiers == 1 and np.any(model.tier_b):
            model.b = model.b + model.tier_b[0]
            model.tier_b = np.zeros_like(model.tier_b)
        return cls(model, arrays["slots"], arrays["games"], meta, curve)

    # ------------------------------------------------------------------ 상태

    def info(self) -> dict:
        return {
            "matches": self.meta["matches"],
            "label": self.meta["label"],
            "use_pairs": self.meta["use_pairs"],
            "patches": " · ".join(self.meta.get("patches", [])[-2:]),
            "trained_at": self.meta.get("trained_at", ""),
            "tier_correction": bool(self.tier_names),
            # 실제로 차이값이 들어간 티어. 화면은 이것만 묻는다
            "corrected_tiers": list(self.meta.get("corrected_tiers", self.tier_names)),
        }

    def default_tier(self):
        """화면의 티어 선택에 처음 놓을 값. 보정이 없는 모델이면 None — 화면이 선택을 숨긴다."""
        return self.tier_names[len(self.tier_names) // 2] if self.tier_names else None

    def tier_index(self, name: str | None) -> int:
        """모르는 이름이면 가운데 티어로 — 실버쯤이 가장 덜 틀린다."""
        if not self.tier_names:
            return 0
        try:
            return self.tier_names.index(name)
        except ValueError:
            return len(self.tier_names) // 2

    # ------------------------------------------------------------------ 추천

    def _evaluate(self, draft: np.ndarray, position: int, champions: list[int], tier: int = 0):
        """챔피언들을 내 자리에 하나씩 넣어 본 승률과, 각자의 슬롯 번호."""
        slots = np.array(
            [self.slot_of.get((c, position), self.unknown[position]) for c in champions],
            dtype=np.int64,
        )
        win, _ = rank_candidates(self.model, draft, position, slots, tier=tier, scale=self.scale)
        return win, slots

    def _curves(self, draft: np.ndarray, position: int, slots: np.ndarray, tier: int) -> np.ndarray:
        """후보마다 길이 구간별 승률 (n_후보, 구간). 사이드 이점은 뺀다.

        "그 길이까지 갔다면 그때 승률"이지, 그만큼 갈 것이라는 예측이 아니다.
        """
        grid = np.repeat(draft, len(slots), axis=0)
        grid[:, position] = slots
        base = self.model.logits(grid, np.full(len(grid), tier, dtype=np.int64)) - self.model.side[0]
        add = self.curve[:, grid]                       # (구간, 후보, 10)
        team = add[:, :, :5].sum(2) - add[:, :, 5:].sum(2)
        return sigmoid(self.scale * (base[None, :] + team)).T

    def _reasons(self, draft: np.ndarray, position: int, slot: int, win: float, tier: int = 0) -> dict:
        """승률을 '그 자리에 평균 챔피언을 뒀을 때' 대비 챔피언 자체 · 아군 궁합 · 상대 상성으로 쪼갠다.

        궁합·상성 값은 그 자체로는 기준점이 없어서 평균 챔피언과의 차이로 쪼갠다.
        로짓 차이를 지금 승률에서의 기울기로 %p로 바꾸므로 합은 승률 차이와 대략 맞는다.
        아직 안 나온 자리와의 몫은 평균 벡터도 0이 아니라서 따로 한 줄로 모은다.
        """
        m, u = self.model, self.unknown[position]
        unknown = set(int(x) for x in self.unknown)
        k = win * (1.0 - win) * 100.0 * self.scale
        row = draft[0]
        W = m.pair_weight()
        dv, da, dd = m.v[slot] - m.v[u], m.a[slot] - m.a[u], m.d[slot] - m.d[u]

        synergy, counter, rest = [], [], 0.0
        for q in range(len(POSITIONS)):
            if q != position:
                j = int(row[q])
                pp = W[position, q] * float(dv @ m.v[j]) * k if self.base.use_pairs else 0.0
                if j in unknown:
                    rest += pp
                else:
                    synergy.append({"championId": self.slots[j][0], "position": POSITIONS[q], "pp": round(pp, 2)})
            j = int(row[5 + q])
            pp = (m.cx[position, q] * float(da @ m.d[j]) - m.cx[q, position] * float(m.a[j] @ dd)) * k \
                if self.base.use_pairs else 0.0
            if j in unknown:
                rest += pp
            else:
                counter.append({"championId": self.slots[j][0], "position": POSITIONS[q], "pp": round(pp, 2)})

        by_size = lambda items: sorted(items, key=lambda r: -abs(r["pp"]))
        return {
            "strength": round(float((m.b[slot] + m.tier_b[tier, slot])
                                    - (m.b[u] + m.tier_b[tier, u])) * k, 2),
            "synergy": by_size(synergy),
            "counter": by_size(counter),
            "unrevealed": round(rest, 2),
        }

    def _row(self, champion: int, win: float, slot: int, draft: np.ndarray, position: int,
             tier: int = 0, curve=None, lang: str | None = None) -> dict:
        games = int(self.games[slot]) if slot < len(self.games) else 0
        row = {"championId": int(champion), "winrate": round(float(win) * 100, 1)}
        row["reasons"] = self._reasons(draft, position, int(slot), float(win), tier)
        if curve is not None:
            row["curve"] = [round(float(v) * 100, 1) for v in curve]
        if games == 0:
            # 이 포지션에서 한 번도 안 나온 챔피언. 평균 챔피언으로 계산된 값이다.
            row["why"] = L("이 포지션 기록 없음 — 평균으로 계산", "No games in this role — estimated from the average", lang)
            row["lowSample"] = True
            return row
        # 슬롯 하나가 팀 logit에 b만큼 더하면 승률은 0.5 근처에서 대략 b/4 움직인다
        effect = float(self.strength[tier, slot] - self.center[tier, slot]) * self.scale * 25.0
        row["why"] = L(f"챔피언 자체 {effect:+.1f}%p · {games:,}경기", f"Champion itself {effect:+.1f}pp · {games:,} games", lang)
        if games < MIN_GAMES:
            row["lowSample"] = True
        # 픽률 = 이 포지션에 나온 경기 ÷ 전체 경기 (OP.GG와 같은 정의). 한 경기에 같은 포지션이
        # 양 팀에 하나씩 있어서, 포지션 슬롯 합으로 나누면 픽률이 절반으로 계산된다 (처음에 그렇게 틀렸다).
        if games < NICHE_SHARE * self.meta["matches"]:
            row["niche"] = True
        if self.off_role(int(champion), games):
            row["offRole"] = True
        return row

    def off_role(self, champion: int, games: int) -> bool:
        """이 챔피언에게 낯선 자리인가 — 자기 경기의 25% 미만이면 그렇다."""
        total = self.champion_games.get(champion, 0)
        return bool(total) and games < OFF_ROLE_SHARE * total

    def _split_reason(self, rows: list[dict], lang: str | None = None) -> str:
        """맞대결에서 1위가 갈린 이유. 지금 모델이 무엇까지 보는지를 넘어서 말하지 않는다."""
        gap = rows[0]["winrate"] - rows[1]["winrate"]
        if not self.base.use_pairs:
            return L(f"챔피언 자체 강함 차이 <em>+{gap:.1f}%p</em> · 지금 모델은 조합 상성을 반영하지 않습니다",
                     f"Champion strength gap <em>+{gap:.1f}pp</em> · this model doesn't account for team matchups", lang)
        return L(f"이 조합에서 <em>+{gap:.1f}%p</em> 앞섭니다", f"Ahead by <em>+{gap:.1f}pp</em> in this draft", lang)

    def recommend(self, req) -> dict:
        position = POSITIONS.index(req.role)
        tier = self.tier_index(getattr(req, "tier", None))
        lang = getattr(req, "lang", None)
        ally = {k: v for k, v in req.ally.items() if k != req.role}   # 내 자리는 비워 둔다
        enemy = dict(req.enemy)
        used = set(ally.values()) | set(enemy.values()) | set(req.bans)
        draft = build_draft(ally, enemy, self.unknown, self.slot_of)

        # --- 지금 고민 중 ------------------------------------------
        cands = [c for c in dict.fromkeys(req.candidates) if c not in used]
        cand_rows: list[dict] = []
        if cands:
            win, slots = self._evaluate(draft, position, cands, tier)
            curves = self._curves(draft, position, slots, tier) if self.curve is not None else [None] * len(cands)
            cand_rows = sorted(
                (self._row(c, w, s, draft, position, tier, cv, lang)
                 for c, w, s, cv in zip(cands, win, slots, curves)),
                key=lambda r: -r["winrate"],
            )
            cand_rows[0]["best"] = True
            for r in cand_rows[1:]:
                r["delta"] = round(r["winrate"] - cand_rows[0]["winrate"], 1)
            if len(cand_rows) > 1:
                cand_rows[0]["why"] = self._split_reason(cand_rows, lang)

        # --- 다른 선택지: 내 풀 + 데이터 상위, 후보는 뺀다 -----------------
        pool = [c for c in dict.fromkeys(req.pool) if c not in used and c not in cands]
        data_slots = np.flatnonzero((self.slot_position == position) & (self.games >= MIN_GAMES))
        data_champs = [self.slots[s][0] for s in data_slots]
        data_champs = [c for c in data_champs if c not in used and c not in cands and c not in pool]
        # 오프포지션 숨기기 — **상위 5를 뽑기 전에** 거른다. 뽑은 뒤에 지우면 2~3개만 남는다.
        # 내 풀과 고민 중은 사용자가 직접 고른 것이라 여기서 거르지 않는다.
        if getattr(req, "hide_off_role", True):
            data_champs = [c for c in data_champs
                           if not self.off_role(c, int(self.games[self.slot_of[(c, position)]]))]

        others: list[dict] = []
        everyone = pool + data_champs
        if everyone:
            win, slots = self._evaluate(draft, position, everyone, tier)
            curves = self._curves(draft, position, slots, tier) if self.curve is not None else [None] * len(everyone)
            rows = [self._row(c, w, s, draft, position, tier, cv, lang)
                    for c, w, s, cv in zip(everyone, win, slots, curves)]
            pool_rows = rows[: len(pool)]
            for r in pool_rows:
                r["inPool"] = True
            data_rows = sorted(rows[len(pool):], key=lambda r: -r["winrate"])[:DATA_TOP_N]
            others = sorted(pool_rows + data_rows, key=lambda r: -r["winrate"])

        # 후보 1위보다 나은 게 있으면 그 차이 — "내가 놓친 선택지가 있나"에 대한 답
        if others and cand_rows and others[0]["winrate"] > cand_rows[0]["winrate"]:
            others[0]["betterBy"] = round(others[0]["winrate"] - cand_rows[0]["winrate"], 1)

        # 비교 기준: 내 자리에 평균 챔피언을 뒀을 때의 승률
        base_win, _ = rank_candidates(self.model, draft, position,
                                      np.array([self.unknown[position]]), tier=tier, scale=self.scale)
        revealed = len(ally) + len(enemy)
        kind = (L("궁합·상성 반영", "synergy & counters", lang) if self.base.use_pairs
                else L("티어표 수준 · 조합 상성 미반영", "tier-list level · no team matchups", lang))
        return {
            "candidates": cand_rows,
            "others": others,
            "note": L(f"{self.meta['matches']:,}경기 · {kind} · {revealed}명 공개",
                      f"{self.meta['matches']:,} games · {kind} · {revealed} revealed", lang),
            "averageWinrate": round(float(base_win[0]) * 100, 1),
            "curveLabels": self.curve_labels,
            "reasonMinPp": round(self.reason_min_pp, 3),
        }
