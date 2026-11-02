"""로컬 웹 서버.

    python -m uvicorn src.web.app:app --reload --port 8000

학습된 모델이 없으면 가짜 추천을 돌려주는 목업 모드로 뜬다.
화면과 조작감을 데이터 수집이 끝나기 전에 확인하려고 이렇게 했다.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data.champions import load as load_champions
from src import region as rg
from src.i18n import L, default_lang
from src.model import mine

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="PickNote")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_champ_data = load_champions()
_recommender = None
_mock = True


def _try_load_model():
    """모델이 있으면 쓰고, 없으면 목업으로 간다."""
    global _recommender, _mock
    try:
        from src.model.recommend import Recommender

        _recommender = Recommender.load()
        _mock = False
    except FileNotFoundError:
        print(L("학습된 모델이 없어 목업으로 뜹니다. `python -m src.model.train`을 먼저 돌리세요.",
                "No trained model found — starting in mock mode. Run `python -m src.model.train` first."),
              file=sys.stderr)
        _recommender, _mock = None, True
    except Exception as exc:
        # 조용히 목업으로 넘어가면 모델 쪽 버그가 가짜 숫자 뒤에 숨는다
        print(L(f"모델을 불러오다 실패해 목업으로 뜹니다: {type(exc).__name__}: {exc}",
                f"Failed to load the model — starting in mock mode: {type(exc).__name__}: {exc}"),
              file=sys.stderr)
        _recommender, _mock = None, True


_try_load_model()


class DraftRequest(BaseModel):
    role: str
    tier: str = "IRON"
    ally: dict[str, int] = {}
    enemy: dict[str, int] = {}
    bans: list[int] = []
    pool: list[int] = []          # 내가 등록해 둔 이 라인의 챔피언 (저장됨)
    candidates: list[int] = []    # 이번 판에 고민 중인 챔피언 (임시)
    hide_off_role: bool = True    # 오프포지션 픽 숨기기
    top_n: int = 5
    lang: str | None = None


# 화면을 고친 뒤 브라우저가 옛 페이지를 캐시에서 꺼내 쓰지 않게 한다.
# no-cache는 "매번 서버에 바뀌었는지 물어보라"는 뜻이라, 안 바뀌었으면 여전히 가볍다.
NO_CACHE = {"Cache-Control": "no-cache"}


def _page(name: str) -> HTMLResponse:
    """정적 페이지에 기본 언어를 넣어 보낸다. 화면에서 바꾼 언어는 브라우저가 기억해 이걸 덮는다."""
    html = (STATIC / name).read_text(encoding="utf-8")
    return HTMLResponse(html.replace('<html lang="ko"', f'<html lang="{default_lang()}"', 1), headers=NO_CACHE)


@app.get("/")
def index():
    return _page("index.html")


@app.get("/api/meta")
def meta(lang: str | None = None):
    if _mock:
        status = L(
            "<b style='color:#c8aa6e'>목업 모드</b><br>학습된 모델이 없어 가짜 숫자를 보여줍니다",
            "<b style='color:#c8aa6e'>Mock mode</b><br>No trained model yet — numbers are fake",
            lang,
        )
    else:
        info = _recommender.info()
        kind = L("궁합·상성 반영", "synergy & counters", lang) if info["use_pairs"] else L("티어표 수준", "tier-list level", lang)
        if info.get("tier_correction"):
            kind += L(" · 티어 보정", " · tier correction", lang)
        status = L(
            f"{info['matches']:,}경기 · {kind}<br>패치 {info['patches']} · {info['trained_at']} 학습",
            f"{info['matches']:,} games · {kind}<br>Patch {info['patches']} · trained {info['trained_at']}",
            lang,
        )
        if rg.server() != rg.DEFAULT and rg.usable_model_dir() == rg.model_dir(rg.DEFAULT):
            # 고른 서버 모델이 아직 없어 한국 모델을 빌려 쓰는 중
            status += L(f"<br>{rg.name(lang=lang)} 모델이 아직 없어 한국 모델을 씁니다",
                        f"<br>No {rg.name(lang=lang)} model yet — using the Korean model", lang)
    return {
        "version": _champ_data["version"],
        "champions": _champ_data["champions"],
        "status": status,
        "defaultTier": "IRON" if _mock else _recommender.default_tier(),
        # 아이언에만 보정이 있으면 화면은 다섯 칸 드롭다운 대신 "아이언입니까?" 한 칸을 띄운다.
        # 체크를 안 하면 baseTier를 보내 보정 없이 받는다.
        "ironOnly": _iron_only(),
        "baseTier": "SILVER",
        # 포지션 필터용. 모델이 이미 들고 있는 슬롯별 경기 수라 계산할 것이 없다.
        "positionGames": _position_games(),
        # 내 경기 기록이 있나. 없으면 화면이 "내 기록" 칸을 숨긴다
        "me": mine.available(),
        "lang": default_lang(),
    }


def _iron_only() -> bool:
    """차이값이 아이언에만 들어 있나 — 그때만 화면이 티어를 묻는다."""
    if _recommender is None:
        return False
    return _recommender.info().get("corrected_tiers") == ["IRON"]


def _position_games() -> dict[str, list[int]]:
    """챔피언 → 포지션 5칸의 경기 수. 화면이 이 숫자로 픽률 0.5%를 가른다.

    목업 모드에서는 비워 보낸다 — 화면이 필터 줄을 숨긴다.
    """
    if _recommender is None:
        return {}
    return {str(cid): _games_of(int(cid)) for cid in _champ_data["champions"]}


@app.post("/api/recommend")
def recommend(req: DraftRequest):
    if _mock:
        return _mock_recommend(req)
    return _with_my_record(_recommender.recommend(req), req.role, req.enemy, req.ally)


def _with_my_record(out: dict, role: str, enemy: dict | None = None, ally: dict | None = None) -> dict:
    """추천 줄에 내 기록을 단다. 사실이지 예측이 아니다 — 승률 숫자는 건드리지 않는다.

    숙련도를 승률에 섞는 것은 검증을 못 넘었다.
    칸이 정해진 만큼 내 기록 줄을 만든다 — 맞은편 라이너 · 상대 봇 둘 · 우리 봇 짝. 챔피언마다 3판 이상이면 표.
    """
    if not mine.available():
        return out
    enemy, ally = enemy or {}, ally or {}
    record = mine.record()
    pct = lambda n, w: round(w / n * 100) if n else None

    lines = []          # (종류, 챔피언들, 조건) — 화면이 위에서부터 그대로 보여 준다
    if enemy.get(role):
        lines.append(("foe", [enemy[role]], {"enemy": {role: enemy[role]}}))
    if role in mine.BOT_PARTNER and enemy.get("BOTTOM") and enemy.get("UTILITY"):
        lines.append(("enemyDuo", [enemy["BOTTOM"], enemy["UTILITY"]],
                      {"enemy": {"BOTTOM": enemy["BOTTOM"], "UTILITY": enemy["UTILITY"]}}))
    partner = mine.BOT_PARTNER.get(role)
    if partner and ally.get(partner):
        lines.append(("partner", [ally[partner]], {"ally": {partner: ally[partner]}}))
    tallies = {kind: mine.matching(role, **cond) for kind, _c, cond in lines}

    tag_key = {"foe": "mineVs", "partner": "mineDuo"}     # 상대 봇 둘 × 내 챔피언은 표본이 안 모여 줄만 둔다
    for row in out.get("candidates", []) + out.get("others", []):
        n, w = record.get((row["championId"], role), (0, 0))
        if n >= mine.TAG_MIN:
            row["mine"] = {"games": n, "winrate": pct(n, w)}
        for kind, key in tag_key.items():
            n, w = tallies.get(kind, {}).get(row["championId"], (0, 0))
            if n >= mine.TAG_MIN:
                row[key] = {"games": n, "winrate": pct(n, w)}
    if lines:
        rn = sum(n for (_c, p), (n, _w) in record.items() if p == role)
        rw = sum(w for (_c, p), (_n, w) in record.items() if p == role)
        rows = []
        for kind, champs, _cond in lines:
            n = sum(n for n, _w in tallies[kind].values())
            w = sum(w for _n, w in tallies[kind].values())
            rows.append({"kind": kind, "champs": champs, "games": n, "winrate": pct(n, w)})
        out["myLane"] = {"roleGames": rn, "roleWinrate": pct(rn, rw), "rows": rows}
    return out


# ---------------------------------------------------------------- 내 기록

@app.get("/me")
def me_page():
    return _page("me.html")


@app.get("/api/me/pool")
def me_pool():
    """자리마다 5판 이상 한 챔피언. 기록이 없으면 빈 것."""
    return {"pool": mine.pool(), "min": mine.POOL_MIN}


@app.get("/api/me/summary")
def me_summary(lang: str | None = None):
    out = mine.summary(_recommender, lang)
    out["version"] = _champ_data["version"]
    out["champions_meta"] = {cid: {"name": c["name"], "en": c["en"], "image": c["image"]}
                             for cid, c in _champ_data["champions"].items()}
    return out


@app.get("/api/me/hexagon")
def me_hexagon():
    """승패 육각형. 기록이나 기준 표가 없으면 available: false — 화면은 칸을 숨긴다."""
    return mine.hexagon()


# ---------------------------------------------------------------- 클라이언트 연동

from src.lcu.client import FileClient, LeagueClient  # noqa: E402
from src.lcu.session import parse_session  # noqa: E402
from src.model.positions import POSITIONS, infer_positions  # noqa: E402

_lcu = FileClient(os.environ["LCU_FAKE"]) if os.environ.get("LCU_FAKE") else LeagueClient()
_lcu_state = {"status": "offline", "session_seq": 0}


def _games_of(champion: int):
    """포지션 5개의 경기 수. 모델이 없으면 0이라 추정은 고르게 나온다."""
    if _recommender is None:
        return [0] * len(POSITIONS)
    out = []
    for p in range(len(POSITIONS)):
        slot = _recommender.slot_of.get((champion, p))
        out.append(int(_recommender.games[slot]) if slot is not None else 0)
    return out


def _parse_fixed(raw: str) -> dict[str, dict[str, int]]:
    """화면이 손으로 지정한 칸. "e:BOTTOM:22,a:TOP:2" → {"enemy": {"BOTTOM": 22}, "ally": {"TOP": 2}}"""
    out: dict[str, dict[str, int]] = {"ally": {}, "enemy": {}}
    for item in (raw or "").split(","):
        parts = item.split(":")
        if len(parts) == 3 and parts[0] in ("a", "e") and parts[1] in POSITIONS and parts[2].isdigit():
            out["ally" if parts[0] == "a" else "enemy"][parts[1]] = int(parts[2])
    return out


def _fixed_entry(champion: int) -> dict:
    return {"championId": champion, "inferred": False, "fixed": True, "confidence": 1.0, "uncertain": False}


@app.get("/api/lcu")
def lcu(fixed: str = "", lang: str | None = None):
    """밴픽 창 상태. 화면이 1초마다 부른다. 읽기만 한다.

    session_seq는 새 밴픽이 시작될 때마다 1씩 오른다 — 화면은 이게 바뀌면 판을 비운다.
    fixed는 화면에서 손으로 지정한 칸이다. 그 칸은 그대로 두고 나머지만 추정한다.
    """
    status, session = _lcu.champ_select()
    if status == "champ_select" and _lcu_state["status"] != "champ_select":
        _lcu_state["session_seq"] += 1
    _lcu_state["status"] = status
    out = {"status": status, "sessionSeq": _lcu_state["session_seq"]}
    if status != "champ_select":
        return out

    draft = parse_session(session)
    pinned = _parse_fixed(fixed)

    ally = {pos: _fixed_entry(c) for pos, c in pinned["ally"].items() if pos != draft["myRole"]}
    pinned_ally = {v["championId"] for v in ally.values()}
    unplaced = [c for c in draft["allyUnplaced"] if c not in pinned_ally]
    for pos, c in draft["ally"].items():
        if c in pinned_ally:
            continue
        if pos in ally:                      # 사람이 이 자리에 다른 챔피언을 지정했다 — 추정으로 돌린다
            unplaced.append(c)
        else:
            ally[pos] = {"championId": c, "inferred": False, "confidence": 1.0, "uncertain": False}
    taken = tuple(ally) + ((draft["myRole"],) if draft["myRole"] else ())
    for guess in infer_positions(unplaced, _games_of, taken=taken):
        ally[guess["position"]] = {**guess, "inferred": True}

    enemy = {pos: _fixed_entry(c) for pos, c in pinned["enemy"].items()}
    pinned_enemy = {v["championId"] for v in enemy.values()}
    rest = [c for c in draft["enemy"] if c not in pinned_enemy]
    for guess in infer_positions(rest, _games_of, taken=tuple(enemy)):
        enemy[guess["position"]] = {**guess, "inferred": True}
    return {
        **out,
        "phase": draft["phase"],
        "myRole": draft["myRole"],
        "myPick": draft["myPick"],
        "hover": draft["hover"],
        "ally": ally,
        "enemy": enemy,
        "bans": draft["bans"],
    }


# ---------------------------------------------------------------- 목업

_MOCK_REASONS = [
    ("상대 {e}에게 강함 <em>+1.2%p</em> · 챔피언 자체 <em>+0.4%p</em>",
     "Strong into {e} <em>+1.2pp</em> · champion itself <em>+0.4pp</em>"),
    ("아군 조합과 궁합 <em>+0.9%p</em> · 상대 {e} 상대 <em>+0.5%p</em>",
     "Synergy with allies <em>+0.9pp</em> · vs {e} <em>+0.5pp</em>"),
    ("이 포지션 기본 승률이 높음 <em>+1.1%p</em>", "High base win rate in this role <em>+1.1pp</em>"),
    ("상대 {e}에게 약함 <em>−0.6%p</em>이지만 아군 궁합이 큼 <em>+1.4%p</em>",
     "Weak into {e} <em>−0.6pp</em> but strong ally synergy <em>+1.4pp</em>"),
]


def _mock_recommend(req: DraftRequest):
    """가짜 추천을 만든다. 화면 확인용."""
    used = set(req.ally.values()) | set(req.enemy.values()) | set(req.bans)
    available = [cid for cid in _champ_data["champions"] if cid not in used]
    random.seed(hash((tuple(sorted(used)), req.role, req.tier)) & 0xFFFFFFFF)

    lang = req.lang
    name_key = "en" if L("ko", "en", lang) == "en" else "name"
    enemy_name = L("상대", "the enemy", lang)
    if req.enemy:
        enemy_name = _champ_data["champions"].get(
            next(iter(req.enemy.values())), {}
        ).get(name_key, enemy_name)

    def fake_winrate(cid: int) -> float:
        return 49.0 + (hash((cid, req.role, req.tier)) % 1000) / 1000 * 5.5

    # 고민 중인 후보 — 승률 순, 1위에만 "왜 갈렸는지"
    cands = [c for c in req.candidates if c not in used]
    cand_rows = sorted(
        ({"championId": c, "winrate": fake_winrate(c)} for c in cands),
        key=lambda r: -r["winrate"],
    )
    if cand_rows:
        cand_rows[0]["best"] = True
        best = _champ_data["champions"][cand_rows[0]["championId"]][name_key]
        cand_rows[0]["why"] = L(f"{best}가 상대 {enemy_name}에게 강한 게 갈랐습니다 <em>+1.1%p</em>",
                                f"{best} being strong into {enemy_name} made the difference <em>+1.1pp</em>", lang)
        for row in cand_rows[1:]:
            row["delta"] = round(row["winrate"] - cand_rows[0]["winrate"], 1)

    # 다른 선택지 — 내 풀 + 데이터 상위, 후보는 뺀다
    pool_ids = [c for c in req.pool if c not in used and c not in cands]
    data_ids = [c for c in random.sample(available, 8)
                if c not in cands and c not in pool_ids][: req.top_n]
    others = [{"championId": c, "winrate": fake_winrate(c), "inPool": c in pool_ids,
               "why": L(*_MOCK_REASONS[i % len(_MOCK_REASONS)], lang).format(e=enemy_name)}
              for i, c in enumerate(pool_ids + data_ids)]
    others.sort(key=lambda r: -r["winrate"])
    others = others[:6]

    # 후보 1위보다 나은 게 있으면 그 차이를 알려준다 - "내가 놓친 게 있나"에 대한 답
    if others and cand_rows and others[0]["winrate"] > cand_rows[0]["winrate"]:
        others[0]["betterBy"] = round(
            others[0]["winrate"] - cand_rows[0]["winrate"], 1
        )

    revealed = len(req.ally) + len(req.enemy)
    return {
        "candidates": cand_rows,
        "others": others,
        "note": L(f"목업 — 실제 데이터가 아닙니다 · {revealed}명 공개", f"Mock — not real data · {revealed} revealed", lang),
    }
