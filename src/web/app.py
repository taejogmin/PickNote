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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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


# ---------------------------------------------------------------- 음성 브리핑
# 클립은 따로 받는 음성 팩이다 (Update.cmd가 받을지 묻는다). data/voice/<언어>/*.ogg — data 폴더는 업데이트해도 남는다.
# 클립이 없으면 목록이 비어 화면이 음성 버튼을 숨긴다. 다른 화면 기능은 음성 없이도 그대로 돈다.
VOICE = ROOT / "data" / "voice"


@app.get("/api/voice")
def voice_list():
    """밴픽 음성 브리핑 — 음성 팩에 든 클립 id. 화면은 있는 클립만 이어 붙인다."""
    if not VOICE.is_dir():
        return {}
    return {d.name: sorted(p.stem for p in d.glob("*.ogg")) for d in VOICE.iterdir() if d.is_dir()}


# 자동 음성은 한 번만 — 탭이 둘이거나 다른 브라우저에도 떠 있으면 같은 말을 1초 차이로 겹쳐 읽었다.
# 같은 말(key)을 20초 안에 먼저 달라고 한 화면만 읽는다. 화면을 오래 켜 둔 채 서버를 다시 켜면 BOOT가 바뀌어 화면이 새로고침한다.
BOOT = f"{random.getrandbits(48):x}"
_voice_claims: dict[str, float] = {}


class VoiceClaim(BaseModel):
    key: str


@app.post("/api/voice/claim")
def voice_claim(req: VoiceClaim):
    import time
    now = time.time()
    for k in [k for k, t in _voice_claims.items() if now - t > 20]:
        del _voice_claims[k]
    if req.key in _voice_claims:
        return {"ok": False}
    _voice_claims[req.key] = now
    return {"ok": True}


@app.get("/voice/{lang}/{clip}.ogg")
def voice_clip(lang: str, clip: str):
    if not (lang.replace("_", "").isalnum() and clip.replace("_", "").isalnum()):
        raise HTTPException(404)
    path = VOICE / lang / f"{clip}.ogg"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/ogg", headers={"Cache-Control": "max-age=3600"})


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
        # 머리 줄에는 한 줄로 짧게, 자세한 것은 마우스를 올리면 (좁은 창에서 다섯 줄로 깨지던 것)
        n, latest = info["matches"], (info["patches"] or "").split(" · ")[-1]
        short = L((f"{n / 10000:,.0f}만" if n >= 10000 else f"{n:,}") + "경기" + (f" · 패치 {latest}" if latest else ""),
                  (f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n:,}") + " games" + (f" · patch {latest}" if latest else ""), lang)
        if rg.server() != rg.DEFAULT and rg.usable_model_dir() == rg.model_dir(rg.DEFAULT):
            short += L(" · 한국 모델", " · KR model", lang)      # 고른 서버 모델이 없어 빌려 쓰는 중 — 짧은 줄에도 남긴다
    return {
        "version": _champ_data["version"],
        "champions": _champ_data["champions"],
        "status": status,
        "statusShort": status if _mock else short,
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
            rows.append({"kind": kind, "champs": champs, "games": n, "wins": w, "winrate": pct(n, w)})
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


@app.get("/api/me/tier")
def me_tier():
    """티어 그래프 — 솔로랭크 판의 승패와 갱신 때 찍은 랭크. 기록이 없으면 available: false."""
    return mine.tier_graph()


@app.get("/api/me/form")
def me_form():
    """한눈에 지표 — 라인전 · 캐리력 · 인분 · 멘탈. 타임라인이나 기준 표가 없으면 available: false."""
    return mine.form()


@app.get("/api/me/curve")
def me_curve():
    """시간대별 격차 — 5분마다 맞은편과 CS·골드·경험치 차이. 자료가 없으면 available: false."""
    return mine.curve()


@app.get("/api/me/goal")
def me_goal():
    """오늘의 목표 — 지표마다 한 티어 위 같은 자리 평균, 이미 넘으면 그 선을 지킨다. 오늘 판마다 달성 여부."""
    return mine.goal()


@app.get("/api/me/session")
def me_session():
    """기울어짐 — 지금 이어서 몇 판째 · 연패 · 내 기록에서 n판째부터 승률. 오늘 판은 클라이언트 최근 전적으로 채운다."""
    return mine.session(_lcu.recent_games())


@app.get("/api/me/friends")
def me_friends():
    """친구 알림 — 같이 큐를 잡은 파티원 중 같이 한 판이 많은 사람. 밴픽이 시작될 때 화면이 한 번 부른다."""
    status, session = _lcu.champ_select()
    return mine.friends(_lcu.party(session if status == "champ_select" else None))


@app.get("/api/loading")
def loading_info():
    """로딩 화면 — 10명 표와 음성 클립 id (src/model/loading.py). 다른 사람 전적은 클라이언트에서 읽고 저장하지 않는다.
    같은 판은 한 번만 조회한다 (화면이 다시 불러도 같은 답)."""
    from src.model import loading
    gid = _lcu.game_id()
    if not gid:
        return {"available": False}
    if gid not in _loading_cache:
        out = loading.build(_lcu.loading(), _lcu_state["ally_pos"], _games_of)
        if not out.get("available"):
            return out
        _loading_cache.clear()
        _loading_cache[gid] = out
    return _loading_cache[gid]


_feedback_cache: dict = {}


@app.get("/api/feedback")
def feedback_info(game: int = 0, lang: str | None = None):
    """게임 후 피드백 (src/model/feedback_live.py). game을 안 주면 클라이언트 최근 전적의 가장 최근 협곡 판.
    못 받았으면 available: false — 화면이 잠시 뒤 다시 부른다. 받은 판만 기억한다."""
    from src.i18n import pick as pick_lang
    from src.model import feedback_live
    if not game:
        recent = [g for g in _lcu.recent_games() if g["queue"] in feedback_live.SR and g["dur"] >= feedback_live.MIN_SEC]
        if not recent:
            return {"available": False}
        game = max(recent, key=lambda g: g["t"])["id"]
    key = (game, pick_lang(lang))
    if key not in _feedback_cache:
        out = feedback_live.build(_lcu, game, _lcu_state.get("my_role", ""), _games_of, lang)
        if not out.get("available"):
            return out
        _feedback_cache[key] = out
    return _feedback_cache[key]


@app.get("/api/me/hexagon")
def me_hexagon():
    """승패 육각형. 기록이나 기준 표가 없으면 available: false — 화면은 칸을 숨긴다."""
    return mine.hexagon()


# ---------------------------------------------------------------- 클라이언트 연동

from src.lcu.client import FileClient, LeagueClient  # noqa: E402
from src.lcu.session import parse_session  # noqa: E402
from src.model.positions import POSITIONS, infer_positions  # noqa: E402

_lcu = FileClient(os.environ["LCU_FAKE"]) if os.environ.get("LCU_FAKE") else LeagueClient()
_lcu_state = {"status": "offline", "session_seq": 0, "ally_pos": {}}
_loading_cache: dict = {}


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
    return {**_lcu_status(fixed), "boot": BOOT}


def _lcu_status(fixed: str = ""):
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
        if status == "idle":
            out["flow"] = _lcu.gameflow()      # 게임이 끝나 로비로 돌아온 때를 화면이 안다 (기울어짐 음성 · 게임 후 피드백)
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
    # 로딩 화면에서 게임이 자리를 안 알려 주면 이걸 쓴다 (챔피언 → 자리)
    _lcu_state["ally_pos"] = {v["championId"]: pos for pos, v in ally.items() if v.get("championId")}
    _lcu_state["my_role"] = draft["myRole"] or ""
    if draft["myRole"] and draft["myPick"]:
        _lcu_state["ally_pos"][draft["myPick"]] = draft["myRole"]
    return {
        **out,
        "phase": draft["phase"],
        "myRole": draft["myRole"],
        "myPick": draft["myPick"],
        "myTurn": draft.get("myTurn", False),
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
