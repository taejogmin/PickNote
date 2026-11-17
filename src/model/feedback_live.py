"""게임 후 피드백 — 방금 끝난 판을 받아 src/model/feedback.py 로 분석한다.

받는 순서 (먼저 되는 것):
1. 내 기록(me.db)에 이미 있으면 그것
2. Riot API 키가 있으면 match-v5 경기 + 타임라인 (끝나고 1~2분 뒤에 올라온다 — 없으면 화면이 잠시 뒤 다시 부른다)
3. 롤 클라이언트(LCU)의 경기 기록 + 타임라인 — 형식이 조금 달라 맞춰 쓴다 (1분 딜량이 없으면 오브젝트 싸움 순위는 빠진다)
받은 판은 저장하지 않는다 (내 기록 갱신은 따로).
매치 ID는 지금 서버의 플랫폼 + 게임 번호 (KR_123, EUW1_123 …), 지역 주소는 src/region.py가 고른다.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import statistics
from collections import defaultdict

from src import region as rg
from src.data import items as itemdata
from src.model import feedback
from src.model.positions import POSITIONS, infer_positions

SR = {400, 420, 430, 440, 480, 490, 700}
MIN_SEC = 300
CORE_MIN_GAMES = 5

_core: dict[int, float] | None = None
_items: dict | None = None


def _db():
    from src.model.mine import DB
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)


def match_id(game_id: int) -> str:
    return f"{rg.platform().upper()}_{game_id}"


def _my_puuids(conn) -> set:
    try:
        return {r[0] for r in conn.execute("SELECT puuid FROM account")}
    except sqlite3.Error:
        return set()


def items() -> dict:
    global _items
    if _items is None:
        _items = itemdata.load()
    return _items


def core_avg() -> dict[int, float]:
    """챔피언 → 내 판들의 첫 코어템 시각 중앙값 (5판 이상). 서버가 켜진 뒤 한 번 계산한다."""
    global _core
    if _core is not None:
        return _core
    times = defaultdict(list)
    try:
        from src.model.mine import DB
        if not DB.exists():
            raise sqlite3.Error
        conn = _db()
        for mr, tr in conn.execute("SELECT m.raw, t.raw FROM my_match m JOIN my_timeline_raw t USING(match_id)"):
            m, line = json.loads(gzip.decompress(mr)), json.loads(gzip.decompress(tr))
            pid = next((p["participantId"] for p in line["info"]["participants"] if p.get("puuid") == "me"), None)
            if not pid:
                continue
            t = feedback.first_core(feedback.Game(m, line, pid), items())
            if t:
                times[m["info"]["participants"][pid - 1]["championId"]].append(t)
    except sqlite3.Error:
        pass
    _core = {c: statistics.median(v) for c, v in times.items() if len(v) >= CORE_MIN_GAMES}
    return _core


def from_db(game_id: int):
    from src.model.mine import DB
    if not DB.exists():
        return None
    try:
        conn = _db()
        q = "SELECT m.raw, t.raw FROM my_match m JOIN my_timeline_raw t USING(match_id) WHERE "
        row = (conn.execute(q + "match_id = ?", (match_id(game_id),)).fetchone()
               # 서버를 바꾼 뒤에도 예전 서버 판을 찾게 — 게임 번호만 맞춰 본다
               or conn.execute(q + "substr(match_id, instr(match_id, '_') + 1) = ?", (str(game_id),)).fetchone())
    except sqlite3.Error:
        return None
    if not row:
        return None
    m, line = json.loads(gzip.decompress(row[0])), json.loads(gzip.decompress(row[1]))
    pid = next((p["participantId"] for p in line["info"]["participants"] if p.get("puuid") == "me"), None)
    return (m, line, pid, "db") if pid else None


def from_api(game_id: int):
    try:
        from src.riot.client import RiotClient
        client = RiotClient(verbose=False)
    except (SystemExit, Exception):
        return None
    mid = match_id(game_id)
    try:
        m = client.match(mid)
        line = client.get(f"https://{rg.region_of_match(mid)}.api.riotgames.com/lol/match/v5/matches/{mid}/timeline",
                          retries=1) if m else None
    except (SystemExit, Exception):          # 키가 만료되면 클라이언트가 SystemExit 를 던진다 — 서버를 끄지 않고 다음 방법으로
        return None
    if not m or not line:
        return None
    try:
        mine = _my_puuids(_db())
    except sqlite3.Error:
        mine = set()
    pid = next((p["participantId"] for p in m["info"]["participants"] if p.get("puuid") in mine), None)
    return (m, line, pid, "api") if pid else None


LANE = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "MID": "MIDDLE"}


def from_lcu(lcu, game_id: int, my_puuid: str | None, my_role: str, games_of):
    game = lcu._get(f"/lol-match-history/v1/games/{game_id}", 10)
    tl = lcu._get(f"/lol-match-history/v1/game-timelines/{game_id}", 10)
    if not game or not tl:
        return None
    ident = {i.get("participantId"): (i.get("player") or {}).get("puuid") for i in game.get("participantIdentities") or []}
    people = []
    for p in game.get("participants") or []:
        t = p.get("timeline") or {}
        lane, role = (t.get("lane") or "").upper(), (t.get("role") or "").upper()
        pos = LANE.get(lane) or ("UTILITY" if "SUPPORT" in role else "BOTTOM" if lane in ("BOTTOM", "BOT") else "")
        people.append({"participantId": p["participantId"], "teamId": p["teamId"], "championId": p["championId"],
                       "win": bool((p.get("stats") or {}).get("win")), "teamPosition": pos})
    pid = next((k for k, v in ident.items() if v and v == my_puuid), None)
    if not pid:
        return None
    me = next(p for p in people if p["participantId"] == pid)
    if my_role:
        me["teamPosition"] = my_role
    for team in {p["teamId"] for p in people}:          # 자리가 비거나 겹치면 챔피언으로 다시 추정
        group = [p for p in people if p["teamId"] == team]
        seen = [p["teamPosition"] for p in group]
        if sorted(seen) != sorted(POSITIONS):
            fixed = [me] if me in group and my_role else []
            rest = [p for p in group if p not in fixed]
            for p, gss in zip(rest, infer_positions([p["championId"] for p in rest], games_of,
                                                    taken=tuple(p["teamPosition"] for p in fixed))):
                p["teamPosition"] = gss["position"]
    frames = tl.get("frames") or (tl.get("info") or {}).get("frames") or []
    m = {"info": {"participants": people, "gameDuration": game.get("gameDuration", 0)}}
    return m, {"info": {"frames": frames}}, pid, "lcu"


def build(lcu, game_id: int, my_role: str, games_of, lang: str | None = None) -> dict:
    got = from_db(game_id) or from_api(game_id)
    if not got:
        me = (lcu._get("/lol-summoner/v1/current-summoner") or {}).get("puuid") if hasattr(lcu, "_get") else None
        got = from_lcu(lcu, game_id, me, my_role, games_of) if me else None
    if not got:
        return {"available": False, "gameId": game_id}
    m, line, pid, src = got
    champ = next(p["championId"] for p in m["info"]["participants"] if p["participantId"] == pid)
    r = feedback.analyze(m, line, pid, core_avg().get(champ), items(), lang)
    r.pop("all", None)
    r.pop("fights", None)
    dur = m["info"].get("gameDuration", 0)
    r["say"] = voice_ids(r)
    r.update(available=True, gameId=game_id, source=src, min=int(dur // 1000 // 60 if dur > 100_000 else dur // 60))
    return r


def voice_ids(r: dict) -> list[str]:
    """음성 한 줄 — 가장 큰 아쉬운 점 (없으면 잘한 점) + "자세한 건 화면에 적어 뒀어요." """
    pick = (r.get("bad") or r.get("good") or [None])[0]
    if not pick or not pick.get("voice"):
        return []
    k, v = pick["k"], pick["voice"]
    n = v[1] if len(v) > 1 else 0
    ids = {
        "obj_death": f"pg_obj_death{min(n, 3)}", "fight_lost": f"pg_fight_lost{min(n, 4)}",
        "gank": f"pg_gank{min(n, 3)}", "spot": f"pg_spot{min(n, 4)}", "obj_fight_low": "pg_obj_fight_low",
        "jg_away": "pg_jg_away", "lane15": "pg_lane15_" + ("good" if n > 0 else "bad"),
        "lv6": "pg_lv6_" + ("bad" if n > 0 else "good"), "ward_gap": f"pg_ward_gap{max(4, min(n, 10))}",
        "core": "pg_core_" + ("bad" if n > 0 else "good"), "fight_won": "pg_fight_won",
        "obj_fight_top": "pg_obj_fight_top", "ward_ok": "pg_ward_ok",
    }.get(k)
    return [ids, "pg_more"] if ids else []
