"""승패 육각형의 기준 데이터를 한 번 받는다.

    python -m src.riot.benchmark timelines            # 내 판 타임라인 → me.db의 my_timeline
    python -m src.riot.benchmark sample               # 티어 다섯 × 1,000판 → data/benchmark.db
    python -m src.riot.benchmark table                # 표 만들기 → models/benchmark.json
    python -m src.riot.benchmark all                  # 위 셋을 차례로

**수집기를 멈추고 돌린다.** 키 한도는 키 하나에 걸린다. 끊겨도 다시 실행하면 이어서 받는다.

타임라인은 크고 다른 사람의 식별값이 들어 있어서 **숫자만 뽑고 원본은 버린다.**
표본도 마찬가지다 — puuid·이름은 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np

from src.region import db_path, region_of_match
from src.riot.client import RiotClient
from src.riot.me import DB as ME_DB, IDENTITY

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
BENCH_DB = ROOT / "data" / "benchmark.db"
MATCHES_DB = db_path()
OUT = ROOT / "models" / "benchmark.json"
TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM")
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
PATCH = "16.21"
PER_TIER = 1000
MINUTE = 15
MIN_DURATION_S = MINUTE * 60
SEED = 20261102
# 표에 쓰는 값. 앞선 비율은 클수록 앞선 것이고, 데스만 적을수록 앞선 것이다.
STATS = ("gold15", "cs15", "level15", "kills15", "gold", "cs", "dmg", "taken", "kp", "deaths", "kills",
         "assists", "kda", "vision", "ctrl", "solo")
LOWER_IS_BETTER = {"deaths"}
CHAMP_MIN = 100          # 챔피언 회색선은 다섯 티어 합쳐 이만큼 쌍이 있을 때만 — 회색선 자체 ±10%p

SCHEMA = """
CREATE TABLE IF NOT EXISTS person (
    match_id  TEXT NOT NULL,
    tier      TEXT NOT NULL,
    position  TEXT NOT NULL,
    team      INTEGER NOT NULL,
    win       INTEGER NOT NULL,
    champion  INTEGER NOT NULL,
    stats     TEXT NOT NULL,            -- STATS 값들 (JSON)
    PRIMARY KEY (match_id, team, position)
);
CREATE TABLE IF NOT EXISTS seen (match_id TEXT PRIMARY KEY);   -- 받았거나 못 쓰는 판
CREATE TABLE IF NOT EXISTS curve (match_id TEXT PRIMARY KEY, curve TEXT);   -- 표본 판의 5분마다 숫자 (시간대별 격차)
CREATE TABLE IF NOT EXISTS raw_timeline (match_id TEXT PRIMARY KEY, raw BLOB NOT NULL);   -- 표본 타임라인 원본 (gzip, puuid 지움)
CREATE TABLE IF NOT EXISTS raw_match (match_id TEXT PRIMARY KEY, raw BLOB NOT NULL);      -- 표본 경기 원본 (gzip, 신원 지움)
"""
MY_SCHEMA = """
CREATE TABLE IF NOT EXISTS my_timeline (
    match_id TEXT PRIMARY KEY,
    frames   TEXT                       -- 참가자 번호 → 15분 숫자 (JSON). 15분 전에 끝났으면 NULL
);
CREATE TABLE IF NOT EXISTS my_timeline_raw (
    match_id TEXT PRIMARY KEY,
    raw      BLOB NOT NULL              -- 타임라인 원본 JSON (gzip). 다른 사람의 puuid는 지웠다
);
CREATE TABLE IF NOT EXISTS my_curve (
    match_id TEXT PRIMARY KEY,
    curve    TEXT                       -- 참가자 번호 → {cs, gold, xp: [5분, 10분, …]} (JSON). 시간대별 격차
);
"""


def timeline(client: RiotClient, match_id: str, region: str | None = None):
    return client.get(f"https://{region or region_of_match(match_id)}.api.riotgames.com"
                      f"/lol/match/v5/matches/{match_id}/timeline")


def at_minute(line: dict) -> dict[int, dict] | None:
    """참가자 번호 → 15분 시점 숫자. 그때까지 안 간 판이면 None."""
    frames = line.get("info", {}).get("frames", [])
    if len(frames) <= MINUTE:
        return None
    out = {}
    for key, p in frames[MINUTE]["participantFrames"].items():
        out[int(key)] = {"gold15": p.get("totalGold", 0), "level15": p.get("level", 0), "xp15": p.get("xp", 0),
                         "cs15": p.get("minionsKilled", 0) + p.get("jungleMinionsKilled", 0),
                         "kills15": 0, "deaths15": 0, "assists15": 0}
    limit = MINUTE * 60_000
    for frame in frames[:MINUTE + 1]:
        for e in frame.get("events", []):
            if e.get("type") != "CHAMPION_KILL" or e.get("timestamp", 0) > limit:
                continue
            for pid, key in ((e.get("killerId"), "kills15"), (e.get("victimId"), "deaths15")):
                if pid in out:
                    out[pid][key] += 1
            for pid in e.get("assistingParticipantIds") or []:
                if pid in out:
                    out[pid]["assists15"] += 1
    return out


def pack(obj) -> bytes:
    return gzip.compress(json.dumps(obj, separators=(",", ":")).encode())


def scrub_timeline(line: dict, label=lambda puuid: None) -> dict:
    """타임라인의 puuid를 지운다. label(puuid)가 값을 주면(내 판의 "me"·같은 팀 해시) 그걸로 바꾼다."""
    meta = line.get("metadata", {})
    if "participants" in meta:
        meta["participants"] = [label(x) for x in meta["participants"]]
    for p in line.get("info", {}).get("participants", []) or []:
        if "puuid" in p:
            p["puuid"] = label(p["puuid"])
    return line


def scrub_match(match: dict) -> dict:
    """표본 경기 — 열 명 모두 신원을 지운다 (표본에는 내가 없다)."""
    for p in match.get("info", {}).get("participants", []):
        for k in IDENTITY:
            p.pop(k, None)
    meta = match.get("metadata", {})
    if "participants" in meta:
        meta["participants"] = [None] * len(meta["participants"])
    return match


def my_labeler(conn):
    """내 판 타임라인용 이름표를 만드는 함수 — label_for(match_id)(puuid).
    내 계정은 "me", 같은 팀은 경기 원본에 이미 있는 해시(소금 + puuid)와 같을 때만 그 해시, 상대 팀은 지운다."""
    mine = {r[0] for r in conn.execute("SELECT puuid FROM account")}
    row = conn.execute("SELECT value FROM me WHERE key = 'salt'").fetchone()
    salt = row[0] if row else ""

    def label_for(match_id: str):
        row = conn.execute("SELECT raw FROM my_match WHERE match_id = ?", (match_id,)).fetchone()
        mates = set()
        if row:
            info = json.loads(gzip.decompress(row[0])).get("info", {})
            mates = {p["mate"] for p in info.get("participants", []) if p.get("mate")}

        def label(puuid):
            if not puuid:
                return None
            if puuid in mine:
                return "me"
            h = hashlib.sha256((salt + puuid).encode()).hexdigest()[:16]
            return h if h in mates else None
        return label
    return label_for


CURVE_MINUTES = (5, 10, 15, 20, 25, 30)


def curve_of(line: dict) -> dict[str, dict] | None:
    """참가자 번호 → {"cs": [...], "gold": [...], "xp": [...]} — 5분마다, 그 판이 간 데까지."""
    frames = line.get("info", {}).get("frames", [])
    mins = [m for m in CURVE_MINUTES if m < len(frames)]
    if not mins:
        return None
    out: dict = {}
    for m in mins:
        for key, p in frames[m]["participantFrames"].items():
            row = out.setdefault(key, {"cs": [], "gold": [], "xp": []})
            row["cs"].append(p.get("minionsKilled", 0) + p.get("jungleMinionsKilled", 0))
            row["gold"].append(p.get("totalGold", 0))
            row["xp"].append(p.get("xp", 0))
    return out


def end_stats(p: dict, team_kills: int, minutes: float) -> dict:
    c = p.get("challenges", {})
    return {"gold": p.get("goldEarned", 0) / minutes,
            "cs": (p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)) / minutes,
            "dmg": p.get("totalDamageDealtToChampions", 0) / minutes,
            "taken": p.get("totalDamageTaken", 0) / minutes,
            "kp": (p.get("kills", 0) + p.get("assists", 0)) / max(team_kills, 1),
            "deaths": p.get("deaths", 0), "kills": p.get("kills", 0), "assists": p.get("assists", 0),
            "vision": p.get("visionScore", 0) / minutes, "ctrl": p.get("visionWardsBoughtInGame", 0),
            "solo": c.get("soloKills", 0), "min": round(minutes, 3)}


def people(match: dict, line: dict) -> list[dict] | None:
    """판 하나를 자리별 숫자로. 다시하기·탈주로 다섯 자리가 안 찬 판은 쓰지 않는다."""
    info = match.get("info", {})
    if info.get("gameDuration", 0) < MIN_DURATION_S or info.get("queueId") != 420:
        return None
    frames = at_minute(line)
    if frames is None:
        return None
    rows = info.get("participants", [])
    minutes = info["gameDuration"] / 60
    kills = {t: sum(p.get("kills", 0) for p in rows if p.get("teamId") == t) for t in (100, 200)}
    out, seats = [], set()
    for p in rows:
        pos, team = p.get("teamPosition") or "", p.get("teamId")
        if pos not in POSITIONS or (team, pos) in seats:
            return None
        seats.add((team, pos))
        stats = end_stats(p, kills[team], minutes)
        stats.update(frames[p["participantId"]])
        out.append({"team": team, "position": pos, "win": int(bool(p.get("win"))),
                    "champion": int(p.get("championId", 0)), "stats": stats})
    return out if len(out) == 10 else None


def my_timelines(conn, client: RiotClient, limit: int = 0) -> int:
    """아직 없는 내 판 타임라인을 받는다. `python -m src.riot.me update`도 이걸 부른다."""
    conn.executescript(MY_SCHEMA)
    # 15분 넘은 판만 받으니 빈 칸은 받다가 실패한 것(키 만료·연결 끊김) — 다음에 다시 받는다
    have = {r[0] for r in conn.execute("SELECT match_id FROM my_timeline WHERE frames IS NOT NULL")}
    ids = [r[0] for r in conn.execute(
        "SELECT match_id FROM my_match WHERE queue = 420 AND duration_s >= ? ORDER BY created_ms DESC",
        (MIN_DURATION_S,)) if r[0] not in have]
    if limit:
        ids = ids[:limit]
    label_for = my_labeler(conn)
    print(f"내 판 타임라인 {len(ids):,}건을 받습니다", flush=True)
    for i, match_id in enumerate(ids, 1):
        line = timeline(client, match_id)
        frames = at_minute(line) if line else None
        conn.execute("INSERT OR REPLACE INTO my_timeline VALUES (?, ?)",
                     (match_id, json.dumps(frames, separators=(",", ":")) if frames else None))
        cv = curve_of(line) if line else None
        if cv:
            conn.execute("INSERT OR REPLACE INTO my_curve VALUES (?, ?)", (match_id, json.dumps(cv, separators=(",", ":"))))
        if line:
            conn.execute("INSERT OR REPLACE INTO my_timeline_raw VALUES (?, ?)", (match_id, pack(scrub_timeline(line, label_for(match_id)))))
        conn.commit()            # 판마다 — 위와 같은 까닭
        if i % 20 == 0 or i == len(ids):
            print(f"  {i:,}/{len(ids):,}", flush=True)
    conn.commit()
    return len(ids)


def cmd_timelines(args) -> None:
    my_timelines(sqlite3.connect(ME_DB), RiotClient(wait_for_new_key=True), args.limit)


def backfill_curves(client: RiotClient, limit: int = 0, sample: bool = True, conn=None) -> None:
    """타임라인 원본이 없는 판을 다시 받아 원본과 5분마다 숫자를 둔다. 내 판(me.db) 다음 표본(benchmark.db). 끊겨도 이어 받는다.
    내 기록 갱신은 sample=False로 내 판만 — 처음 한 번만 예전 판 수만큼 요청하고, 그 뒤로는 새 판에 같이 받아 0건이다."""
    jobs = []
    me = conn or sqlite3.connect(ME_DB)
    me.executescript(MY_SCHEMA)
    have = {r[0] for r in me.execute("SELECT match_id FROM my_timeline_raw")}
    jobs.append(("내 판", me, "my", [r[0] for r in me.execute(
        "SELECT match_id FROM my_match WHERE queue = 420 AND duration_s >= ? ORDER BY created_ms DESC",
        (MIN_DURATION_S,)) if r[0] not in have]))
    if sample and BENCH_DB.exists():
        bench = sqlite3.connect(BENCH_DB)
        bench.executescript(SCHEMA)
        have = {r[0] for r in bench.execute("SELECT match_id FROM raw_timeline")}
        jobs.append(("표본", bench, "sample", [r[0] for r in bench.execute(
            "SELECT DISTINCT match_id FROM person ORDER BY match_id") if r[0] not in have]))
    for name, conn, table, ids in jobs:
        if limit:
            ids = ids[:limit]
        if not ids:
            continue
        print(f"{name} 타임라인 {len(ids):,}건을 통째로 받습니다", flush=True)
        start = time.time()
        label_for = my_labeler(conn) if table == "my" else (lambda _m: (lambda puuid: None))
        for i, match_id in enumerate(ids, 1):
            line = timeline(client, match_id)
            cv = curve_of(line) if line else None
            conn.execute(f"INSERT OR REPLACE INTO {'my_curve' if table == 'my' else 'curve'} VALUES (?, ?)",
                         (match_id, json.dumps(cv, separators=(",", ":")) if cv else None))
            if line:
                conn.execute(f"INSERT OR REPLACE INTO {'my_timeline_raw' if table == 'my' else 'raw_timeline'} VALUES (?, ?)",
                             (match_id, pack(scrub_timeline(line, label_for(match_id)))))
            conn.commit()        # 판마다 — 원본이 커서 모아 두면 쓰기 잠금이 길어져 화면이 내 기록을 못 읽는다
            if i % 50 == 0 or i == len(ids):
                left = (len(ids) - i) * (time.time() - start) / i / 60
                print(f"  {name} {i:,}/{len(ids):,} · 남은 약 {left:.0f}분", flush=True)
        conn.commit()


def cmd_curves(args) -> None:
    backfill_curves(RiotClient(wait_for_new_key=True), args.limit)


def candidates() -> dict[str, list[str]]:
    src = sqlite3.connect(f"file:{quote(MATCHES_DB.as_posix())}?mode=ro", uri=True)
    out = {}
    for tier in TIERS:
        ids = sorted(r[0] for r in src.execute(
            "SELECT match_id FROM match WHERE tier = ? AND patch = ? AND duration_s >= ?",
            (tier, PATCH, MIN_DURATION_S)))
        random.Random(f"{SEED}-{tier}").shuffle(ids)
        out[tier] = ids
    src.close()
    return out


def cmd_sample(args) -> None:
    conn = sqlite3.connect(BENCH_DB)
    conn.executescript(SCHEMA)
    pool = candidates()
    seen = {r[0] for r in conn.execute("SELECT match_id FROM seen")}
    done = dict(conn.execute("SELECT tier, COUNT(DISTINCT match_id) FROM person GROUP BY tier"))
    client = RiotClient(wait_for_new_key=True)
    next_at = dict.fromkeys(TIERS, 0)
    start = time.time()
    got = 0
    # 티어를 돌아가며 한 판씩 — 중간에 끊겨도 어느 티어나 비슷하게 차 있다
    while any(done.get(t, 0) < args.per_tier for t in TIERS):
        for tier in TIERS:
            if done.get(tier, 0) >= args.per_tier:
                continue
            ids = pool[tier]
            while next_at[tier] < len(ids) and ids[next_at[tier]] in seen:
                next_at[tier] += 1
            if next_at[tier] >= len(ids):
                raise SystemExit(f"{tier} 후보가 모자랍니다")
            match_id = ids[next_at[tier]]
            seen.add(match_id)
            match = client.match(match_id)
            line = timeline(client, match_id) if match else None
            rows = people(match, line) if match and line else None
            conn.execute("INSERT OR IGNORE INTO seen VALUES (?)", (match_id,))
            if rows:
                cv = curve_of(line)
                conn.execute("INSERT OR REPLACE INTO curve VALUES (?, ?)", (match_id, json.dumps(cv, separators=(",", ":")) if cv else None))
                conn.execute("INSERT OR REPLACE INTO raw_timeline VALUES (?, ?)", (match_id, pack(scrub_timeline(line))))
                conn.execute("INSERT OR REPLACE INTO raw_match VALUES (?, ?)", (match_id, pack(scrub_match(match))))
                conn.executemany(
                    "INSERT OR REPLACE INTO person VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(match_id, tier, r["position"], r["team"], r["win"], r["champion"],
                      json.dumps(r["stats"], separators=(",", ":"))) for r in rows])
                done[tier] = done.get(tier, 0) + 1
                got += 1
            conn.commit()
            if got and got % 50 == 0 and rows:
                left = sum(max(0, args.per_tier - done.get(t, 0)) for t in TIERS)
                rate = got / max(time.time() - start, 1)
                print(f"  {' '.join(f'{t[:2]} {done.get(t, 0)}' for t in TIERS)} · 남은 약 {left / rate / 60:.0f}분", flush=True)


def lead(a: np.ndarray, b: np.ndarray) -> float:
    return float(100 * (np.mean(a > b) + 0.5 * np.mean(a == b)))


def champ_table(conn) -> dict:
    """챔피언 회색선. 다섯 티어를 합쳐 챔피언이 맞은편보다 앞선 비율을 결과마다 잰다.
    화면은 자리·티어 기준에 (챔피언 − 자리 합) 차이만 더한다 — 그래서 자리 합(`posPooled`)도 같이 둔다."""
    seat: dict = {}
    for match_id, pos, team, win, champ, stats in conn.execute(
            "SELECT match_id, position, team, win, champion, stats FROM person"):
        s = json.loads(stats)
        s["kda"] = (s["kills"] + s["assists"]) / max(s["deaths"], 1)
        seat[match_id, pos, team] = (win, champ, s)
    sides = {"all": (0, 1), "win": (1,), "loss": (0,)}
    pairs: dict = defaultdict(lambda: defaultdict(list))     # (자리, 챔피언 또는 None) → 결과 → [(나, 맞은편)]
    for (match_id, pos, team), (win, champ, s) in seat.items():
        other = seat.get((match_id, pos, 300 - team))
        if other is None:
            continue
        for key in ((pos, champ), (pos, None)):
            pairs[key][win].append((s, other[2]))

    def side_lead(rows) -> dict:
        out = {}
        for k in STATS:
            a = np.array([r[0][k] for r in rows], float)
            b = np.array([r[1][k] for r in rows], float)
            out[k] = round(lead(b, a) if k in LOWER_IS_BETTER else lead(a, b), 1)
        return out

    def block(by_win) -> dict:
        n = len(by_win[0]) + len(by_win[1])
        return {"pairs": n,
                "n": {side: sum(len(by_win[w]) for w in ws) for side, ws in sides.items()},   # 회색선 흔들림 폭에
                "lead": {side: side_lead([r for w in ws for r in by_win[w]]) for side, ws in sides.items()
                         if any(by_win[w] for w in ws)},
                "mean": {side: {k: round(float(np.mean([r[0][k] for w in ws for r in by_win[w]])), 3) for k in STATS}
                         for side, ws in sides.items() if any(by_win[w] for w in ws)}}

    champs: dict = defaultdict(dict)
    pooled = {}
    for (pos, champ), by_win in pairs.items():
        if champ is None:
            pooled[pos] = block(by_win)
        elif len(by_win[0]) + len(by_win[1]) >= CHAMP_MIN:
            champs[pos][str(champ)] = block(by_win)
    return {"posPooled": pooled, "champs": champs}


# ---------------------------------------------------------------- 한눈에 지표 (라인전 · 캐리력 · 인분 · 멘탈)
# mine.py의 _form()과 같은 식이어야 기준과 비교된다.

def lane_share(me: dict, foe: dict) -> float:
    """15분 골드 몫과 경험치 몫의 평균 (0~100). 50이 맞은편과 같음."""
    g = me["gold15"] / max(me["gold15"] + foe["gold15"], 1)
    x = me["xp15"] / max(me["xp15"] + foe["xp15"], 1)
    return 50 * (g + x)


def team_shares(s: dict, team: list[dict]) -> tuple[float, float, float]:
    """팀 안에서 딜 몫 · 골드 몫 · 킬 관여. team에는 나도 들어 있다."""
    return (s["dmg"] / max(sum(t["dmg"] for t in team), 1e-9),
            s["gold"] / max(sum(t["gold"] for t in team), 1e-9),
            s["kp"])


# 지표 칸 — 판 하나에서 한 사람의 값. s·team·foe는 people()의 stats 꼴(15분 값 포함), minutes는 경기 길이(분)
TILE_LOWER = {"deaths"}                       # 적을수록 좋은 것


def tile_values(s: dict, team: list[dict], foe: dict, minutes: float | None) -> dict:
    team_k15 = max(sum(t["kills15"] for t in team), 1)
    out = {
        "kda": (s["kills"] + s["assists"]) / max(s["deaths"], 1),
        "lane": lane_share(s, foe),
        "cs15": s["cs15"], "gold15": s["gold15"], "xp15": s["xp15"],
        "cs15d": s["cs15"] - foe["cs15"], "gold15d": s["gold15"] - foe["gold15"], "xp15d": s["xp15"] - foe["xp15"],
        "kp15": (s["kills15"] + s["assists15"]) / team_k15,
        "kp": s["kp"], "kills": s["kills"], "assists": s["assists"], "deaths": s["deaths"],
        "solo": s["solo"], "soloShare": s["solo"] / max(s["kills"], 1),
        "cs": s["cs"], "gold": s["gold"], "dmg": s["dmg"], "taken": s["taken"],
        "dmgPerGold": s["dmg"] / max(s["gold"], 1e-9),
        "dmgShare": s["dmg"] / max(sum(t["dmg"] for t in team), 1e-9),
        "vision": s["vision"], "ctrl": s["ctrl"],
    }
    if minutes:
        out["dmgPerDeath"] = s["dmg"] * minutes / max(s["deaths"], 1)
        out["takenPerDeath"] = s["taken"] * minutes / max(s["deaths"], 1)
    return out


def mates_share(team: list[dict], rivals: list[dict]) -> float:
    """팀운 — 내 자리를 뺀 팀원 넷과 상대 넷의 15분 골드 몫·경험치 몫 평균 (50이 대등)."""
    g = sum(t["gold15"] for t in team) / max(sum(t["gold15"] for t in team + rivals), 1)
    x = sum(t["xp15"] for t in team) / max(sum(t["xp15"] for t in team + rivals), 1)
    return 50 * (g + x)


def snowball(s: dict, foe: dict, win: int) -> int | None:
    """라인전을 이긴 판에서 끝까지 맞은편보다 골드가 많고 이겼나. 라인전 못 이긴 판은 None."""
    if lane_share(s, foe) <= 50:
        return None
    return int(bool(win) and s["gold"] > foe["gold"])


def _durations(ids: list[str]) -> dict[str, float]:
    """표본 경기 길이(분) — 예전 표본은 stats에 없어 수집 DB에서 찾는다."""
    if not MATCHES_DB.exists():
        return {}
    conn = sqlite3.connect(f"file:{quote(MATCHES_DB.as_posix())}?mode=ro", uri=True)
    try:
        out = {}
        for i in range(0, len(ids), 900):
            part = ids[i:i + 900]
            out.update({m: d / 60 for m, d in conn.execute(
                f"SELECT match_id, duration_s FROM match WHERE match_id IN ({','.join('?' * len(part))})", part)})
        return out
    finally:
        conn.close()



def form_table(conn) -> dict:
    """자리마다 팀 안 몫의 평균(인분의 1.0)과 기여도 분위수(캐리력), 티어·자리마다 멘탈 세 비율의 기준."""
    seats: dict = defaultdict(dict)
    for match_id, tier, pos, team, win, stats in conn.execute(
            "SELECT match_id, tier, position, team, win, stats FROM person"):
        seats[match_id][team, pos] = (tier, win, json.loads(stats))
    games = [g for g in seats.values() if len(g) == 10]
    shares: dict = defaultdict(list)
    for g in games:
        for (team, pos), (_t, _w, s) in g.items():
            shares[pos].append(team_shares(s, [v[2] for (t, _p), v in g.items() if t == team]))
    avg = {pos: [float(np.mean([r[k] for r in rows])) for k in range(3)] for pos, rows in shares.items()}
    contrib = lambda pos, sh: sum(sh[k] / avg[pos][k] for k in range(3)) / 3
    cq = {pos: [round(float(q), 4) for q in np.quantile([contrib(pos, r) for r in rows], np.linspace(0, 1, 101))]
          for pos, rows in shares.items()}

    rates: dict = defaultdict(lambda: defaultdict(lambda: {"lost": [0, 0], "won": [0, 0], "behind": [0, 0]}))
    for g in games:
        for (team, pos), (tier, win, s) in g.items():
            foe = g[300 - team, pos][2]
            lane = lane_share(s, foe)
            c = contrib(pos, team_shares(s, [v[2] for (t, _p), v in g.items() if t == team]))
            mates = sum(v[2]["gold15"] for (t, q), v in g.items() if t == team and q != pos)
            rivals = sum(v[2]["gold15"] for (t, q), v in g.items() if t != team and q != pos)
            r = rates[tier][pos]
            if lane < 50:                                   # 라인전 지고 이긴 판
                r["lost"][0] += 1; r["lost"][1] += win
            elif lane > 50:                                 # 라인전 이기고 진 판
                r["won"][0] += 1; r["won"][1] += 1 - win
            if mates < rivals:                              # 팀원 넷이 15분에 밀릴 때 내가 끝까지 잘해서 이긴 판
                r["behind"][0] += 1; r["behind"][1] += int(win and lane > 50 and c >= 1)
    tiers = {t: {pos: {k: [round(100 * w / n, 1) if n else None, n] for k, (n, w) in v.items()}
                 for pos, v in by.items()} for t, by in rates.items()}
    # 지표 칸 분위수 (티어 · 자리 · 지표마다 5% 간격 21개) · 스노우볼 기준
    minutes = _durations(list(seats))
    vals: dict = defaultdict(lambda: defaultdict(list))
    snow: dict = defaultdict(lambda: [0, 0])
    for match_id, g in seats.items():
        if len(g) != 10:
            continue
        for (team, pos), (tier, win, st) in g.items():
            foe = g[300 - team, pos][2]
            mine = [v[2] for (t, _p), v in g.items() if t == team]
            m = st.get("min") or minutes.get(match_id)
            row = tile_values(st, mine, foe, m)
            row["mates"] = mates_share([v[2] for (t, q), v in g.items() if t == team and q != pos],
                                       [v[2] for (t, q), v in g.items() if t != team and q != pos])
            for k, v in row.items():
                vals[tier, pos][k].append(v)
            sb = snowball(st, foe, win)
            if sb is not None:
                snow[tier, pos][0] += 1; snow[tier, pos][1] += sb
    grid = np.linspace(0, 1, 21)
    q: dict = defaultdict(dict)
    for (tier, pos), by in vals.items():
        q[tier][pos] = {k: [round(float(x), 4) for x in np.quantile(v, grid)] for k, v in by.items()}
    for (tier, pos), (n, w) in snow.items():
        tiers[tier][pos]["snow"] = [round(100 * w / n, 1) if n else None, n]
    return {"posShare": {pos: [round(a, 4) for a in v] for pos, v in avg.items()}, "contribQ": cq, "tiers": tiers,
            "q": q, "lower": sorted(TILE_LOWER)}



# ---------------------------------------------------------------- 시간대별 격차 (맞은편과 CS · 골드 · 경험치 차이, 5분마다)
CURVE_KEYS = ("cs", "gold", "xp")


def seat_curves(curve: dict, seats: dict) -> dict | None:
    """참가자 번호 → (팀, 자리). 표본에는 번호가 없어 15분 숫자(골드·경험치·CS)가 같은 사람으로 잇는다."""
    out = {}
    for pid, c in curve.items():
        if len(c["gold"]) < 3:
            return None
        key = (c["gold"][2], c["xp"][2], c["cs"][2])
        hit = [seat for seat, st in seats.items() if (st["gold15"], st["xp15"], st["cs15"]) == key]
        if len(hit) != 1:
            return None
        out[hit[0]] = c
    return out if len(out) == 10 else None


def curve_gaps(by_seat: dict, team: int, pos: str) -> dict:
    """그 자리의 나 − 맞은편, 5분마다 (그 판이 간 데까지)."""
    me, foe = by_seat[team, pos], by_seat[300 - team, pos]
    return {k: [a - b for a, b in zip(me[k], foe[k])] for k in CURVE_KEYS}


def curve_table(conn) -> dict:
    """티어·자리마다, 그리고 챔피언마다(다섯 티어 합) 5분마다 평균 격차 — 전체 · 이긴 판 · 진 판."""
    curves = dict(conn.execute("SELECT match_id, curve FROM curve WHERE curve IS NOT NULL"))
    seats: dict = defaultdict(dict)
    for match_id, tier, pos, team, win, champ, stats in conn.execute(
            "SELECT match_id, tier, position, team, win, champion, stats FROM person"):
        if match_id in curves:
            seats[match_id][team, pos] = (tier, win, champ, json.loads(stats))
    n_min = len(CURVE_MINUTES)
    acc = defaultdict(lambda: {side: {k: [[0.0, 0] for _ in range(n_min)] for k in CURVE_KEYS} for side in ("all", "win", "loss")})
    used = 0
    for match_id, g in seats.items():
        if len(g) != 10:
            continue
        by_seat = seat_curves(json.loads(curves[match_id]), {seat: v[3] for seat, v in g.items()})
        if by_seat is None:
            continue
        used += 1
        for (team, pos), (tier, win, champ, _st) in g.items():
            gaps = curve_gaps(by_seat, team, pos)
            for key in ((tier, pos), ("*", pos), ("champ", pos, champ)):
                for side in ("all", "win" if win else "loss"):
                    for k in CURVE_KEYS:
                        for i, v in enumerate(gaps[k]):
                            cell = acc[key][side][k][i]
                            cell[0] += v; cell[1] += 1

    def pack(block: dict, need: int = 1) -> dict:
        return {side: {**{k: [round(t / n, 2) if n >= need else None for t, n in cells] for k, cells in by.items()},
                       "n": [n for _t, n in by["cs"]]} for side, by in block.items()}

    tiers: dict = defaultdict(dict)
    pooled, champs = {}, defaultdict(dict)
    for key, block in acc.items():
        if key[0] == "*":
            pooled[key[1]] = pack(block)
        elif key[0] == "champ":
            if block["all"]["cs"][0][1] >= CHAMP_MIN:          # 5분 판 수가 기준 — 늦은 시간은 판마다 다시 거른다
                champs[key[1]][str(key[2])] = pack(block, need=CHAMP_MIN // 2)
        else:
            tiers[key[0]][key[1]] = pack(block)
    return {"minutes": list(CURVE_MINUTES), "games": used, "tiers": tiers, "posPooled": pooled, "champs": champs}



def cmd_table(_args) -> None:
    sqlite3.connect(BENCH_DB).executescript(SCHEMA)          # curve 표가 없던 예전 파일
    conn = sqlite3.connect(f"file:{BENCH_DB}?mode=ro", uri=True)
    pairs: dict = defaultdict(dict)
    means: dict = defaultdict(list)
    for match_id, tier, pos, win, stats in conn.execute("SELECT match_id, tier, position, win, stats FROM person"):
        s = json.loads(stats)
        s["kda"] = (s["kills"] + s["assists"]) / max(s["deaths"], 1)
        pairs[tier, pos, match_id][win] = s
        means[tier, pos, win].append([s[k] for k in STATS])
    table: dict = {}
    for tier in TIERS:
        table[tier] = {}
        for pos in POSITIONS:
            rows = [v for (t, q, _m), v in pairs.items() if t == tier and q == pos and 0 in v and 1 in v]
            if not rows:
                continue
            won = {}
            for k in STATS:
                a = np.array([r[1][k] for r in rows], float)
                b = np.array([r[0][k] for r in rows], float)
                won[k] = round(lead(b, a) if k in LOWER_IS_BETTER else lead(a, b), 1)
            # 표의 티어 평균 — 이긴 판·진 판 보기에서는 같은 결과의 사람끼리
            avg = {"all": np.mean(means[tier, pos, 1] + means[tier, pos, 0], axis=0),
                   "win": np.mean(means[tier, pos, 1], axis=0), "loss": np.mean(means[tier, pos, 0], axis=0)}
            table[tier][pos] = {"pairs": len(rows), "winLead": won,
                                "mean": {side: {k: round(float(v), 3) for k, v in zip(STATS, a)}
                                         for side, a in avg.items()}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"patch": PATCH, "minute": MINUTE, "made": time.strftime("%Y-%m-%d %H:%M"),
                               "tiers": table, "champMin": CHAMP_MIN, **champ_table(conn),
                               "form": form_table(conn), "curve": curve_table(conn)},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}")
    for tier in TIERS:
        for pos in ("BOTTOM", "UTILITY"):
            e = table[tier].get(pos)
            if e:
                w = e["winLead"]
                print(f"  {tier} {pos} {e['pairs']:,}쌍 · 15분 골드 {w['gold15']} · 골드 {w['gold']} · 딜 {w['dmg']}"
                      f" · 킬 관여 {w['kp']} · 생존 {w['deaths']} · 시야 {w['vision']}")


def cmd_all(args) -> None:
    cmd_timelines(args)
    cmd_sample(args)
    cmd_table(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="승패 육각형 기준 데이터")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("timelines", cmd_timelines), ("sample", cmd_sample), ("table", cmd_table), ("all", cmd_all),
                       ("curves", cmd_curves)):
        p = sub.add_parser(name)
        p.add_argument("--per-tier", type=int, default=PER_TIER)
        p.add_argument("--limit", type=int, default=0, help="내 판을 이만큼만 (시험용)")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
