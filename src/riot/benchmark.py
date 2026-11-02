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
from src.riot.me import DB as ME_DB

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
"""
MY_SCHEMA = """
CREATE TABLE IF NOT EXISTS my_timeline (
    match_id TEXT PRIMARY KEY,
    frames   TEXT                       -- 참가자 번호 → 15분 숫자 (JSON). 15분 전에 끝났으면 NULL
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


def end_stats(p: dict, team_kills: int, minutes: float) -> dict:
    c = p.get("challenges", {})
    return {"gold": p.get("goldEarned", 0) / minutes,
            "cs": (p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)) / minutes,
            "dmg": p.get("totalDamageDealtToChampions", 0) / minutes,
            "taken": p.get("totalDamageTaken", 0) / minutes,
            "kp": (p.get("kills", 0) + p.get("assists", 0)) / max(team_kills, 1),
            "deaths": p.get("deaths", 0), "kills": p.get("kills", 0), "assists": p.get("assists", 0),
            "vision": p.get("visionScore", 0) / minutes, "ctrl": p.get("visionWardsBoughtInGame", 0),
            "solo": c.get("soloKills", 0)}


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
    have = {r[0] for r in conn.execute("SELECT match_id FROM my_timeline")}
    ids = [r[0] for r in conn.execute(
        "SELECT match_id FROM my_match WHERE queue = 420 AND duration_s >= ? ORDER BY created_ms DESC",
        (MIN_DURATION_S,)) if r[0] not in have]
    if limit:
        ids = ids[:limit]
    print(f"내 판 타임라인 {len(ids):,}건을 받습니다", flush=True)
    for i, match_id in enumerate(ids, 1):
        line = timeline(client, match_id)
        frames = at_minute(line) if line else None
        conn.execute("INSERT OR REPLACE INTO my_timeline VALUES (?, ?)",
                     (match_id, json.dumps(frames, separators=(",", ":")) if frames else None))
        if i % 20 == 0 or i == len(ids):
            conn.commit()
            print(f"  {i:,}/{len(ids):,}", flush=True)
    conn.commit()
    return len(ids)


def cmd_timelines(args) -> None:
    my_timelines(sqlite3.connect(ME_DB), RiotClient(wait_for_new_key=True), args.limit)


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


def cmd_table(_args) -> None:
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
                               "tiers": table, "champMin": CHAMP_MIN, **champ_table(conn)},
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
    for name, func in (("timelines", cmd_timelines), ("sample", cmd_sample), ("table", cmd_table), ("all", cmd_all)):
        p = sub.add_parser(name)
        p.add_argument("--per-tier", type=int, default=PER_TIER)
        p.add_argument("--limit", type=int, default=0, help="내 판을 이만큼만 (시험용)")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
