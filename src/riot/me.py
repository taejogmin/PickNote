"""내 경기 기록을 받는다.

    python -m src.riot.me fetch --riot-id "이름#태그"    # 계정마다 처음 한 번 — API가 주는 데까지 거슬러
    python -m src.riot.me update                          # 그 뒤로 새 경기만 (계정 전부)
    python -m src.riot.me stats                           # 무엇이 쌓였나

**받는 것까지다.** 무엇을 보여 줄지는 쌓인 것을 들여다본 뒤 정한다.

- **계정이 여럿이면 다 합친다**. 두 번째 계정도 `fetch`로 한 번 거슬러 받으면 `update`가 같이 챙긴다.
- 솔로랭크(420)와 자유랭크(440). 밴픽이 있는 판만 이 앱이 다루는 상황과 같다.
- **따로 된 파일 `data/me.db`** 에 둔다. 저티어 학습 데이터와 섞이지 않는다.
- 경기 원본 JSON을 통째로, 압축해서 둔다 — 무엇을 분석할지 아직 모르니 열을 미리 정하지 않는다.
- **다른 사람의 신원은 저장하지 않는다.** 상대 팀은 이름·태그·puuid를 지운다. 같은 팀은 이름·태그를 지우고
  puuid를 이 PC에만 있는 소금으로 해시해 남긴다 — 누구인지는 몰라도 "같은 사람과 다시 한 판"은 알 수 있다.

**처음 거슬러 받기는 수집기를 멈추고 돌린다.** 개발용 키 한도(2분에 95회)는 키 하나에 걸리는데
수집기가 이미 그 한도를 꽉 쓰고 있다. 같이 돌리면 둘 다 429를 맞고, 반복해서 넘기면 키가 막힐 수도 있다.
새 경기만 받는 `update`는 요청 몇 번이라 같이 돌아도 된다.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import secrets
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from src import region as rg
from src.i18n import L
from src.riot.client import RiotClient, StalePuuid

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "me.db"
QUEUES = (420, 440)                 # 솔로랭크 · 자유랭크
QUEUE_KO = {420: L("솔로랭크", "Solo/Duo"), 440: L("자유랭크", "Flex")}
PAGE = 100                          # 매치 ID 한 번에 최대
# 지울 것 — 다른 사람의 신원
IDENTITY = ("riotIdGameName", "riotIdTagline", "summonerName", "summonerId", "puuid", "profileIcon")

SCHEMA = """
CREATE TABLE IF NOT EXISTS me (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS my_match (
    match_id    TEXT PRIMARY KEY,
    queue       INTEGER NOT NULL,
    created_ms  INTEGER NOT NULL,
    duration_s  INTEGER NOT NULL,
    patch       TEXT NOT NULL,
    champion_id INTEGER NOT NULL,
    position    TEXT NOT NULL,          -- 비어 있을 수 있다 (예전 경기)
    win         INTEGER NOT NULL,
    raw         BLOB NOT NULL           -- 경기 원본 JSON (gzip). 다른 사람의 신원은 지웠다
);
CREATE INDEX IF NOT EXISTS my_match_time ON my_match(created_ms);
CREATE TABLE IF NOT EXISTS account (    -- 내 계정들
    puuid    TEXT PRIMARY KEY,
    riot_id  TEXT NOT NULL,
    added_ms INTEGER NOT NULL,
    server   TEXT NOT NULL DEFAULT 'kr'  -- 어느 서버 계정인가 (src/region.py)
);
"""


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn) -> None:
    """계정 하나만 받던 때의 파일을 계정 여럿 꼴로 바꾼다. 그때 받은 경기는 전부 그 한 계정 것이다."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(my_match)")}
    if "account" not in cols:
        conn.execute("ALTER TABLE my_match ADD COLUMN account TEXT")
    if "server" not in {r[1] for r in conn.execute("PRAGMA table_info(account)")}:
        conn.execute("ALTER TABLE account ADD COLUMN server TEXT NOT NULL DEFAULT 'kr'")   # 서버를 고르기 전 계정은 한국
    first = get_me(conn, "puuid")
    if first:
        conn.execute("INSERT OR IGNORE INTO account (puuid, riot_id, added_ms) VALUES (?, ?, ?)",
                     (first, get_me(conn, "riot_id") or "", int(time.time() * 1000)))
        conn.execute("UPDATE my_match SET account = ? WHERE account IS NULL", (first,))
    conn.commit()


def accounts(conn) -> list[tuple[str, str]]:
    """(puuid, Riot ID), 먼저 등록한 것부터."""
    return conn.execute("SELECT puuid, riot_id FROM account ORDER BY added_ms").fetchall()


def account_servers(conn) -> dict[str, str]:
    """puuid → 서버."""
    return dict(conn.execute("SELECT puuid, server FROM account"))


def rekey(conn, old: str, new: str) -> None:
    """같은 계정의 puuid를 바꾼다. puuid는 API 키(앱)마다 달라서 키를 바꾸면 새로 받아야 한다."""
    if old == new:
        return
    if conn.execute("SELECT 1 FROM account WHERE puuid = ?", (old,)).fetchone():
        conn.execute("DELETE FROM account WHERE puuid = ?", (new,))
        conn.execute("UPDATE account SET puuid = ? WHERE puuid = ?", (new, old))
    conn.execute("UPDATE my_match SET account = ? WHERE account = ?", (new, old))
    if get_me(conn, "puuid") == old:
        set_me(conn, "puuid", new)
    conn.commit()


def refresh_puuid(conn, client: RiotClient, puuid: str, riot_id: str) -> str:
    """예전 키로 받은 puuid를 Riot ID로 다시 찾아 바꾼다."""
    name, _, tag = riot_id.rpartition("#")
    account = client.account_by_riot_id(name, tag) if name else None
    if not account or not account.get("puuid"):
        raise SystemExit(L(f"{riot_id} 계정을 다시 찾지 못했습니다. Riot ID가 바뀌었다면 my_accounts.txt 를 고쳐 주세요.",
                           f"Couldn't look up {riot_id} again. If the Riot ID changed, fix it in my_accounts.txt."))
    print(L("  API 키가 바뀌어 계정 ID를 새로 받았습니다.", "  The API key changed, so the account ID was looked up again."))
    rekey(conn, puuid, account["puuid"])
    return account["puuid"]


def get_me(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM me WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_me(conn, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO me(key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def collector_running() -> bool:
    """2_Collect.cmd의 중복 실행 막기와 같은 질의."""
    query = (
        "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Where-Object { $_.CommandLine -like '*src.riot.collect run*' }) { exit 1 }"
    )
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                              capture_output=True, timeout=30)
        return done.returncode == 1
    except Exception:
        return False


def scrub(match: dict, my_puuid: str, salt: str) -> tuple[dict, dict]:
    """다른 사람의 신원을 지운 경기와, 내 참가자 행을 돌려준다."""
    info = match.get("info", {})
    people = info.get("participants", [])
    mine = next((p for p in people if p.get("puuid") == my_puuid), None)
    if mine is None:
        raise ValueError("이 경기에 내가 없다")
    my_team = mine.get("teamId")

    mates = {}
    for p in people:
        if p is mine:
            continue
        puuid = p.get("puuid", "")
        if p.get("teamId") == my_team and puuid:
            # 누구인지는 모르지만 같은 사람이면 같은 값 — 듀오 분석용
            p["mate"] = hashlib.sha256((salt + puuid).encode()).hexdigest()[:16]
            mates[puuid] = p["mate"]
        for k in IDENTITY:
            p.pop(k, None)

    meta = match.get("metadata", {})
    if "participants" in meta:
        meta["participants"] = [
            ("me" if x == my_puuid else mates.get(x)) for x in meta["participants"]
        ]
    return match, mine


def save(conn, match_id: str, queue: int, match: dict, mine: dict, account: str) -> None:
    info = match["info"]
    version = info.get("gameVersion", "")
    patch = ".".join(version.split(".")[:2])
    raw = gzip.compress(json.dumps(match, ensure_ascii=False, separators=(",", ":")).encode())
    conn.execute(
        "INSERT OR REPLACE INTO my_match (match_id, queue, created_ms, duration_s, patch, champion_id, "
        "position, win, raw, account) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (match_id, queue, int(info.get("gameCreation", 0)), int(info.get("gameDuration", 0)),
         patch, int(mine.get("championId", 0)), mine.get("teamPosition") or "",
         int(bool(mine.get("win"))), raw, account),
    )


def all_ids(client: RiotClient, puuid: str, queue: int, since_s: int | None = None) -> list[str]:
    """API가 주는 데까지 거슬러 매치 ID를 모은다 (100개씩 넘긴다)."""
    out: list[str] = []
    start = 0
    while True:
        page = client.match_ids(puuid, queue=queue, count=PAGE, start=start, start_time=since_s)
        if not page:
            break
        out.extend(page)
        print(L(f"  {QUEUE_KO[queue]} 매치 ID {len(out):,}개", f"  {QUEUE_KO[queue]} match IDs {len(out):,}"), end="\r", flush=True)
        if len(page) < PAGE:
            break
        start += PAGE
    print()
    return out


def pull(conn, client: RiotClient, puuid: str, salt: str, since_s: int | None) -> int:
    have = {r[0] for r in conn.execute("SELECT match_id FROM my_match")}
    saved = 0
    for queue in QUEUES:
        ids = [m for m in all_ids(client, puuid, queue, since_s) if m not in have]
        print(L(f"  {QUEUE_KO[queue]} 새로 받을 경기 {len(ids):,}개", f"  {QUEUE_KO[queue]} new games to download: {len(ids):,}"))
        for i, match_id in enumerate(ids, 1):
            match = client.match(match_id)
            if not match:
                continue
            try:
                match, mine = scrub(match, puuid, salt)
            except ValueError:
                continue
            save(conn, match_id, queue, match, mine, puuid)
            saved += 1
            if i % 20 == 0 or i == len(ids):
                conn.commit()
                print(f"    {i:,}/{len(ids):,}", end="\r", flush=True)
        conn.commit()
        print()
    return saved


def cmd_fetch(args) -> None:
    if "#" not in args.riot_id:
        raise SystemExit(L('Riot ID는 "이름#태그" 꼴로 주세요.', 'Give the Riot ID as "Name#Tag".'))
    if collector_running() and not args.force:
        raise SystemExit(L(
            "수집기가 돌고 있습니다. 처음 거슬러 받기는 수천 번 요청이라 키 한도를 같이 쓰면 둘 다 막힙니다.\n"
            "수집기 창을 잠시 닫고 다시 실행해 주세요. 끝나면 2_Collect.cmd 로 다시 켜면 이어서 받습니다.",
            "The collector is running. The first full download takes thousands of requests and would share the key's rate limit.\n"
            "Close the collector window and try again. Afterwards, 2_Collect.cmd picks up where it left off.",
        ))
    name, tag = args.riot_id.rsplit("#", 1)
    conn = connect()
    server = rg.pick(getattr(args, "server", None))
    client = RiotClient(server=server)
    account = client.account_by_riot_id(name.strip(), tag.strip())
    if not account or not account.get("puuid"):
        raise SystemExit(L(f"{args.riot_id} 계정을 찾지 못했습니다. 이름과 태그를 확인해 주세요.",
                           f"Couldn't find the account {args.riot_id}. Check the name and tag."))
    puuid = account["puuid"]
    riot_id = f"{account.get('gameName', name)}#{account.get('tagLine', tag)}"
    for old, known in accounts(conn):            # 같은 계정을 예전 키로 받아 둔 적이 있으면 이어 붙인다
        if known.lower() == riot_id.lower():
            rekey(conn, old, puuid)
    if not get_me(conn, "puuid"):                 # 첫 계정 — 화면 맨 위 이름
        set_me(conn, "puuid", puuid)
        set_me(conn, "riot_id", riot_id)
    conn.execute("INSERT OR IGNORE INTO account (puuid, riot_id, added_ms, server) VALUES (?, ?, ?, ?)",
                 (puuid, riot_id, int(time.time() * 1000), server))
    conn.execute("UPDATE account SET riot_id = ?, server = ? WHERE puuid = ?", (riot_id, server, puuid))
    conn.commit()
    # 소금은 계정과 상관없이 하나 — 두 계정에서 같은 사람과 한 판이 같은 "동료"로 묶인다
    if not get_me(conn, "salt"):
        set_me(conn, "salt", secrets.token_hex(16))
    print(L(f"{riot_id} 의 경기를 받습니다 — {', '.join(QUEUE_KO[q] for q in QUEUES)}",
            f"Downloading games for {riot_id} — {', '.join(QUEUE_KO[q] for q in QUEUES)}"))
    start = time.time()
    n = pull(conn, client, puuid, get_me(conn, "salt"), since_s=None)
    print(L(f"\n{n:,}경기 저장 ({(time.time() - start) / 60:.0f}분) → {DB}",
            f"\nSaved {n:,} games ({(time.time() - start) / 60:.0f} min) → {DB}"))
    print_stats(conn)


def cmd_update(_args) -> None:
    conn = connect()
    accs = accounts(conn)
    if not accs:
        raise SystemExit(L("등록된 계정이 없습니다. 4_MyRecord.cmd 로 Riot ID를 적어 주세요.",
                           "No accounts registered yet. Add your Riot ID with 4_MyRecord.cmd."))
    servers = account_servers(conn)
    clients = {}                       # 서버마다 하나 — 매치 ID 목록을 그 서버 지역에 묻는다
    for puuid, riot_id in accs:
        server = servers.get(puuid, rg.DEFAULT)
        client = clients.setdefault(server, RiotClient(server=server))
        last = conn.execute("SELECT MAX(created_ms) FROM my_match WHERE account = ?", (puuid,)).fetchone()[0]
        since = int(last / 1000) - 3600 if last else None     # 겹치게 한 시간 앞에서부터
        print(riot_id)
        try:
            n = pull(conn, client, puuid, get_me(conn, "salt"), since_s=since)
        except StalePuuid:
            puuid = refresh_puuid(conn, client, puuid, riot_id)
            n = pull(conn, client, puuid, get_me(conn, "salt"), since_s=since)
        print(L(f"  새 경기 {n:,}개 저장", f"  Saved {n:,} new games"))
    # 승패 육각형의 라인전 축 — 새 판 수만큼이라 수집기와 같이 돌아도 된다
    from src.riot.benchmark import my_timelines
    my_timelines(conn, client)         # 타임라인은 매치 ID로 서버를 알아서 어느 client든 된다


def print_stats(conn) -> None:
    total = conn.execute("SELECT COUNT(*) FROM my_match").fetchone()[0]
    if not total:
        print(L("저장된 경기가 없습니다.", "No games saved yet."))
        return
    lo, hi = conn.execute("SELECT MIN(created_ms), MAX(created_ms) FROM my_match").fetchone()
    fmt = lambda ms: time.strftime("%Y-%m-%d", time.localtime(ms / 1000))
    print(f"\n{' + '.join(r for _p, r in accounts(conn))} · {total:,}{L('경기', ' games')} · {fmt(lo)} ~ {fmt(hi)}")
    for puuid, riot_id in accounts(conn):
        n, w, a, b = conn.execute("SELECT COUNT(*), SUM(win), MIN(created_ms), MAX(created_ms) "
                                  "FROM my_match WHERE account = ?", (puuid,)).fetchone()
        if n:
            print(L(f"  {riot_id:<20} {n:>6,}경기 · 승률 {w / n * 100:.1f}% · {fmt(a)} ~ {fmt(b)}",
                    f"  {riot_id:<20} {n:>6,} games · win rate {w / n * 100:.1f}% · {fmt(a)} ~ {fmt(b)}"))
    for q, n, w in conn.execute("SELECT queue, COUNT(*), SUM(win) FROM my_match GROUP BY queue"):
        print(L(f"  {QUEUE_KO.get(q, q):<6} {n:>6,}경기 · 승률 {w / n * 100:.1f}%",
                f"  {QUEUE_KO.get(q, q):<9} {n:>6,} games · win rate {w / n * 100:.1f}%"))
    names = {}
    try:
        names = {int(k): v[L("name", "en")] for k, v in json.loads(
            (ROOT / "data" / "champions.json").read_text(encoding="utf-8"))["champions"].items()}
    except Exception:
        pass
    print(L("  가장 많이 한 챔피언", "  Most played"))
    for cid, pos, n, w in conn.execute(
            "SELECT champion_id, position, COUNT(*) c, SUM(win) FROM my_match "
            "GROUP BY champion_id, position ORDER BY c DESC LIMIT 10"):
        print(f"    {names.get(cid, cid):<12} {pos or '-':<8} {n:>5}{L('경기', ' games')} · {w / n * 100:5.1f}%")


def cmd_stats(_args) -> None:
    print_stats(connect())


def main() -> None:
    parser = argparse.ArgumentParser(description="내 경기 기록 받기")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="처음 한 번 — API가 주는 데까지 거슬러 받는다")
    f.add_argument("--riot-id", required=True, help='"이름#태그"')
    f.add_argument("--server", default=None, help="kr · na1 · euw1 … (기본: 설정한 서버)")
    f.add_argument("--force", action="store_true", help="수집기가 돌아도 받는다 (권하지 않는다)")
    f.set_defaults(func=cmd_fetch)
    sub.add_parser("update", help="새 경기만").set_defaults(func=cmd_update)
    sub.add_parser("stats", help="무엇이 쌓였나").set_defaults(func=cmd_stats)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
