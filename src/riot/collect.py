"""솔로랭크 아이언~플래티넘 매치 수집기. 서버는 설정한 것 (src/region.py, 기본 한국).

    python -m src.riot.collect seed          씨앗 소환사 확보
    python -m src.riot.collect run           씨앗 → 매치ID → 매치상세 (계속)
    python -m src.riot.collect stats         진행 상황

개발용 키 기준 하루 5~6만 경기가 상한이다. 언제 끊겨도 다시 `run`을 치면 이어서 받는다.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from src.data.store import connect, counts
from src.i18n import L
from src.riot.client import RiotClient, StalePuuid

TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM"]
# 고티어는 실험용. `run --tiers apex`로만 모은다.
APEX_TIERS = ["MASTER", "GRANDMASTER", "CHALLENGER"]


def parse_tiers(raw: str) -> list[str]:
    """'low' · 'apex' · 'SILVER,GOLD' 같은 목록"""
    key = (raw or "low").strip().lower()
    if key == "low":
        return list(TIERS)
    if key == "apex":
        return list(APEX_TIERS)
    tiers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in TIERS + APEX_TIERS]
    if unknown:
        raise SystemExit(L(f"모르는 티어: {unknown}", f"Unknown tier: {unknown}"))
    return tiers


def _in(tiers: list[str]) -> tuple[str, list[str]]:
    return f"tier IN ({','.join('?' * len(tiers))})", list(tiers)
DIVISIONS = ["IV", "III", "II", "I"]
POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}

# 리메이크와 초반 항복은 조합의 영향을 재기 어렵다. 15분 미만은 버린다.
MIN_DURATION_S = 900


# ---------------------------------------------------------------- 1단계: 씨앗

_last_pages: dict[tuple[str, str], int] = {}


def last_page(client: RiotClient, tier: str, division: str) -> int:
    """디비전 목록의 마지막 쪽. 이분 탐색이라 요청 12건쯤 — 켤 때 한 번만 찾는다."""
    key = (tier, division)
    if key not in _last_pages:
        lo, hi = 1, 4096
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if client.league_entries(tier, division, mid):
                lo = mid
            else:
                hi = mid - 1
        _last_pages[key] = lo
    return _last_pages[key]


def collect_seeds(client: RiotClient, conn, per_division: int, max_page: int) -> int:
    """티어·디비전별로 소환사를 모은다. 한 페이지에 최대 205명.

    페이지를 무작위로 고르는 이유: 1페이지만 계속 받으면 늘 같은 사람들이라
    표본이 한쪽으로 쏠린다. 1~40쪽만 보다가 그 안의 사람을 다 긁어 버려서
    마지막 쪽까지 연다. max_page가 0이면 끝까지.
    """
    added = 0
    for tier in TIERS:
        for division in DIVISIONS:
            got = 0
            end = last_page(client, tier, division)
            if max_page:
                end = min(end, max_page)
            pages = random.sample(range(1, end + 1), k=min(end, 12))
            for page in pages:
                if got >= per_division:
                    break
                entries = client.league_entries(tier, division, page)
                if not entries:
                    continue
                rows = []
                for entry in entries:
                    puuid = entry.get("puuid")
                    if not puuid:
                        continue  # summonerId만 오는 옛 응답은 건너뛴다
                    rows.append((puuid, tier, division))
                cur = conn.executemany(
                    "INSERT OR IGNORE INTO seed_player(puuid, tier, division) VALUES (?,?,?)",
                    rows,
                )
                conn.commit()
                added += cur.rowcount
                got += len(rows)
            print(f"  {tier:9s} {division:3s}  {got:4d}{L('명', ' players')}", flush=True)
    return added


def collect_apex_seeds(client: RiotClient, conn) -> int:
    """마스터·그랜드마스터·챌린저 전원을 씨앗으로 넣는다. 디비전이 없어 요청 3번이면 끝난다."""
    added = 0
    for tier in APEX_TIERS:
        rows = [(e["puuid"], tier, "I") for e in client.apex_league(tier) if e.get("puuid")]
        cur = conn.executemany(
            "INSERT OR IGNORE INTO seed_player(puuid, tier, division) VALUES (?,?,?)", rows
        )
        conn.commit()
        added += cur.rowcount
        print(f"  {tier:12s} {len(rows):5d}{L('명', ' players')}", flush=True)
    return added


# ---------------------------------------------------------------- 2단계: 매치 ID

def scrape_ids(client: RiotClient, conn, limit: int, days: int, per_player: int,
               tiers: list[str] = TIERS) -> int:
    """아직 안 긁은 씨앗에게서 최근 매치 ID를 모은다.

    요청 1건이 최대 100개의 매치 ID를 준다. 중복이 많아서 실제로는
    1건당 새 매치 30~70개 정도가 남는다. 그래서 매치 상세보다 훨씬 값싸다.
    """
    start_time = int(time.time()) - days * 86400
    # 무작위 순서로 꺼낸다. 넣은 순서(아이언→브론즈→…)대로 쓰면
    # 아이언을 다 훑기 전에는 골드·플래 경기가 한 건도 안 쌓인다.
    where, args = _in(tiers)
    seeds = conn.execute(
        f"SELECT puuid, tier FROM seed_player WHERE scraped_at IS NULL AND {where} "
        "ORDER BY RANDOM() LIMIT ?",
        (*args, limit),
    ).fetchall()

    new_ids = 0
    for seed in seeds:
        try:
            ids = client.match_ids(
                seed["puuid"], count=per_player, start_time=start_time
            )
        except StalePuuid:
            # 키를 바꾸면 남은 씨앗이 전부 못 쓰게 된다. 하나씩 실패시키지 말고
            # 한꺼번에 버린다. 다음 바퀴에 씨앗이 비었으니 새 키로 다시 받는다.
            print(L("\n씨앗이 예전 API 키로 받은 것이라 쓸 수 없습니다 — 버리고 새 키로 다시 모읍니다.",
                    "\nThese seed players came from an old API key and can't be used — dropping them and fetching new ones."), flush=True)
            conn.execute("UPDATE seed_player SET scraped_at = ? WHERE scraped_at IS NULL", (int(time.time()),))
            conn.commit()
            return new_ids
        if ids:
            cur = conn.executemany(
                "INSERT OR IGNORE INTO match_queue(match_id, tier) VALUES (?,?)",
                [(mid, seed["tier"]) for mid in ids],
            )
            new_ids += cur.rowcount
        conn.execute(
            "UPDATE seed_player SET scraped_at = ? WHERE puuid = ?",
            (int(time.time()), seed["puuid"]),
        )
        conn.commit()
    return new_ids


# ---------------------------------------------------------------- 3단계: 매치 상세

def parse_perks(participant: dict):
    """룬을 (핵심 룬, 주 계열, 보조 계열, 선택한 룬 전체)로 편다."""
    styles = ((participant.get("perks") or {}).get("styles")) or []
    primary = next((s for s in styles if s.get("description") == "primaryStyle"), None)
    sub = next((s for s in styles if s.get("description") == "subStyle"), None)
    chosen = [
        sel.get("perk")
        for style in styles
        for sel in (style.get("selections") or [])
        if sel.get("perk")
    ]
    keystone = (primary or {}).get("selections", [{}])[0].get("perk") if primary else None
    return (
        keystone,
        (primary or {}).get("style"),
        (sub or {}).get("style"),
        ",".join(str(p) for p in chosen) or None,
    )


def parse_match(payload: dict, tier: str):
    """매치 JSON에서 필요한 것만 뽑는다. 쓸 수 없는 경기면 None."""
    info = payload.get("info") or {}
    if info.get("queueId") != 420:
        return None
    duration = info.get("gameDuration") or 0
    if duration < MIN_DURATION_S:
        return None

    participants = info.get("participants") or []
    if len(participants) != 10:
        return None

    rows = []
    seen = set()
    for participant in participants:
        position = participant.get("teamPosition") or ""
        if position not in POSITIONS:
            return None  # 포지션 판별 실패 — 조합 학습에 못 쓴다
        team = participant.get("teamId")
        if (team, position) in seen:
            return None  # 같은 팀에 같은 포지션 둘 — 데이터가 이상하다
        seen.add((team, position))
        keystone, perk_primary, perk_sub, perks = parse_perks(participant)
        rows.append(
            (
                payload["metadata"]["matchId"],
                team,
                position,
                participant["championId"],
                1 if participant.get("win") else 0,
                # item0~5는 인벤토리 6칸, item6은 장신구. 구매 순서가 아니라 종료 시점 상태다.
                ",".join(str(participant.get(f"item{i}", 0)) for i in range(7)),
                f"{participant.get('summoner1Id')},{participant.get('summoner2Id')}",
                keystone,
                perk_primary,
                perk_sub,
                perks,
                participant.get("goldEarned"),
                participant.get("champLevel"),
                participant.get("totalDamageDealtToChampions"),
            )
        )
    if len(seen) != 10:
        return None

    winner = 100 if next(r for r in rows if r[1] == 100)[4] == 1 else 200
    # gameVersion "15.17.712.4923" → "15.17"
    patch = ".".join((info.get("gameVersion") or "0.0").split(".")[:2])
    match_row = (
        payload["metadata"]["matchId"],
        tier,
        patch,
        info.get("gameCreation") or 0,
        duration,
        winner,
    )
    return match_row, rows


def fetch_matches(client: RiotClient, conn, limit: int, tiers: list[str] = TIERS) -> tuple[int, int]:
    """대기 중인 매치의 상세를 받는다. 요청 1건 = 매치 1건, 여기가 병목이다."""
    # 여기도 무작위. 대기열은 티어별로 뭉쳐서 들어오므로 앞에서부터 꺼내면
    # 한 티어만 계속 받게 된다. 대기열은 수천 건 규모라 정렬 비용은 무시할 만하다.
    where, args = _in(tiers)
    pending = conn.execute(
        f"SELECT match_id, tier FROM match_queue WHERE status = 0 AND {where} "
        "ORDER BY RANDOM() LIMIT ?",
        (*args, limit),
    ).fetchall()

    saved = dropped = 0
    for index, row in enumerate(pending, 1):
        payload = client.match(row["match_id"])
        parsed = parse_match(payload, row["tier"]) if payload else None
        if parsed is None:
            conn.execute(
                "UPDATE match_queue SET status = 2 WHERE match_id = ?", (row["match_id"],)
            )
            dropped += 1
        else:
            match_row, part_rows = parsed
            conn.execute(
                "INSERT OR REPLACE INTO match"
                "(match_id, tier, patch, created_ms, duration_s, winner) VALUES (?,?,?,?,?,?)",
                match_row,
            )
            conn.executemany(
                "INSERT OR REPLACE INTO participant"
                "(match_id, team, position, champion_id, win,"
                " items, spells, keystone, perk_primary, perk_sub, perks,"
                " gold, level, dmg_champ)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                part_rows,
            )
            conn.execute(
                "UPDATE match_queue SET status = 1 WHERE match_id = ?", (row["match_id"],)
            )
            saved += 1
        if index % 50 == 0:
            conn.commit()
            print(
                L(f"  {index}/{len(pending)}  저장 {saved} 버림 {dropped}", f"  {index}/{len(pending)}  saved {saved} skipped {dropped}"),
                end="\r",
                flush=True,
            )
    conn.commit()
    return saved, dropped


# ---------------------------------------------------------------- 계속 돌리기

def run_forever(client: RiotClient, conn, args) -> None:
    """대기열이 마르지 않게 채우면서 매치 상세를 계속 받는다.

    매치 ID 요청 1건이 매치 상세 수십 건 어치를 벌어 주므로,
    대기열이 넉넉하면 요청을 전부 상세 받기에 쓴다.
    """
    started = time.time()
    total_saved = 0
    while True:
        try:
            total_saved = _run_once(client, conn, args, started, total_saved)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # 하룻밤을 통째로 날리는 것보다 낫다
            print(L(f"\n[오류] {type(exc).__name__}: {exc} — 30초 후 계속합니다",
                    f"\n[error] {type(exc).__name__}: {exc} — continuing in 30s"), flush=True)
            time.sleep(30)


def _run_once(client: RiotClient, conn, args, started: float, total_saved: int) -> int:
    """수집 한 바퀴. 대기열이 마르면 채우고, 아니면 매치 상세를 받는다."""
    tiers = parse_tiers(args.tiers)
    where, targs = _in(tiers)
    pending = conn.execute(
        f"SELECT COUNT(*) FROM match_queue WHERE status = 0 AND {where}", targs
    ).fetchone()[0]

    if pending < args.low_water:
        todo = conn.execute(
            f"SELECT COUNT(*) FROM seed_player WHERE scraped_at IS NULL AND {where}", targs
        ).fetchone()[0]
        if todo == 0:
            print(L("\n씨앗이 떨어졌습니다. 더 확보합니다…", "\nOut of seed players. Fetching more…"), flush=True)
            if set(tiers) <= set(APEX_TIERS):
                collect_apex_seeds(client, conn)
            else:
                collect_seeds(client, conn, args.per_division, args.max_page)
        print(L(f"\n대기열 {pending}건 — 매치 ID를 더 모읍니다", f"\nQueue at {pending} — gathering more match IDs"), flush=True)
        found = scrape_ids(client, conn, limit=60, days=args.days, per_player=100, tiers=tiers)
        print(L(f"  새 매치 {found}건 발견", f"  Found {found} new matches"), flush=True)
        return total_saved

    saved, dropped = fetch_matches(client, conn, limit=args.batch, tiers=tiers)
    total_saved += saved
    elapsed = time.time() - started
    rate = total_saved / elapsed * 3600 if elapsed else 0
    print(
        L(f"\n누적 저장 {total_saved}건 · 시간당 {rate:.0f}건 · 요청 {client.request_count}건",
          f"\nSaved {total_saved} total · {rate:.0f}/hour · {client.request_count} requests"),
        flush=True,
    )
    if saved + dropped == 0:
        time.sleep(5)
    return total_saved


# ---------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="솔로랭크 매치 수집기")
    parser.add_argument(
        "command", choices=["seed", "seed-apex", "ids", "fetch", "run", "stats"]
    )
    parser.add_argument("--tiers", default="low",
                        help="low(아이언~플래, 기본) · apex(마스터 이상) · SILVER,GOLD 처럼 직접")
    parser.add_argument("--per-division", type=int, default=600, help="티어·디비전당 씨앗 수")
    parser.add_argument("--max-page", type=int, default=0,
                        help="리그 목록에서 고를 페이지 범위 (0이면 마지막 쪽까지)")
    parser.add_argument("--limit", type=int, default=200, help="한 번에 처리할 건수")
    parser.add_argument("--batch", type=int, default=300, help="run에서 한 번에 받을 매치 수")
    parser.add_argument("--days", type=int, default=30, help="최근 며칠치 경기를 볼지")
    parser.add_argument("--low-water", type=int, default=1500, help="대기열이 이보다 적으면 채운다")
    args = parser.parse_args()

    conn = connect()

    if args.command == "stats":
        for label, value in counts(conn).items():
            print(f"{label:24s} {value:>9,}")
        by_tier = conn.execute(
            "SELECT tier, COUNT(*) c FROM match GROUP BY tier ORDER BY c DESC"
        ).fetchall()
        if by_tier:
            print(L("\n티어별 저장 매치", "\nSaved matches by tier"))
            for row in by_tier:
                print(f"  {row['tier']:10s} {row['c']:>8,}")
        return

    # run은 며칠씩 도는 명령이라 키가 만료돼도 죽지 않고 기다리게 한다.
    client = RiotClient(wait_for_new_key=(args.command == "run"))   # 서버는 설정을 따른다
    if args.command == "seed":
        added = collect_seeds(client, conn, args.per_division, args.max_page)
        print(L(f"\n새 씨앗 {added:,}명", f"\nNew seed players: {added:,}"))
    elif args.command == "seed-apex":
        added = collect_apex_seeds(client, conn)
        print(L(f"\n새 고티어 씨앗 {added:,}명", f"\nNew high-tier seed players: {added:,}"))
    elif args.command == "ids":
        found = scrape_ids(client, conn, args.limit, args.days, per_player=100,
                           tiers=parse_tiers(args.tiers))
        print(L(f"새 매치 {found:,}건 발견", f"Found {found:,} new matches"))
    elif args.command == "fetch":
        saved, dropped = fetch_matches(client, conn, args.limit, tiers=parse_tiers(args.tiers))
        print(L(f"\n저장 {saved:,}건 · 버림 {dropped:,}건", f"\nSaved {saved:,} · skipped {dropped:,}"))
    elif args.command == "run":
        try:
            run_forever(client, conn, args)
        except KeyboardInterrupt:
            print(L("\n중단했습니다. 다시 켜면 이어서 받습니다.", "\nStopped. Start it again to resume."), file=sys.stderr)


if __name__ == "__main__":
    main()
