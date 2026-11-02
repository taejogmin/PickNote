"""SQLite 저장소.

수집은 며칠에 걸쳐 돌고 개발용 키는 24시간마다 만료되므로,
어디까지 했는지가 전부 DB에 남아 있어야 한다. 모든 단계는 중단·재개가 가능하다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.i18n import L
from src.region import db_path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = """
PRAGMA journal_mode = WAL;

-- 티어를 알고 있는 소환사. 매치 ID를 긁어 오는 출발점.
CREATE TABLE IF NOT EXISTS seed_player (
    puuid      TEXT PRIMARY KEY,
    tier       TEXT NOT NULL,
    division   TEXT NOT NULL,
    scraped_at INTEGER              -- 매치 ID를 긁어 간 시각. NULL이면 아직 안 함
);
CREATE INDEX IF NOT EXISTS ix_seed_todo ON seed_player(scraped_at) WHERE scraped_at IS NULL;

-- 발견했지만 아직 상세를 안 받은 매치. tier는 이 매치를 찾아 준 씨앗의 티어.
CREATE TABLE IF NOT EXISTS match_queue (
    match_id TEXT PRIMARY KEY,
    tier     TEXT NOT NULL,
    status   INTEGER NOT NULL DEFAULT 0   -- 0 대기 / 1 저장됨 / 2 버림
);
CREATE INDEX IF NOT EXISTS ix_queue_todo ON match_queue(status) WHERE status = 0;

CREATE TABLE IF NOT EXISTS match (
    match_id   TEXT PRIMARY KEY,
    tier       TEXT NOT NULL,
    patch      TEXT NOT NULL,          -- 예: "15.17"
    created_ms INTEGER NOT NULL,
    duration_s INTEGER NOT NULL,
    winner     INTEGER NOT NULL        -- 100 블루 / 200 레드
);
CREATE INDEX IF NOT EXISTS ix_match_created ON match(created_ms);

CREATE TABLE IF NOT EXISTS participant (
    match_id    TEXT NOT NULL,
    team        INTEGER NOT NULL,      -- 100 / 200
    position    TEXT NOT NULL,         -- TOP JUNGLE MIDDLE BOTTOM UTILITY
    champion_id INTEGER NOT NULL,
    win         INTEGER NOT NULL,
    PRIMARY KEY (match_id, team, position)   -- 팀당 포지션 중복 불가
);
CREATE INDEX IF NOT EXISTS ix_part_champ ON participant(champion_id, position);
"""

# 나중에 추가한 열. 매치 응답에 이미 들어 있어 추가 요청 없이 저장할 수 있다.
# 주의: item0~6은 "게임이 끝났을 때의 인벤토리"이지 구매 순서가 아니다.
# 순서와 타이밍은 별도의 timeline 엔드포인트가 필요하다 (요청 2배).
EXTRA_COLUMNS = [
    ("items", "TEXT"),          # "1054,3047,6631,3742,3046,0,3340" (6칸 + 장신구)
    ("spells", "TEXT"),         # "14,4"
    ("keystone", "INTEGER"),    # 핵심 룬
    ("perk_primary", "INTEGER"),
    ("perk_sub", "INTEGER"),
    ("perks", "TEXT"),          # 선택한 룬 전체
    ("gold", "INTEGER"),
    ("level", "INTEGER"),
    ("dmg_champ", "INTEGER"),   # 챔피언에게 준 피해 — 아이템 효과를 통제할 때 쓴다
]


def migrate(conn: sqlite3.Connection) -> list[str]:
    """이미 만들어진 DB에 열을 더한다. SQLite에는 ADD COLUMN IF NOT EXISTS가 없다."""
    have = {row["name"] for row in conn.execute("PRAGMA table_info(participant)")}
    added = []
    for name, kind in EXTRA_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE participant ADD COLUMN {name} {kind}")
            added.append(name)
    if added:
        conn.commit()
    return added


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()       # 서버마다 따로 (src/region.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    def one(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    return {
        L("씨앗 소환사", "Seed players"): one("SELECT COUNT(*) FROM seed_player"),
        L("  └ 아직 안 긁음", "  └ not scraped yet"): one("SELECT COUNT(*) FROM seed_player WHERE scraped_at IS NULL"),
        L("발견한 매치", "Matches found"): one("SELECT COUNT(*) FROM match_queue"),
        L("  └ 상세 대기", "  └ waiting for details"): one("SELECT COUNT(*) FROM match_queue WHERE status = 0"),
        L("저장한 매치", "Matches saved"): one("SELECT COUNT(*) FROM match"),
        L("버린 매치", "Matches skipped"): one("SELECT COUNT(*) FROM match_queue WHERE status = 2"),
    }
