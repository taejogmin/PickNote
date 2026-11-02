"""DB → 학습용 배열.

    X      (N, 10)  0~4열 블루(포지션 순), 5~9열 레드. 값은 (챔피언,포지션) 슬롯 번호
    y      (N,)     블루가 이겼으면 1
    w      (N,)     시간 가중치. 오래된 경기일수록 작다
    tier   (N,)     티어 번호
    dur    (N,)     경기 길이(초) — 시간대별 승률에 쓴다
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.data.store import connect

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM"]
# 고티어는 실험용으로 따로 모은다. 기본 학습에는 넣지 않는다.
APEX_TIERS = ["MASTER", "GRANDMASTER", "CHALLENGER"]
ALL_TIERS = TIERS + APEX_TIERS
TIER_INDEX = {t: i for i, t in enumerate(ALL_TIERS)}

HALF_LIFE_DAYS = 14.0


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    tier: np.ndarray
    dur: np.ndarray
    created_ms: np.ndarray
    patch: list[str]
    slots: list[tuple[int, int]]          # 슬롯 번호 → (챔피언 id, 포지션 번호)
    slot_of: dict[tuple[int, int], int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    def split_by_time(self, val_days: float = 5.0):
        """마지막 val_days를 검증셋으로 뗀다.

        무작위로 나누면 안 된다. 같은 패치·같은 유저가 양쪽에 섞여
        성능이 실제보다 좋아 보인다. 우리가 쓰는 방식(과거로 배워 미래를 맞힘)과도
        어긋난다.
        """
        cutoff = self.created_ms.max() - val_days * 86400_000
        is_val = self.created_ms >= cutoff
        return ~is_val, is_val

    def counts(self) -> np.ndarray:
        """슬롯별 등장 경기 수."""
        return np.bincount(self.X.ravel(), minlength=self.n_slots)


def load(conn=None, *, min_matches_per_slot: int = 0, tiers=tuple(TIERS)) -> Dataset:
    """tiers를 주지 않으면 아이언~플래티넘만 읽는다.

    고티어 경기가 DB에 들어와도 기본 학습에 저절로 섞이지 않게 하려는 것이다.
    그 전에는 모르는 티어를 아이언(0)으로 처리해서, 고티어를 모으는 순간 섞일 뻔했다.
    전부 읽으려면 tiers=None.
    """
    conn = conn or connect()

    where, args = "", []
    if tiers:
        where = f"WHERE tier IN ({','.join('?' * len(tiers))})"
        args = list(tiers)
    rows = conn.execute(
        f"SELECT match_id, tier, patch, created_ms, duration_s, winner "
        f"FROM match {where} ORDER BY created_ms",
        args,
    ).fetchall()
    if not rows:
        raise SystemExit("학습할 매치가 없습니다. 먼저 수집기를 돌려 주세요.")

    order = {row["match_id"]: i for i, row in enumerate(rows)}
    n = len(rows)
    X = np.full((n, 10), -1, dtype=np.int64)

    slot_of: dict[tuple[int, int], int] = {}
    slots: list[tuple[int, int]] = []
    for match_id, team, position, champion in conn.execute(
        "SELECT match_id, team, position, champion_id FROM participant"
    ):
        i = order.get(match_id)
        if i is None:
            continue
        pos = POS_INDEX.get(position)
        if pos is None:
            continue
        key = (champion, pos)
        slot = slot_of.get(key)
        if slot is None:
            slot = len(slots)
            slot_of[key] = slot
            slots.append(key)
        X[i, (0 if team == 100 else 5) + pos] = slot

    complete = (X >= 0).all(axis=1)
    X = X[complete]
    kept = [row for row, ok in zip(rows, complete) if ok]

    y = np.array([1.0 if r["winner"] == 100 else 0.0 for r in kept])
    tier = np.array([TIER_INDEX[r["tier"]] for r in kept], dtype=np.int64)
    dur = np.array([r["duration_s"] for r in kept], dtype=np.int64)
    created = np.array([r["created_ms"] for r in kept], dtype=np.int64)
    patch = [r["patch"] for r in kept]

    # 시간 가중치. 기준은 "가장 최근 경기"이지 오늘이 아니다 —
    # 며칠 뒤에 다시 학습해도 같은 데이터에서 같은 가중치가 나와야 한다.
    age_days = (created.max() - created) / 86400_000
    w = 0.5 ** (age_days / HALF_LIFE_DAYS)

    data = Dataset(X, y, w, tier, dur, created, patch, slots, slot_of)

    if min_matches_per_slot:
        data = drop_rare(data, min_matches_per_slot)
    return data


def drop_rare(data: Dataset, minimum: int) -> Dataset:
    """표본이 너무 적은 (챔피언,포지션)이 낀 경기를 통째로 뺀다.

    예: 서폿 자리에 나온 럭스가 30경기뿐이라면 그 슬롯은 추정이 요동친다.
    슬롯만 빼면 그 경기의 10칸 중 하나가 비어 버리므로 경기째로 뺀다.
    """
    counts = data.counts()
    rare = counts < minimum
    if not rare.any():
        return data
    keep = ~rare[data.X].any(axis=1)
    X = data.X[keep]

    # 남은 슬롯만으로 번호를 다시 매긴다
    used = np.unique(X)
    remap = np.full(data.n_slots, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Dataset(
        X=remap[X],
        y=data.y[keep], w=data.w[keep], tier=data.tier[keep],
        dur=data.dur[keep], created_ms=data.created_ms[keep],
        patch=[p for p, k in zip(data.patch, keep) if k],
        slots=[data.slots[i] for i in used],
        slot_of={data.slots[i]: j for j, i in enumerate(used)},
    )


def describe(data: Dataset) -> str:
    from collections import Counter

    patches = Counter(data.patch).most_common(4)
    tiers = Counter(ALL_TIERS[t] for t in data.tier).most_common()
    span = (data.created_ms.max() - data.created_ms.min()) / 86400_000
    lines = [
        f"매치 {len(data):,}건 · 슬롯 {data.n_slots:,}개 · {span:.1f}일치",
        f"  블루 승률 {data.y.mean() * 100:.2f}%  (사이드 이점이 여기 보인다)",
        f"  티어  {' · '.join(f'{t} {c:,}' for t, c in tiers)}",
        f"  패치  {' · '.join(f'{p} {c:,}' for p, c in patches)}",
        f"  시간 가중치 합 {data.w.sum():,.0f} (실제 경기 수 대비 "
        f"{data.w.sum() / len(data) * 100:.0f}%)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    start = time.time()
    data = load()
    print(describe(data))
    print(f"\n({time.time() - start:.1f}초)")
    train, val = data.split_by_time()
    print(f"시간 분할: 학습 {train.sum():,} / 검증 {val.sum():,}")
