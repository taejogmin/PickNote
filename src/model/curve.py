"""경기 길이 구간 정의."""

from __future__ import annotations

import numpy as np

EDGES = (0, 20, 25, 30, 35, 10_000)        # 분
LABELS = ["~20분", "20~25분", "25~30분", "30~35분", "35분~"]


def buckets(data) -> np.ndarray:
    """경기마다 길이 구간 번호."""
    minutes = data.dur / 60.0
    return np.clip(np.searchsorted(EDGES, minutes, side="right") - 1, 0, len(LABELS) - 1)
