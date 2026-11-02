"""무리별 '차이값'을 정확히 푸는 공용 도구.

전체로 배운 모델의 로짓을 출발점(offset)으로 두고, 무리(티어·경기 길이 구간)마다
그 무리의 경기만으로 슬롯 차이값을 L2로 정확히 푼다.
"""

from __future__ import annotations

import numpy as np

from src.model.exact import fit_strength

# 무리별 경기는 전체의 1/5쯤이라 전체(300~1)보다 센 쪽까지 본다
GROUP_LAMBDAS = (10_000, 3_000, 1_000, 300, 100, 30)


def group_deltas(data, base_logit: np.ndarray, tr, lam: float, group: np.ndarray, n_groups: int):
    """무리(티어·경기 길이 구간)마다 '전체 위에 얹는 차이값'을 푼다. (슬롯 차이값, 사이드 차이값)"""
    deltas, sides = {}, {}
    for t in range(n_groups):
        rows = tr[group[tr] == t]
        if not len(rows):
            continue
        b, side, _, _ = fit_strength(
            data.X[rows], data.y[rows], np.ones(len(rows)), lam, data.n_slots,
            offset=base_logit[rows],
        )
        deltas[t], sides[t] = b, side
    return deltas, sides


def apply_deltas(data, rows, base_logit: np.ndarray, deltas, sides, group: np.ndarray) -> np.ndarray:
    """경기마다 자기 무리의 차이값을 더한 점수."""
    out = base_logit[rows].copy()
    X = data.X[rows]
    tier = group[rows]
    for t, b in deltas.items():
        k = tier == t
        if not k.any():
            continue
        add = b[X[k][:, :5]].sum(1) - b[X[k][:, 5:]].sum(1)
        out[k] += add + sides[t]
    return out
