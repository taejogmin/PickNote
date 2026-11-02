"""챔피언 강함을 뉴턴법으로 정확히 푼다 (L2 로지스틱 회귀).

Adam으로 강함을 배우면 파라미터마다 걸음을 정규화하는 탓에 드문 슬롯을 크게 부풀리고
흔한 슬롯은 덜 배워서 단순 승률표보다도 낮아졌다.
볼록 문제라 정확히 풀면 답이 하나고 시드에 따라 흔들리지 않는다.
파라미터가 슬롯 수 + 1(사이드)개뿐이라 헤세 행렬을 통째로 풀어도 반복당 1초 안팎이다.
"""

from __future__ import annotations

import numpy as np

from src.model.fm import sigmoid

# 한 경기 = 블루 5칸(+1) · 레드 5칸(-1) · 사이드(+1)
SIGN = np.array([1.0] * 5 + [-1.0] * 5 + [1.0])
_OUTER = np.outer(SIGN, SIGN).ravel()


def _rows(X: np.ndarray, n_slots: int) -> np.ndarray:
    return np.hstack([X, np.full((len(X), 1), n_slots, dtype=X.dtype)])


def fit_strength(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    lam: float,
    n_slots: int,
    beta0: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    tol: float = 1e-6,
    iters: int = 50,
    chunk: int = 40_000,
):
    """(슬롯별 강함, 사이드, 전체 계수, 반복 수)를 돌려준다.

    lam은 L2 세기(경기 수 단위). 사이드는 정규화하지 않는다.
    beta0를 주면 거기서 출발한다 — λ를 여러 개 풀 때 앞의 답을 넘기면 반복이 준다.
    offset을 주면 그 로짓 위에 얹어서 푼다 — 티어별 보정처럼 '차이값만' 구할 때 쓴다
   .
    """
    idx = _rows(X, n_slots)
    size = n_slots + 1
    beta = np.zeros(size) if beta0 is None else beta0.astype(float).copy()
    reg = np.full(size, float(lam))
    reg[n_slots] = 0.0

    off = 0.0 if offset is None else np.asarray(offset, dtype=float)

    def objective(b):
        z = off + (b[idx] * SIGN).sum(1)
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(reg * b * b))

    f = objective(beta)
    for it in range(1, iters + 1):
        p = sigmoid(off + (beta[idx] * SIGN).sum(1))
        grad = np.bincount(idx.ravel(), weights=(SIGN[None, :] * (w * (p - y))[:, None]).ravel(),
                           minlength=size) + reg * beta
        if np.abs(grad).max() < tol:
            break
        d = w * p * (1.0 - p)
        hess = np.zeros(size * size)
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            flat = (part[:, :, None] * size + part[:, None, :]).reshape(len(part), -1)
            hess += np.bincount(flat.ravel(), weights=(d[s:s + chunk, None] * _OUTER[None, :]).ravel(),
                                minlength=size * size)
        hess = hess.reshape(size, size) + np.diag(reg) + 1e-8 * np.eye(size)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while objective(beta - t * step) > f and t > 1e-6:
            t *= 0.5
        beta = beta - t * step
        f = objective(beta)
    return beta[:n_slots].copy(), float(beta[n_slots]), beta, it
