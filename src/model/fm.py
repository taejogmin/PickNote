"""조합 승률 모델.

    logit = 사이드 이점
          + Σ 아군 강함 − Σ 상대 강함                      (b, 티어 보정 포함)
          + Σ(아군 쌍)⟨v,v⟩ − Σ(상대 쌍)⟨v,v⟩              궁합
          + Σ⟨a_아군, d_상대⟩ − Σ⟨a_상대, d_아군⟩           상성

궁합과 상성에 서로 다른 벡터를 쓰는 이유는 docs/how-it-works.md에 적었다.
하나로 묶으면 "궁합이 좋으면 상성도 좋다"는 말이 안 되는 제약이 걸린다.

torch를 안 쓰는 이유: 파라미터가 수만 개뿐이라 numpy로 충분하고,
Python 3.14에 휠이 없을 수도 있는 큰 의존성을 피하고 싶었다.
"""

from __future__ import annotations

import sys

import numpy as np

# 윈도우 콘솔은 기본이 cp949라 일부 기호에서 UnicodeEncodeError로 죽는다.
# 학습 로그 한 줄 때문에 몇 분짜리 학습이 날아가면 안 된다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """오버플로 없는 시그모이드."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class DraftModel:
    """X는 (N, 10) 정수 배열. 0~4열이 블루(포지션 순), 5~9열이 레드."""

    def __init__(
        self,
        n_slots: int,
        n_tiers: int = 1,
        k: int = 4,
        seed: int = 0,
        use_pairs: bool = True,
        learn_pair_weights: bool = False,
        learn_strength: bool = True,
        init_scale: float = 0.03,
    ):
        rng = np.random.default_rng(seed)
        # 상호작용 항은 모든 벡터가 0 근처면 그래디언트도 0 근처가 되는 안장점이 있다.
        # (궁합 그래디언트가 g·(Σv − v_i)라, 남들이 작으면 나도 못 큰다)
        # 그래서 0에 너무 가깝게 초기화하면 영영 못 배운다.
        scale = init_scale
        self.k = k
        self.n_slots = n_slots
        self.n_tiers = n_tiers
        # False면 궁합·상성을 끄고 "챔피언 강함"만 쓴다.
        # 이게 기준선(로지스틱 회귀)이다.
        # 본 모델이 이걸 못 이기면 복잡한 쪽을 쓸 이유가 없다.
        self.use_pairs = use_pairs
        # 기본은 1로 고정. 봇 듀오 궁합을 더 무겁게 보자는 발상은 맞지만,
        # 슬롯이 (챔피언, 포지션) 쌍이라 임베딩 크기가 이미 그 역할을 한다.
        # c[봇,서폿]을 올리는 것과 |v_서폿|을 키우는 게 같은 효과다.
        # 가짜 데이터에서 12배 차이를 심어도 못 배웠다.
        self.learn_pair_weights = learn_pair_weights
        # False면 챔피언 강함·티어 보정·사이드를 건드리지 않고 궁합·상성만 배운다.
        # 강함은 exact.py로 정확히 푼 값을 넣어 두고 쓴다. Adam이 강함까지 배우면
        # 드문 슬롯을 크게 부풀린다.
        self.learn_strength = learn_strength
        self.b = np.zeros(n_slots)                          # 챔피언 자체 강함
        self.tier_b = np.zeros((n_tiers, n_slots))          # 티어별 보정
        self.v = rng.normal(0, scale, (n_slots, k))         # 궁합
        self.a = rng.normal(0, scale, (n_slots, k))         # 공격 (상성)
        self.d = rng.normal(0, scale, (n_slots, k))         # 피격 (상성)
        # 포지션 쌍마다 궁합·상성이 얼마나 중요한지. 봇 듀오는 15분을 붙어 있고
        # 탑과 원딜은 거의 안 만나는데 지금까지는 10쌍을 전부 똑같이 봤다.
        # 파라미터가 10개(같은 팀) + 25개(다른 팀)뿐이라 과적합 위험이 거의 없다.
        # cross의 대각선 cx[p,p]가 곧 '같은 라인끼리의 상성'이다.
        self.cw = np.ones((5, 5))                           # 같은 팀 (대칭, 대각 무시)
        self.cx = np.ones((5, 5))                           # 아군 p vs 상대 q
        self.side = np.zeros(1)                             # 블루 사이드 이점
        self._adam = {}

    # ------------------------------------------------------------------ 순전파

    def _parts(self, X: np.ndarray, tier: np.ndarray):
        """logit과 역전파에 필요한 중간값을 함께 돌려준다."""
        blue, red = X[:, :5], X[:, 5:]

        # 필요한 10개 슬롯만 뽑는다. (N, 전체슬롯) 배열을 만들면 메모리를 크게 먹는다.
        b_all = self.b[X] + self.tier_b[tier[:, None], X]   # (N, 10)
        strength = b_all[:, :5].sum(1) - b_all[:, 5:].sum(1)

        if not self.use_pairs:
            return self.side[0] + strength, (blue, red, None, None, None, None)

        vb, vr = self.v[blue], self.v[red]                  # (N, 5, k)
        ab, ar = self.a[blue], self.a[red]
        db, dr = self.d[blue], self.d[red]

        # 포지션 쌍별로 가중치를 다르게 주려면 |Σv|² 요령을 못 쓴다.
        # 쌍마다 내적을 따로 구한다. (N,5,5)라 배치 8192에서도 20만 개 남짓이다.
        Pb = np.einsum("npk,nqk->npq", vb, vb)
        Pr = np.einsum("npk,nqk->npq", vr, vr)
        W = self.pair_weight()                              # 대칭, 대각 0
        synergy = 0.5 * (np.einsum("npq,pq->n", Pb, W) - np.einsum("npq,pq->n", Pr, W))

        Xbr = np.einsum("npk,nqk->npq", ab, dr)             # 아군이 상대에게
        Xrb = np.einsum("npk,nqk->npq", ar, db)
        counter = np.einsum("npq,pq->n", Xbr - Xrb, self.cx)

        logit = self.side[0] + strength + synergy + counter
        return logit, (blue, red, vb, vr, ab, ar, db, dr, Pb, Pr, Xbr, Xrb)

    def pair_weight(self) -> np.ndarray:
        """같은 팀 포지션 쌍 가중치. 대칭이고 자기 자신과의 쌍은 없다."""
        W = self.cw + self.cw.T
        np.fill_diagonal(W, 0.0)
        return W * 0.5

    def logits(self, X: np.ndarray, tier: np.ndarray) -> np.ndarray:
        return self._parts(X, tier)[0]

    def predict(self, X: np.ndarray, tier: np.ndarray, *, side: bool = False) -> np.ndarray:
        """블루(아군) 승률. side=False면 사이드 이점을 뺀 '순수 조합' 값."""
        logit = self.logits(X, tier)
        if not side:
            logit = logit - self.side[0]
        return sigmoid(logit)

    # ------------------------------------------------------------------ 학습

    def _step(self, name, param, grad, lr, decay=0.0, center=0.0,
              beta1=0.9, beta2=0.999, eps=1e-8):
        """AdamW. 감쇠를 그래디언트에 더하지 않고 따로 뺀다.

        L2를 그래디언트에 더하면 Adam이 크기를 정규화하면서 감쇠까지 같이 눌러 버려
        사실상 정규화가 안 걸린다. 5만 경기에서 과적합한 원인이 이것이었다.
        """
        state = self._adam.setdefault(
            name, {"m": np.zeros_like(param), "v": np.zeros_like(param), "t": 0}
        )
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * grad * grad
        mhat = state["m"] / (1 - beta1 ** state["t"])
        vhat = state["v"] / (1 - beta2 ** state["t"])
        # center는 감쇠가 끌고 갈 지점. 보통 0이지만 포지션 쌍 가중치는
        # 1(모든 쌍이 똑같이 중요)로 끌어야 한다. 0으로 끌면 상호작용이 통째로 죽는다.
        param -= lr * (mhat / (np.sqrt(vhat) + eps) + decay * (param - center))

    # ------------------------------------------------------------------ 되돌리기

    def snapshot(self) -> dict:
        return {n: getattr(self, n).copy() for n in ("b", "tier_b", "v", "a", "d", "cw", "cx", "side")}

    def restore(self, snap: dict) -> None:
        for name, value in snap.items():
            setattr(self, name, value.copy())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray | None = None,
        tier: np.ndarray | None = None,
        *,
        epochs: int = 80,
        batch: int = 8192,
        lr: float = 0.01,
        # 감쇠는 스텝마다 lr*decay 만큼 줄이므로, 0.02 같은 값은 수백 스텝을 돌려도
        # 1~2%밖에 안 줄어 사실상 정규화가 안 걸린다. 가짜 데이터 실험에서
        # 0.5가 4.0보다 뚜렷하게 좋았다 (4.0은 벡터를 눌러 버려 상호작용을 못 배운다).
        decay_b: float = 0.5,
        decay_pair: float = 0.5,
        decay_tier: float = 0.5,
        seed: int = 0,
        verbose: bool = True,
        val: tuple | None = None,    # (X, y, tier, w) 조기 종료용
        patience: int = 5,
    ):
        n = len(X)
        w = np.ones(n) if w is None else w.astype(np.float64)
        tier = np.zeros(n, dtype=np.int64) if tier is None else tier
        y = y.astype(np.float64)
        rng = np.random.default_rng(seed)

        best = (np.inf, -1, None)                 # (검증 손실, epoch, 파라미터)
        self.history = []

        for epoch in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                Xb, yb, wb, tb = X[idx], y[idx], w[idx], tier[idx]
                logit, cache = self._parts(Xb, tb)
                p = sigmoid(logit)
                total += float(
                    np.sum(wb * -(yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12)))
                )
                g = (p - yb) * wb / len(idx)
                self._backward(g, cache, Xb, tb, lr, decay_b, decay_pair, decay_tier)

            train_loss = total / w.sum()
            line = f"  epoch {epoch + 1:2d}/{epochs}  train {train_loss:.5f}"

            if val is not None:
                Xv, yv, tv, wv = val
                pv = sigmoid(self.logits(Xv, tv))
                vl = log_loss(pv, yv, wv)
                va = auc(pv, yv)
                self.history.append((train_loss, vl, va))
                line += f"   검증 {vl:.5f}  AUC {va:.4f}"
                if vl < best[0] - 1e-6:
                    best = (vl, epoch, self.snapshot())
                    line += "  *"
                elif best[2] is None:
                    raise ValueError(
                        "검증 손실이 NaN입니다. 검증셋이 비었거나 라벨이 잘못됐는지 보세요."
                    )
                elif epoch - best[1] >= patience:
                    if verbose:
                        print(line, flush=True)
                        print(
                            f"  조기 종료 - {patience}번째 개선 없음. "
                            f"epoch {best[1] + 1}로 되돌립니다 (검증 {best[0]:.5f})",
                            flush=True,
                        )
                    self.restore(best[2])
                    return self
            if verbose:
                print(line, flush=True)

        if val is not None and best[2] is not None:
            self.restore(best[2])
            if verbose:
                print(f"  최적 epoch {best[1] + 1} (검증 {best[0]:.5f})로 되돌립니다", flush=True)
        return self

    def _backward(self, g, cache, X, tier, lr, decay_b, decay_pair, decay_tier,
                  decay_cw=0.05):
        # 기준선(use_pairs=False)은 캐시가 짧다. 여기서 12칸을 풀면 죽는다.
        blue, red = cache[0], cache[1]
        g1 = g[:, None]
        g2 = g[:, None, None]

        # --- 챔피언 강함 -------------------------------------------------
        gb = np.zeros_like(self.b)
        np.add.at(gb, blue, g1)
        np.add.at(gb, red, -g1)

        gt = np.zeros_like(self.tier_b)
        np.add.at(gt, (tier[:, None], blue), g1)
        np.add.at(gt, (tier[:, None], red), -g1)

        if self.learn_strength:
            self._step("b", self.b, gb, lr, decay_b)
            self._step("tier_b", self.tier_b, gt, lr, decay_tier)
            self._step("side", self.side, np.array([g.sum()]), lr)
        if not self.use_pairs:
            return

        _, _, vb, vr, ab, ar, db, dr, Pb, Pr, Xbr, Xrb = cache
        W = self.pair_weight()

        # --- 궁합: d/dv_p = Σ_q W[p,q] v_q -------------------------------
        gv = np.zeros_like(self.v)
        np.add.at(gv, blue, g2 * np.einsum("pq,nqk->npk", W, vb))
        np.add.at(gv, red, -g2 * np.einsum("pq,nqk->npk", W, vr))

        # --- 상성 --------------------------------------------------------
        ga = np.zeros_like(self.a)
        np.add.at(ga, blue, g2 * np.einsum("pq,nqk->npk", self.cx, dr))
        np.add.at(ga, red, -g2 * np.einsum("pq,nqk->npk", self.cx, db))

        gd = np.zeros_like(self.d)
        np.add.at(gd, red, g2 * np.einsum("pq,npk->nqk", self.cx, ab))
        np.add.at(gd, blue, -g2 * np.einsum("pq,npk->nqk", self.cx, ar))

        # --- 포지션 쌍 가중치 (10개 + 25개) -------------------------------
        gcw = 0.5 * np.einsum("n,npq->pq", g, Pb - Pr)
        np.fill_diagonal(gcw, 0.0)
        gcx = np.einsum("n,npq->pq", g, Xbr - Xrb)

        self._step("v", self.v, gv, lr, decay_pair)
        self._step("a", self.a, ga, lr, decay_pair)
        self._step("d", self.d, gd, lr, decay_pair)
        if self.learn_pair_weights:
            self._step("cw", self.cw, gcw, lr, decay_cw, center=1.0)
            self._step("cx", self.cx, gcx, lr, decay_cw, center=1.0)


# ---------------------------------------------------------------- 지표

def log_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones_like(y, dtype=float) if w is None else w
    ll = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return float((ll * w).sum() / w.sum())


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """순위 기반 AUC. scipy 없이 계산한다."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # 동점 처리
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
