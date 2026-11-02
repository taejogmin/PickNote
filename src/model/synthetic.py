"""정답을 아는 가짜 경기를 만든다.

진짜 데이터에서 나온 숫자를 믿으려면, 먼저 **정답이 있는 문제에서 모델이 그 정답을
되찾아 오는지** 봐야 한다. 여기서 실패하면 진짜 데이터로 뽑은 승률도 못 믿는다.

챔피언 강함·궁합·상성을 직접 심어 놓고 경기를 생성한 뒤,
학습한 모델이 심어 둔 값과 얼마나 일치하는지 잰다.

    python -m src.model.synthetic
"""

from __future__ import annotations

import numpy as np

from src.model.fm import DraftModel, auc, log_loss, sigmoid


class TruePlan:
    """실제 롤과 비슷한 규모의 '진짜 세계'."""

    def __init__(self, n_champs: int = 170, k: int = 4, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.n_champs = n_champs
        self.n_slots = n_champs * 5           # (챔피언, 포지션)
        self.k = k
        # 챔피언 강함: 승률 ±3%p 정도가 되도록. logit 0.12 ≈ 3%p
        self.b = rng.normal(0, 0.12, self.n_slots)
        # 궁합·상성은 쌍의 내적이라 크기가 제곱으로 작아진다. 0.05로 두면
        # 기여가 사실상 0이 되어 "모델이 상호작용을 배우는가"를 검증할 수 없다.
        # 강함 : 상호작용이 대략 6:4가 되도록 잡았다.
        self.v = rng.normal(0, 0.15, (self.n_slots, k))
        self.a = rng.normal(0, 0.14, (self.n_slots, k))
        self.d = rng.normal(0, 0.14, (self.n_slots, k))
        # 포지션 쌍마다 궁합·상성의 무게가 다르다. 봇 듀오는 15분을 붙어 있고
        # 탑과 원딜은 거의 안 만난다. 순서는 TOP JUNGLE MIDDLE BOTTOM UTILITY.
        self.cw = np.array([
            [0.0, 1.5, 0.8, 0.2, 0.3],   # 탑
            [1.5, 0.0, 1.5, 0.9, 0.9],   # 정글 - 갱킹 관계라 위아래로 넓게
            [0.8, 1.5, 0.0, 0.6, 0.5],   # 미드
            [0.2, 0.9, 0.6, 0.0, 2.5],   # 원딜 - 서폿과 압도적
            [0.3, 0.9, 0.5, 2.5, 0.0],   # 서폿
        ])
        # 다른 팀: 같은 라인(대각선)이 제일 크고, 봇 듀오끼리 교차도 있다
        self.cx = np.array([
            [2.0, 0.6, 0.3, 0.2, 0.2],
            [0.8, 2.0, 0.8, 0.6, 0.6],
            [0.3, 0.8, 2.0, 0.3, 0.3],
            [0.2, 0.6, 0.3, 2.0, 1.2],
            [0.2, 0.6, 0.3, 1.2, 2.0],
        ])
        self.side = 0.06                      # 블루 사이드 이점 ≈ 1.5%p
        # 픽률: 실제처럼 한쪽으로 쏠리게 (일부 챔피언이 훨씬 자주 나온다)
        pick = rng.lognormal(0, 1.0, (5, n_champs))
        self.pick = pick / pick.sum(1, keepdims=True)

    def slot(self, champ: np.ndarray, position: int) -> np.ndarray:
        return champ * 5 + position

    def true_logit(self, X: np.ndarray) -> np.ndarray:
        blue, red = X[:, :5], X[:, 5:]
        strength = self.b[blue].sum(1) - self.b[red].sum(1)
        vb, vr = self.v[blue], self.v[red]
        Sb, Sr = vb.sum(1), vr.sum(1)
        syn = 0.5 * ((Sb * Sb).sum(1) - np.einsum("nik,nik->n", vb, vb)) \
            - 0.5 * ((Sr * Sr).sum(1) - np.einsum("nik,nik->n", vr, vr))
        counter = (self.a[blue].sum(1) * self.d[red].sum(1)).sum(1) \
                - (self.a[red].sum(1) * self.d[blue].sum(1)).sum(1)
        return self.side + strength + syn + counter

    def generate(self, n: int, seed: int = 0):
        """경기 n건. 각 포지션마다 픽률대로 뽑고, 한 경기에 같은 챔피언은 안 나온다."""
        rng = np.random.default_rng(seed)
        X = np.empty((n, 10), dtype=np.int64)
        for team in range(2):
            for position in range(5):
                col = team * 5 + position
                X[:, col] = self.slot(
                    rng.choice(self.n_champs, size=n, p=self.pick[position]), position
                )
        # 같은 챔피언이 두 번 나온 경기는 버린다 (실제 밴픽에서는 불가능).
        # 행마다 set()을 만들면 100만 건에서 너무 느려서 정렬로 판정한다.
        champs = np.sort(X // 5, axis=1)
        ok = ~np.any(np.diff(champs, axis=1) == 0, axis=1)
        X = X[ok]
        logit = self.true_logit(X)
        y = (rng.random(len(X)) < sigmoid(logit)).astype(np.float64)
        return X, y, logit


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / denom) if denom else 0.0


def main() -> None:
    for n_matches in (50_000, 200_000):
        print(f"\n{'=' * 62}\n가짜 경기 {n_matches:,}건으로 검증\n{'=' * 62}")
        truth = TruePlan()
        X, y, true_logit = truth.generate(int(n_matches * 1.05), seed=1)
        X, y, true_logit = X[:n_matches], y[:n_matches], true_logit[:n_matches]

        cut = int(len(X) * 0.85)
        Xtr, ytr = X[:cut], y[:cut]
        Xte, yte, lte = X[cut:], y[cut:], true_logit[cut:]

        tier_tr = np.zeros(len(Xtr), dtype=np.int64)
        tier_te = np.zeros(len(Xte), dtype=np.int64)
        val = (Xte, yte, tier_te, np.ones(len(Xte)))

        # 이 문제에서 이론상 가능한 최고 성능 — 정답 logit을 그대로 썼을 때
        best_auc = auc(sigmoid(lte), yte)
        best_ll = log_loss(sigmoid(lte), yte)
        # 정답 중 '챔피언 강함'만 썼을 때. 이 값이 위와 비슷하면 상호작용이
        # 애초에 중요하지 않은 세계이고, 그러면 시험지 자체가 잘못된 것이다.
        b_only_logit = (
            truth.side
            + truth.b[Xte[:, :5]].sum(1)
            - truth.b[Xte[:, 5:]].sum(1)
        )
        strength_auc = auc(sigmoid(b_only_logit), yte)

        results = {}
        for label, use_pairs in (("기준선 (강함만)", False), ("본 모델 (FM)", True)):
            print(f"\n  ── {label} ──")
            m = DraftModel(n_slots=truth.n_slots, n_tiers=1, k=6, seed=0,
                           use_pairs=use_pairs)
            m.fit(Xtr, ytr, tier=tier_tr, val=val,
                  epochs=80, lr=0.003, batch=8192, patience=6, verbose=True)
            results[label] = m

        print(f"\n  {'':24s} {'AUC':>8s} {'log loss':>10s}")
        for label, m in results.items():
            p = m.predict(Xte, tier_te, side=True)
            print(f"  {label:24s} {auc(p, yte):>8.4f} {log_loss(p, yte):>10.5f}")
        print(f"  {'정답 중 강함만':24s} {strength_auc:>8.4f}")
        print(f"  {'정답 전부 (천장)':24s} {best_auc:>8.4f} {best_ll:>10.5f}")

        model = results["본 모델 (FM)"]
        # 실제로 자주 나오는 챔피언만 비교한다. 표본이 없는 슬롯은 못 배우는 게 정상이다.
        seen = np.bincount(X.ravel(), minlength=truth.n_slots)
        common = seen >= 200
        print(f"\n  심어 둔 값을 되찾았나 (표본 200경기 이상인 슬롯 {common.sum()}개)")
        print(f"  {'챔피언 강함 b 상관':24s} {correlation(model.b[common], truth.b[common]):>8.3f}")
        print(f"  {'사이드 이점':24s} {model.side[0]:>8.3f}  (정답 {truth.side})")

        # 추천이 실제로 맞나 — 미드 자리에 후보를 넣어 보고 진짜 최선과 비교
        print()
        for label, m in results.items():
            rank_hit(truth, m, seed=99, label=label)


def rank_hit(truth: TruePlan, model: DraftModel, trials: int = 400, seed: int = 0,
             label: str = "") -> None:
    """내 자리에 후보를 넣어 고른 픽이, 정답 기준으로 상위 몇 %인지."""
    rng = np.random.default_rng(seed)
    n_champs = truth.n_champs
    position = 2                                   # 미드
    common = np.argsort(-truth.pick[position])[:60]  # 자주 쓰이는 60종만 후보

    ranks, gains = [], []
    for _ in range(trials):
        base, _, _ = truth.generate(1, seed=int(rng.integers(1 << 30)))
        if not len(base):
            continue
        cand = base.repeat(len(common), axis=0)
        cand[:, position] = truth.slot(common, position)
        keep = np.array([len(set(r // 5)) == 10 for r in cand])
        cand = cand[keep]
        if len(cand) < 10:
            continue
        true_p = sigmoid(truth.true_logit(cand))
        pred_p = model.predict(cand, np.zeros(len(cand), dtype=np.int64), side=True)
        chosen = int(np.argmax(pred_p))
        # 모델이 고른 픽이 정답 순위로 몇 등인가 (0 = 최선)
        ranks.append(int((true_p > true_p[chosen]).sum()) / len(cand))
        gains.append(float(true_p[chosen] - true_p.mean()))

    print(
        f"  {label:24s} 추천 1순위가 실제 상위 {np.mean(ranks) * 100:>4.1f}%  ·  "
        f"무작위 픽 대비 {np.mean(gains) * 100:>+5.2f}%p"
    )


if __name__ == "__main__":
    main()
