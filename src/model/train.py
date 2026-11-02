"""진짜 데이터로 학습하고, 티어표를 이기는지 정직하게 잰다.

    python -m src.model.train

기간을 셋으로 나눈다.

    학습        마지막 8일 이전
    튜닝        마지막 8일의 앞쪽 절반 — 설정 고르기·조기 종료·과신 보정에만 쓴다
    최종 평가   뒤쪽 절반 — 무엇을 고를 때도 보지 않는다

단순한 것부터 세운다.

    티어표   슬롯별 단순 승률을 더한 점수
    기준선   챔피언 강함을 뉴턴법으로 정확히 푼 로지스틱 회귀 (exact.py)
    FM       기준선의 강함을 고정하고 궁합·상성만 학습, 시드 5개

Adam으로 강함까지 같이 배우면 드문 슬롯을 2.5배 부풀려 티어표보다 낮아졌다.
기준선은 티어표를 부트스트랩 구간 하단 > 0으로 이겨야 채택하고, FM은 다음 규칙
(5시드 중 4 이상 구간이 0 위, 평균 > 2 x 시드 표준편차)으로 채택한다.
시간 가중치는 쓰지 않는다.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.data.champions import load as load_champions
from src.i18n import L
from src.region import model_dir
from src.model.dataset import POSITIONS, TIERS, describe, load
from src.model.curve import EDGES, LABELS, buckets
from src.model.deltas import GROUP_LAMBDAS, apply_deltas, group_deltas
from src.model.exact import fit_strength
from src.model.fm import DraftModel, log_loss, sigmoid

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT = model_dir()      # 서버마다 따로 (src/region.py)
POS_KO = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "BOTTOM": "원딜", "UTILITY": "서폿"}

HOLDOUT_DAYS = 8
TIERLIST_PRIORS = (10, 25, 50, 100, 200)
BASELINE_LAMBDAS = (300, 100, 30, 10, 3, 1)   # 센 것부터 — 앞의 답을 다음 출발점으로
DECAY_PAIR = 8.0        # 궁합·상성 감쇠. 2 / 8 / 32 중 튜닝 loss 최적
FM_SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP = 1000

# 티어 보정 — **아이언만 켠다.**
# False면 완전히 끄고, "IRON"이면 아이언 경기에만 차이값을 둔다.
#
# 아래는 10-05에 다섯 티어 전부를 껐을 때의 기록이다. 그때 열어 둔 질문
# ("아이언 안에서만은 +0.011인데 사후 선택이라 믿을 수 없다")을 사전 등록해 두고
# 평가 구간이 새 패치로 채워진 뒤 쟀더니 FM 시드 다섯 전부 하한 > 0으로 넘었다.
# 다섯 티어 전부에 두면 λ가 가장 약한 쪽에 맞춰져 아이언의 이득까지 눌린다.
#
# (10-05 기록)
# 09-16에 켤 때 통과시킨 이득이 전부 **티어별 사이드 상수**에서 온 것이었다
# ("아이언에서는 블루 사이드 이점이 2%p 작다"). 화면은 사이드 이득을 빼고 보여주므로
# 그 이득은 사용자에게 가지 않는다. 사이드를 뺀 값으로 다시 재니 FM 시드 다섯 전부,
# 기준선까지 하한이 0 아래였다 (0/5).
# 아이언만 켜는 안은 사전 등록해 두고, 다음 재학습의 평가 구간에서 잰다.
TIER_CORRECTION = "IRON"

# 궁합·상성 전체를 튜닝 구간에서 다시 키운다.
# DECAY_PAIR가 짝 항을 3배쯤 누르고 있었다 — 172만 경기 · 시드 다섯 모두 배율 3.0~3.3, 5/5 넘음.
PAIR_SCALE = True


# ---------------------------------------------------------------- 지표

def fast_auc(score: np.ndarray, y: np.ndarray) -> float:
    """순위 기반 AUC. 연속 점수라 동점이 거의 없어서 동점 처리를 생략했다 (부트스트랩용으로 빠르게)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int = BOOTSTRAP, seed: int = 0):
    """같은 경기를 다시 뽑아 두 점수의 AUC 차이를 잰다. (평균, 2.5%, 97.5%)"""
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        ii = rng.integers(0, len(y), len(y))
        d[i] = fast_auc(a[ii], y[ii]) - fast_auc(b[ii], y[ii])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def platt(z: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """실제 승률 = sigmoid(a x 모델로짓 + c) 를 맞춘다. a < 1이면 모델이 과신하고 있다는 뜻."""
    a, c = 1.0, 0.0
    for _ in range(iters):
        q = sigmoid(a * z + c)
        r = q - y
        wq = q * (1 - q)
        hess = np.array([[np.sum(wq * z * z), np.sum(wq * z)], [np.sum(wq * z), np.sum(wq)]])
        step = np.linalg.solve(hess, [np.sum(r * z), np.sum(r)])
        a, c = a - step[0], c - step[1]
    return float(a), float(c)


# ---------------------------------------------------------------- 분할과 학습

def split(data):
    cm = data.created_ms
    train = cm < cm.max() - HOLDOUT_DAYS * 86400_000
    rest = ~train
    mid = np.median(cm[rest])
    return train, rest & (cm < mid), rest & (cm >= mid)


def tierlist(data, tr, prior: float):
    """슬롯별 단순 승률을 더한 점수 — OP.GG 같은 사이트가 주는 정보.

    표본이 적은 슬롯이 승률 100%로 튀지 않게 prior 경기만큼 50%로 당긴다.
    """
    X, y = data.X[tr], data.y[tr]
    n = data.n_slots
    wins = (np.bincount(X[:, :5].ravel(), weights=np.repeat(y, 5), minlength=n)
            + np.bincount(X[:, 5:].ravel(), weights=np.repeat(1 - y, 5), minlength=n))
    games = np.bincount(X.ravel(), minlength=n)
    p = (wins + prior * 0.5) / (games + prior)
    score = np.log(p / (1 - p))
    return lambda mask: score[data.X[mask][:, :5]].sum(1) - score[data.X[mask][:, 5:]].sum(1)


def fit_baseline(data, tr, va):
    """λ를 여러 개 정확히 풀고 튜닝 loss가 가장 낮은 것을 고른다."""
    X, y, w = data.X[tr], data.y[tr], np.ones(int(tr.sum()))
    zeros_va = np.zeros(int(va.sum()), dtype=np.int64)
    beta, best, grid = None, None, []
    for lam in BASELINE_LAMBDAS:
        b, side, beta, iters = fit_strength(X, y, w, lam, data.n_slots, beta0=beta)
        model = DraftModel(n_slots=data.n_slots, n_tiers=1, use_pairs=False)
        model.b, model.side = b, np.array([side])
        loss = log_loss(sigmoid(model.logits(data.X[va], zeros_va)), data.y[va])
        grid.append((lam, loss, iters))
        if best is None or loss < best[1]:
            best = (lam, loss, model)
    return best[2], best[0], grid


def fit_fm(data, tr, va, base: DraftModel, seed: int):
    """기준선의 강함·사이드를 고정하고 궁합·상성 벡터만 학습한다."""
    zeros = np.zeros(len(data), dtype=np.int64)
    model = DraftModel(n_slots=data.n_slots, n_tiers=1, k=4, seed=seed,
                       use_pairs=True, learn_strength=False)
    model.b, model.side = base.b.copy(), base.side.copy()
    model.fit(
        data.X[tr], data.y[tr], w=np.ones(int(tr.sum())), tier=zeros[tr],
        val=(data.X[va], data.y[va], zeros[va], np.ones(int(va.sum()))),
        epochs=300, lr=0.01, decay_pair=DECAY_PAIR, patience=8, seed=seed, verbose=False,
    )
    tune_loss = log_loss(sigmoid(model.logits(data.X[va], zeros[va])), data.y[va])
    return model, tune_loss


def scale_pairs(model: DraftModel, base: DraftModel, data, va, iters: int = 50) -> float:
    """짝 항(궁합 + 상성)의 배율을 튜닝 구간에서 맞추고 모델에 넣는다.

    FM은 강함·사이드를 기준선 그대로 쓰므로 FM logit − 기준선 logit이 곧 짝 항이다.
    ⟨v,v⟩와 ⟨a,d⟩가 둘 다 이차라 v·a·d에 √배율을 곱하면 짝 항이 정확히 그 배율이 된다.
    """
    zero = np.zeros(int(va.sum()), dtype=np.int64)
    X, y = data.X[va], data.y[va]
    z = model.logits(X, zero)
    p = z - base.logits(X, zero)
    s = 0.0
    for _ in range(iters):
        q = sigmoid(z + s * p)
        step = (p @ (q - y)) / ((p * p) @ (q * (1 - q)))
        s -= step
        if abs(step) < 1e-9:
            break
    scale = 1.0 + s
    if scale <= 0:
        return 1.0      # 뒤집으라는 답은 못 쓴다 — 그대로 둔다
    root = np.sqrt(scale)
    for name in ("v", "a", "d"):
        setattr(model, name, getattr(model, name) * root)
    return float(scale)


def tier_column(model: DraftModel, data) -> np.ndarray:
    """모델이 티어를 아는 만큼만 티어를 준다 — 티어가 하나뿐이면 0을 준다.

    티어 보정을 끈 모델(`n_tiers 1`)에 `data.tier`를 그대로 주면 `tier_b` 인덱스가 넘친다.
    """
    if model.n_tiers > 1:
        return data.tier
    return np.zeros(len(data), dtype=np.int64)


def add_tier_correction(model: DraftModel, data, tr, va, te, only: str | None = None):
    """고른 모델 위에 티어별 차이값을 얹는다.

    λ는 튜닝 기간 loss로 고른다. 티어별 사이드 이점은 버린다 — 전체 사이드와 마찬가지로
    화면에서는 빼고 보여주기 때문이다 (두 팀이 같으면 50%가 나와야 한다).

    `only`에 티어 이름을 주면 **그 티어에만** 차이값을 둔다.
    나머지 티어의 차이값은 0이라, 모델이 그 티어를 받아도 전체 모델 그대로 답한다.
    """
    rows = np.flatnonzero(tr)
    logit = model.logits(data.X, np.zeros(len(data), dtype=np.int64))
    keep = None if only is None else TIERS.index(only)
    # 한 티어만 둘 때는 λ도 그 티어 차이값에만 걸어 고른다 — 다른 티어가 λ를 끌어당기면 안 된다
    trim = lambda d: d if keep is None else {t: v for t, v in d.items() if t == keep}
    best = None
    for lam in GROUP_LAMBDAS:
        d, s = group_deltas(data, logit, rows, lam, data.tier, len(TIERS))
        d, s = trim(d), {t: 0.0 for t in trim(s)}
        loss = log_loss(sigmoid(apply_deltas(data, np.flatnonzero(va), logit, d, s, data.tier)),
                        data.y[va])
        if best is None or loss < best[1]:
            best = (lam, loss, d, s)
    lam, _, deltas, _ = best

    out = DraftModel(n_slots=model.n_slots, n_tiers=len(TIERS), k=model.k,
                     use_pairs=model.use_pairs)
    for name in ("b", "v", "a", "d", "cw", "cx", "side"):
        setattr(out, name, getattr(model, name).copy())
    for t, delta in deltas.items():
        out.tier_b[t] = delta

    te_rows = np.flatnonzero(te)
    plain = logit[te_rows]
    fixed = out.logits(data.X[te_rows], data.tier[te_rows])
    mean, lo, hi = bootstrap_diff(fixed, plain, data.y[te])
    print(f"\n티어 보정 (λ {lam}) — 최종 평가 AUC {fast_auc(plain, data.y[te]):.4f}"
          f" → {fast_auc(fixed, data.y[te]):.4f}   {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for i, name in enumerate(TIERS):
        k = data.tier[te_rows] == i
        if k.sum() >= 500:
            print(f"  {name:<10}{k.sum():>7,}  {fast_auc(plain[k], data.y[te][k]):.4f}"
                  f" → {fast_auc(fixed[k], data.y[te][k]):.4f}")
    return out, {"lambda": lam, "auc_plain": fast_auc(plain, data.y[te]),
                 "auc_fixed": fast_auc(fixed, data.y[te]), "diff": [mean, lo, hi]}


def fit_curve(model: DraftModel, data, tr, va, te):
    """경기 길이 구간별 차이값.

    길이는 밴픽 때 알 수 없으므로 예측에는 쓰지 않는다 — 화면에서 "그만큼 갔다면" 곡선으로만 쓴다.
    그래서 채점도 구간 안에서만 한다.
    """
    g = buckets(data)
    rows = np.flatnonzero(tr)
    logit = model.logits(data.X, tier_column(model, data))
    best = None
    for lam in GROUP_LAMBDAS:
        d, s = group_deltas(data, logit, rows, lam, g, len(LABELS))
        loss = log_loss(sigmoid(apply_deltas(data, np.flatnonzero(va), logit, d, s, g)), data.y[va])
        if best is None or loss < best[1]:
            best = (lam, loss, d, s)
    lam, _, deltas, _ = best

    curve = np.zeros((len(LABELS), model.n_slots))
    for t_i, delta in deltas.items():
        curve[t_i] = delta

    te_rows = np.flatnonzero(te)
    gt, yt = g[te_rows], data.y[te]
    plain = logit[te_rows]
    add = curve[gt][np.arange(len(te_rows))[:, None], data.X[te_rows]]
    fixed = plain + add[:, :5].sum(1) - add[:, 5:].sum(1)
    within = lambda s: sum(fast_auc(s[gt == b], yt[gt == b]) * (gt == b).sum()
                           for b in range(len(LABELS)) if (gt == b).sum() >= 200) / len(te_rows)
    print(f"\n시간대별 곡선 (λ {lam}) — 구간 안 AUC {within(plain):.4f} → {within(fixed):.4f}")
    for b, label in enumerate(LABELS):
        k = gt == b
        if k.sum() >= 200:
            print(f"  {label:<9}{k.sum():>7,}  {fast_auc(plain[k], yt[k]):.4f} → {fast_auc(fixed[k], yt[k]):.4f}")
    return curve, {"lambda": lam, "labels": list(LABELS), "edges": list(EDGES),
                   "within_auc": [within(plain), within(fixed)]}


# ---------------------------------------------------------------- 상식 검사

def sanity(model, data, names, scale: float, min_games: int = 1500) -> None:
    """숫자가 상식과 맞는지 사람이 보는 칸. 진짜 데이터에는 정답이 없어서 이게 검증이다."""
    games = data.counts()
    pos = np.array([p for _, p in data.slots])
    champ = np.array([c for c, _ in data.slots])
    # 한 슬롯이 팀 logit에 b만큼 더하면 승률은 0.5 근처에서 대략 b/4 움직인다 (보정 기울기를 곱해 읽는다)
    # 포지션별 상수는 예측에서 상쇄되므로(한 팀에 포지션이 하나씩) 그 포지션 평균을 빼고 읽는다
    effect = model.b.copy()
    for q in range(len(POSITIONS)):
        k = pos == q
        effect[k] -= np.average(model.b[k], weights=np.maximum(games[k], 1e-9))
    effect *= scale * 25.0

    print(f"\n포지션별 챔피언 강함 (표본 {min_games:,}경기 이상, 보정 후 %p)")
    for p, position in enumerate(POSITIONS):
        mask = (pos == p) & (games >= min_games)
        idx = np.flatnonzero(mask)[np.argsort(-effect[mask])]
        top = " · ".join(f"{names.get(champ[i], champ[i])} {effect[i]:+.1f}" for i in idx[:5])
        low = " · ".join(f"{names.get(champ[i], champ[i])} {effect[i]:+.1f}" for i in idx[-3:])
        print(f"  {POS_KO[position]:<3} 강함  {top}")
        print(f"       약함  {low}   ({mask.sum()}종)")


# ---------------------------------------------------------------- 저장

# 다시 학습한 모델이 지금 쓰는 모델보다 이만큼 넘게 낮으면 바꾸지 않는다.
# 평가 기간이 달라 정확한 비교는 아니지만, 학습이 망가진 경우(데이터가 이상함 등)를 막는다.
KEEP_MARGIN = 0.005


def previous_auc() -> float | None:
    """지금 쓰는 모델의 최종 평가 AUC. 없거나 못 읽으면 None."""
    try:
        old = json.loads((OUT / "draft.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if "final_auc" in old:
        return float(old["final_auc"])
    return max(old.get("auc", {}).values(), default=None)


def backup() -> None:
    """바꾸기 전에 지금 모델을 draft.prev.*로 남긴다 — 문제가 생기면 이름만 되돌리면 된다."""
    for name in ("draft.npz", "draft.json"):
        src = OUT / name
        if src.exists():
            shutil.copyfile(src, OUT / name.replace("draft.", "draft.prev."))


def save(model, data, label: str, meta: dict, curve=None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    backup()
    path = OUT / "draft.npz"
    np.savez(
        path,
        b=model.b, tier_b=model.tier_b, v=model.v, a=model.a, d=model.d,
        cw=model.cw, cx=model.cx, side=model.side,
        slots=np.array(data.slots, dtype=np.int64),
        games=data.counts(),
        curve=np.zeros((0, model.n_slots)) if curve is None else curve,
    )
    info = {
        "label": label,
        "k": model.k,
        "n_tiers": model.n_tiers,
        "use_pairs": model.use_pairs,
        "matches": len(data),
        "patches": sorted(set(data.patch)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        **meta,
    }
    (OUT / "draft.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 실행

def verdict(lo: float, hi: float) -> str:
    # 구간이 통째로 0 아래면 "구분 안 됨"이 아니라 확실히 나쁜 것이다.
    return "채택" if lo > 0 else ("확실히 못함" if hi < 0 else "구분 안 됨")


def main() -> None:
    start = time.time()
    print(L("학습을 시작합니다. 경기 수에 따라 수십 분 걸릴 수 있습니다. 아래는 진행 기록입니다.",
            "Training started. It can take tens of minutes depending on the number of games. Progress log below (in Korean)."))
    data = load()
    print(describe(data))
    tr, va, te = split(data)
    print(f"\n학습 {tr.sum():,} / 튜닝 {va.sum():,} / 최종 평가 {te.sum():,}"
          f"  (마지막 {HOLDOUT_DAYS}일을 반으로 · 시간 가중치 없음)")
    yv, yt = data.y[va], data.y[te]
    zeros = np.zeros(len(data), dtype=np.int64)

    # --- 티어표: prior를 튜닝 AUC로 고른다 --------------------------------
    prior = max(TIERLIST_PRIORS, key=lambda p: fast_auc(tierlist(data, tr, p)(va), yv))
    s_tier = tierlist(data, tr, prior)(te)

    # --- 기준선 ---------------------------------------------------------
    t0 = time.time()
    base, lam, grid = fit_baseline(data, tr, va)
    s_base = base.logits(data.X[te], zeros[te])
    print(f"\n기준선 λ 고르기 (튜닝 loss): " + " · ".join(f"{l} {loss:.5f}" for l, loss, _ in grid)
          + f"  → λ {lam} ({time.time() - t0:.0f}초)")

    # --- FM -------------------------------------------------------------
    fms = []
    for seed in FM_SEEDS:
        t0 = time.time()
        model, tune_loss = fit_fm(data, tr, va, base, seed)
        fms.append((model, tune_loss, model.logits(data.X[te], zeros[te])))
        print(f"  FM 시드 {seed}  튜닝 loss {tune_loss:.5f}  ({len(model.history)}ep {time.time() - t0:.0f}초)",
              flush=True)

    print("\n최종 평가 AUC (고를 때 안 본 기간 · 무작위 = 0.5)")
    print(f"  티어표 (prior {prior})   {fast_auc(s_tier, yt):.4f}")
    print(f"  기준선 (λ {lam})         {fast_auc(s_base, yt):.4f}")
    for seed, (_, _, s) in zip(FM_SEEDS, fms):
        print(f"  FM 시드 {seed}              {fast_auc(s, yt):.4f}")

    # --- 판정 -----------------------------------------------------------
    print(f"\n판정 (짝지은 부트스트랩 {BOOTSTRAP}회 95% 구간)")
    intervals = {}
    mean, lo, hi = bootstrap_diff(s_base, s_tier, yt)
    intervals["기준선 - 티어표"] = [mean, lo, hi]
    chosen = "기준선" if lo > 0 else "티어표"
    print(f"  기준선 - 티어표   {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]   {verdict(lo, hi)}")

    ref = s_base if chosen == "기준선" else s_tier
    diffs = [bootstrap_diff(s, ref, yt, seed=seed) for seed, (_, _, s) in zip(FM_SEEDS, fms)]
    means = np.array([d[0] for d in diffs])
    above = sum(d[1] > 0 for d in diffs)
    sd = float(means.std(ddof=1))
    adopt_fm = above >= 4 and means.mean() > 2 * sd
    intervals[f"FM - {chosen}"] = [list(d) for d in diffs]
    for seed, d in zip(FM_SEEDS, diffs):
        print(f"  FM 시드 {seed} - {chosen}   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
    print(f"  FM 규칙: 구간 0 위 {above}/{len(FM_SEEDS)} (4 이상) · 평균 {means.mean():+.4f} > 2 x 시드SD {2 * sd:.4f}"
          f"   → {'채택' if adopt_fm else '안 켬'}")
    if adopt_fm:
        chosen = "FM"

    # 티어표는 DraftModel이 아니라 추천 화면에 못 쓴다. 그때는 통계적으로 같은 기준선을 쓴다.
    if chosen == "FM":
        seed_i = int(np.argmin([f[1] for f in fms]))
        model, label = fms[seed_i][0], f"FM (시드 {FM_SEEDS[seed_i]})"
    else:
        model, label = base, "기준선"
        if chosen == "티어표":
            print("\n  어느 모델도 티어표를 확실히 이기지 못해 통계적으로 같은 '기준선'을 씁니다.")

    # --- 짝 항 배율 -------------------------------------
    pair_scale = None
    if chosen == "FM" and PAIR_SCALE:
        pair_scale = scale_pairs(model, base, data, va)
        s_scaled = model.logits(data.X[te], zeros[te])
        d = bootstrap_diff(s_scaled, fms[seed_i][2], yt)
        print(f"\n짝 항 배율 {pair_scale:.2f} (튜닝 구간에서 맞춤) — 최종 평가 AUC "
              f"{fast_auc(fms[seed_i][2], yt):.4f} → {fast_auc(s_scaled, yt):.4f}   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")

    # --- 티어 보정 -----------------------
    if TIER_CORRECTION:
        print(f"\n티어 보정 — {TIER_CORRECTION}에만 둡니다")
        model, tier_info = add_tier_correction(model, data, tr, va, te, only=TIER_CORRECTION)
    else:
        tier_info = None
        print("\n티어 보정은 끈 채로 둡니다")
    tiers_all = tier_column(model, data)

    # --- 과신 보정 (튜닝 기간에서 맞추고, 최종 평가 기간에서 확인) --------
    a, c = platt(model.logits(data.X[va], tiers_all[va]), yv)
    z_te = model.logits(data.X[te], tiers_all[te])
    raw, cal = sigmoid(z_te), sigmoid(a * z_te + c)
    print(f"\n과신 보정: 기울기 {a:.3f} · 절편 {c:+.3f}  (최종 평가 log loss {log_loss(raw, yt):.5f} → {log_loss(cal, yt):.5f})")
    print(f"  {'보정 후 예측':<11}{'경기':>7}{'예측':>8}{'실제':>8}")
    for lo_, hi_ in ((0, .40), (.40, .45), (.45, .50), (.50, .55), (.55, .60), (.60, 1.01)):
        k = (cal >= lo_) & (cal < hi_)
        if k.sum():
            print(f"  {lo_ * 100:.0f}~{min(hi_, 1) * 100:.0f}%{'':<6}{k.sum():>7,}{cal[k].mean() * 100:>7.1f}%{yt[k].mean() * 100:>7.1f}%")

    curve, curve_info = fit_curve(model, data, tr, va, te)

    print(f"  최종 평가 AUC {fast_auc(z_te, yt):.4f}"
          + (" (티어 보정까지 넣은 값)" if TIER_CORRECTION else ""))

    names = {cid: ch["name"] for cid, ch in load_champions()["champions"].items()}
    sanity(model, data, names, scale=a)

    final_auc = fast_auc(z_te, yt)
    old_auc = previous_auc()
    if old_auc is not None and final_auc < old_auc - KEEP_MARGIN:
        print(L(f"\n새 모델의 AUC {final_auc:.4f}가 지금 모델 {old_auc:.4f}보다 낮아 바꾸지 않습니다. 지금 모델을 그대로 씁니다.",
                f"\nThe new model's AUC {final_auc:.4f} is below the current model's {old_auc:.4f}, so the current model is kept."))
        return

    path = save(model, data, label, {
        "decision": chosen,
        "holdout_days": HOLDOUT_DAYS,
        "test_matches": int(te.sum()),
        "time_weight": False,
        # 티어가 비면 recommend.py가 tier_names를 비우고 화면이 티어 고르기를 숨긴다
        "tiers": list(TIERS) if TIER_CORRECTION else [],
        # 실제로 차이값이 들어간 티어 — 화면은 이것만 묻는다
        "corrected_tiers": [TIER_CORRECTION] if TIER_CORRECTION else [],
        "tier_correction": tier_info,
        "curve": curve_info,
        "tierlist_prior": prior,
        "baseline_lambda": lam,
        "decay_pair": DECAY_PAIR,
        "pair_scale": pair_scale,
        "auc": {
            "티어표": fast_auc(s_tier, yt),
            "기준선": fast_auc(s_base, yt),
            **{f"FM 시드 {s}": fast_auc(f[2], yt) for s, f in zip(FM_SEEDS, fms)},
        },
        "intervals": intervals,
        "calibration": {"a": a, "c": c},
        "final_auc": final_auc,
    }, curve=curve)
    print(L(f"\n'{label}' 모델을 저장했습니다 → {path}  (전체 {time.time() - start:.0f}초)",
            f"\nSaved the '{label}' model → {path}  (total {time.time() - start:.0f}s)"))


if __name__ == "__main__":
    main()
