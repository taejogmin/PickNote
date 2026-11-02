"""챔피언만 알고 포지션은 모를 때 가장 그럴듯한 배치를 찾는다.

클라이언트는 상대 챔피언만 알려주고 누가 어느 라인인지는 알려주지 않는다.
챔피언마다 포지션별 경기 수로 P(포지션 | 챔피언)을 잡고, 공개된 챔피언들을 서로 다른
포지션에 놓는 모든 배치(최대 5! = 120가지)의 확률을 곱해 비교한다.
칸마다의 확신도는 그 칸이 같은 포지션인 배치들의 확률 합이다.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
UNCERTAIN = 0.7      # 이보다 낮으면 화면에 '확실치 않음'
SMOOTHING = 5.0      # 표본이 적은 챔피언은 이만큼의 경기가 고르게 있었다고 치고 당긴다


def position_probs(games_by_position: np.ndarray) -> np.ndarray:
    games = np.asarray(games_by_position, dtype=float)
    return (games + SMOOTHING / len(POSITIONS)) / (games.sum() + SMOOTHING)


def infer_positions(champions: list[int], games_of, taken: tuple[str, ...] = ()) -> list[dict]:
    """games_of(champion) -> 포지션 5개의 경기 수. taken은 이미 주인이 정해진 포지션.

    돌려주는 것: [{"championId", "position", "confidence", "uncertain"}, ...] (챔피언 순서대로)
    """
    free = [i for i, p in enumerate(POSITIONS) if p not in taken]
    champions = list(champions)[: len(free)]
    if not champions:
        return []
    probs = np.array([position_probs(games_of(c)) for c in champions])      # (k, 5)

    layouts, weights = [], []
    for layout in permutations(free, len(champions)):
        layouts.append(layout)
        weights.append(np.prod([probs[i, pos] for i, pos in enumerate(layout)]))
    weights = np.array(weights)
    total = weights.sum()
    weights = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
    best = layouts[int(np.argmax(weights))]

    out = []
    for i, champion in enumerate(champions):
        confidence = float(sum(w for layout, w in zip(layouts, weights) if layout[i] == best[i]))
        out.append({
            "championId": int(champion),
            "position": POSITIONS[best[i]],
            "confidence": round(confidence, 3),
            "uncertain": confidence < UNCERTAIN,
        })
    return out
