"""내 경기 기록 읽기.

`python -m src.riot.me`가 받아 둔 `data/me.db`를 읽기만 한다. 화면과 숙련도 실험이 같이 쓴다.
파일이 없으면(내 기록을 안 받은 PC) 모든 함수가 빈 값을 돌려준다 — 화면은 그걸 보고 내 기록 칸을 숨긴다.

**조합 기대**: 판마다 모델이 그 판의 열 챔피언만 보고 낸 내 팀 승률. "평균 실력이었다면"의 기댓값이라,
실제 승률에서 빼면 조합 탓을 덜어 낸 내 몫이 된다.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.i18n import L
from src.model.fm import sigmoid

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "me.db"
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
REMAKE_S = 300          # 5분 안에 끝난 판은 다시하기다 — 세지 않는다
POOL_MIN = 5            # 그 자리에서 이만큼 했으면 내 풀
POOL_MAX = 8
TAG_MIN = 3             # 추천 줄에 내 기록을 다는 최소 판 수
NO_CORRECTION_TIER = "SILVER"   # 내 티어는 묻지 않는다 — 보정 없는 쪽으로


@dataclass
class Game:
    match_id: str
    created_ms: int
    duration_s: int
    queue: int
    champion: int
    position: str               # 비어 있을 수 있다
    win: int
    team: int                   # 100 또는 200
    lineup: dict = field(default_factory=dict)   # 팀 → {포지션: 챔피언}
    mates: list = field(default_factory=list)    # 같은 팀 넷의 해시 (이름은 저장하지 않았다)
    account: str = ""                            # 어느 내 계정의 판인가
    season: str = ""                             # 패치 앞자리 ("16")
    duel: dict | None = None                     # 나와 맞은편의 숫자 — 솔로랭크 15분 넘은 판만

    @property
    def foe(self) -> int | None:
        """같은 포지션의 상대 챔피언 — 상대 라이너."""
        other = 200 if self.team == 100 else 100
        return self.lineup.get(other, {}).get(self.position)

    @property
    def full(self) -> bool:
        return all(len(self.lineup.get(t, {})) == 5 for t in (100, 200))


_cache: dict = {"mtime": None, "games": [], "me": {}, "accounts": []}


def _load() -> None:
    if not DB.exists():
        _cache.update(mtime=None, games=[], me={}, accounts=[])
        return
    mtime = DB.stat().st_mtime
    if mtime == _cache["mtime"]:
        return
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        me = dict(conn.execute("SELECT key, value FROM me"))
        try:            # 계정이 하나뿐이던 때 파일 — `python -m src.riot.me`를 한 번 돌리면 바뀐다
            accounts = conn.execute("SELECT puuid, riot_id FROM account ORDER BY added_ms").fetchall()
            account_col = "account"
        except sqlite3.OperationalError:
            accounts = [(me.get("puuid", ""), me.get("riot_id", ""))]
            account_col = "NULL"
        rows = conn.execute(
            "SELECT match_id, created_ms, duration_s, queue, champion_id, position, win, raw, "
            f"{account_col} FROM my_match ORDER BY created_ms").fetchall()
        try:            # 기준 표본을 받기 전 파일에는 없다 — 라인전 축만 빈다
            timelines = dict(conn.execute("SELECT match_id, frames FROM my_timeline"))
        except sqlite3.OperationalError:
            timelines = {}
    finally:
        conn.close()
    label = dict(accounts)
    games = []
    for match_id, created, duration, queue, champ, pos, win, raw, account in rows:
        if duration < REMAKE_S:
            continue
        g = Game(match_id, created, duration, queue, champ, pos or "", win, 0,
                 account=label.get(account, accounts[0][1] if accounts else ""))
        info = json.loads(gzip.decompress(raw))["info"]
        g.season = (info.get("gameVersion") or "").split(".")[0]
        for p in info["participants"]:
            team, where = p.get("teamId"), p.get("teamPosition") or ""
            if "puuid" in p:                     # 신원이 남아 있는 건 나뿐이다 (me.py가 지웠다)
                g.team = team
            if where in POSITIONS:
                g.lineup.setdefault(team, {})[where] = int(p.get("championId", 0))
            if p.get("mate"):
                g.mates.append(p["mate"])
        g.duel = _duel(g, info["participants"], timelines.get(match_id))   # 내 팀을 안 뒤에
        if g.team:
            games.append(g)
    _cache.update(mtime=mtime, games=games, me=me, accounts=[r for _p, r in accounts])


def games() -> list[Game]:
    """다시하기를 뺀 내 경기, 오래된 것부터."""
    _load()
    return _cache["games"]


def riot_id() -> str:
    """계정이 여럿이면 " + "로 잇는다."""
    _load()
    return " + ".join(_cache["accounts"]) or _cache["me"].get("riot_id", "")


def available() -> bool:
    return bool(games())


# ---------------------------------------------------------------- 풀 · 기록

def record() -> dict[tuple[int, str], tuple[int, int]]:
    """(챔피언, 포지션) → (판, 이긴 판)."""
    out: dict = defaultdict(lambda: [0, 0])
    for g in games():
        if g.position:
            out[(g.champion, g.position)][0] += 1
            out[(g.champion, g.position)][1] += g.win
    return {k: (n, w) for k, (n, w) in out.items()}


BOT_PARTNER = {"BOTTOM": "UTILITY", "UTILITY": "BOTTOM"}   # 봇 둘 기준은 이 두 자리만


def matching(position: str, enemy: dict[str, int] | None = None,
             ally: dict[str, int] | None = None) -> dict[int, tuple[int, int]]:
    """이 자리에서 상대 칸·우리 칸이 주어진 챔피언이었던 판 — 내 챔피언 → (판, 이긴 판).

    enemy={"UTILITY": 147}이면 맞은편 서폿이 세라핀이었던 판, ally={"BOTTOM": 222}면 우리 원딜이 징크스였던 판.
    """
    enemy, ally = enemy or {}, ally or {}
    out: dict = defaultdict(lambda: [0, 0])
    for g in games():
        if g.position != position:
            continue
        theirs = g.lineup.get(200 if g.team == 100 else 100, {})
        ours = g.lineup.get(g.team, {})
        if all(theirs.get(p) == c for p, c in enemy.items()) and all(ours.get(p) == c for p, c in ally.items()):
            out[g.champion][0] += 1
            out[g.champion][1] += g.win
    return {k: (n, w) for k, (n, w) in out.items()}


def pool() -> dict[str, list[int]]:
    """자리마다 5판 이상 한 챔피언, 판 수 순으로 최대 8개. 승률은 보지 않는다."""
    by_pos: dict[str, list] = defaultdict(list)
    for (champ, pos), (n, _w) in record().items():
        if n >= POOL_MIN:
            by_pos[pos].append((n, champ))
    return {pos: [c for _n, c in sorted(items, reverse=True)[:POOL_MAX]] for pos, items in by_pos.items()}


# ---------------------------------------------------------------- 조합 기대

def my_logits(rec, gs: list[Game]) -> tuple[np.ndarray, np.ndarray]:
    """판마다 모델이 본 내 팀 로짓 (과신 보정 기울기를 곱하고 사이드 이점은 넣는다).

    돌려주는 것: (로짓, 열 자리가 다 찬 판인가). 다 안 찬 판의 로짓은 0이다.
    모델에 없는 (챔피언, 포지션)은 그 포지션 평균 챔피언으로 둔다 — 추천 화면과 같다.
    """
    ok = np.array([g.full for g in gs], dtype=bool)
    X = np.zeros((len(gs), 10), dtype=np.int64)
    for i, g in enumerate(gs):
        if not ok[i]:
            continue
        for p, pos in enumerate(POSITIONS):
            for offset, team in ((0, 100), (5, 200)):
                champ = g.lineup[team][pos]
                X[i, offset + p] = rec.slot_of.get((champ, p), rec.unknown[p])
    out = np.zeros(len(gs))
    if ok.any():
        tiers = np.full(int(ok.sum()), rec.tier_index(NO_CORRECTION_TIER), dtype=np.int64)
        blue = rec.model.logits(X[ok], tiers)
        sign = np.array([1.0 if g.team == 100 else -1.0 for g in gs])[ok]
        out[ok] = rec.scale * blue * sign
    return out, ok


# ---------------------------------------------------------------- 숙련도

def fit_personal(o: np.ndarray, y: np.ndarray, groups: list[np.ndarray], sds: list[float],
                 iters: int = 30) -> list[np.ndarray]:
    """o를 고정 출발점으로 두고 그 위에 얹는 그룹 값들을 푼다 (가우스 사전분포, 뉴턴법).

    groups[k]는 판마다 그 그룹의 번호(0부터), sds[k]는 그 그룹 값의 사전 표준편차.
    """
    sizes = [int(g.max()) + 1 if len(g) else 0 for g in groups]
    starts = np.cumsum([0] + sizes)
    Z = np.zeros((len(o), starts[-1]))
    prec = np.zeros(starts[-1])
    for k, g in enumerate(groups):
        Z[np.arange(len(o)), starts[k] + g] = 1.0
        prec[starts[k]:starts[k + 1]] = 1.0 / sds[k] ** 2
    theta = np.zeros(starts[-1])
    for _ in range(iters):
        p = sigmoid(o + Z @ theta)
        grad = Z.T @ (p - y) + prec * theta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(prec)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.abs(step).max() < 1e-8:
            break
    return [theta[starts[k]:starts[k + 1]] for k in range(len(groups))]


# ---------------------------------------------------------------- 내 기록 화면

def _pct(w: float, n: int):
    return round(w / n * 100, 1) if n else None


# ---------------------------------------------------------------- 승패 육각형

BENCHMARK = ROOT / "models" / "benchmark.json"   # `python -m src.riot.benchmark`가 만든 티어별 기준
HEX_MIN_S = 900          # 15분 안에 끝난 판은 세지 않는다
HEX_MIN_GAMES = 10       # 고른 묶음이 이만큼은 돼야 그린다
HEX_STATS = ("gold15", "gold", "dmg", "kp", "deaths", "vision", "cs", "kda", "solo", "ctrl")
LOWER_IS_BETTER = {"deaths"}


def _duel(g: Game, people: list[dict], frames: str | None) -> dict | None:
    """나와 맞은편 라이너의 숫자. 숫자 뽑는 법은 src/riot/benchmark.py와 같아야 기준과 비교된다."""
    if g.queue != 420 or g.duration_s < HEX_MIN_S or g.position not in POSITIONS:
        return None
    me = next((p for p in people if "puuid" in p), None)
    foe = next((p for p in people if p.get("teamId") != g.team and p.get("teamPosition") == g.position), None)
    if me is None or foe is None:
        return None
    minutes = g.duration_s / 60
    kills = {t: sum(p.get("kills", 0) for p in people if p.get("teamId") == t) for t in (100, 200)}
    at15 = json.loads(frames) if frames else {}
    out = {}
    for side, p in (("me", me), ("foe", foe)):
        c = p.get("challenges", {})
        d = p.get("deaths", 0)
        out[side] = {"gold": p.get("goldEarned", 0) / minutes,
                     "dmg": p.get("totalDamageDealtToChampions", 0) / minutes,
                     "kp": (p.get("kills", 0) + p.get("assists", 0)) / max(kills[p.get("teamId")], 1),
                     "deaths": d, "vision": p.get("visionScore", 0) / minutes,
                     "cs": (p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)) / minutes,
                     "kda": (p.get("kills", 0) + p.get("assists", 0)) / max(d, 1),
                     "solo": c.get("soloKills", 0), "ctrl": p.get("visionWardsBoughtInGame", 0)}
        line = at15.get(str(p.get("participantId")))
        if line:
            out[side]["gold15"] = line["gold15"]
    return out


def _ahead(a, b, lower: bool):
    if a is None or b is None:
        return None
    if lower:
        a, b = b, a
    return 1 if a > b else 0.5 if a == b else 0


def hexagon() -> dict:
    """판마다 맞은편보다 앞섰나와 내 값, 그리고 티어별 기준. 묶기·섞기는 화면이 한다 (필터가 바뀔 때마다 서버에 안 묻게)."""
    gs = [g for g in games() if g.duel]
    if not gs or not BENCHMARK.exists():
        return {"available": False}
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    rows = []
    for g in gs:
        me, foe = g.duel["me"], g.duel["foe"]
        rows.append({"pos": g.position, "win": g.win, "champ": g.champion, "season": g.season, "account": g.account,
                     "lead": {k: _ahead(me.get(k), foe.get(k), k in LOWER_IS_BETTER) for k in HEX_STATS},
                     "me": {k: round(v, 3) for k, v in me.items()}})
    # 챔피언 회색선 — 표에는 모든 챔피언이 있지만 화면에는 내가 한 것만 보낸다
    played = {(r["pos"], str(r["champ"])) for r in rows}
    champs = {pos: {c: v for c, v in by.items() if (pos, c) in played} for pos, by in bench.get("champs", {}).items()}
    return {"available": True, "games": rows, "patch": bench["patch"], "season": bench["patch"].split(".")[0],
            "tiers": bench["tiers"], "minGames": HEX_MIN_GAMES,
            "champs": champs, "posPooled": bench.get("posPooled", {}), "champMin": bench.get("champMin")}


def summary(rec=None, lang: str | None = None) -> dict:
    gs = games()
    if not gs:
        return {"available": False}
    if rec is not None:
        logit, ok = my_logits(rec, gs)
        expect = sigmoid(logit)
    else:
        ok, expect = np.zeros(len(gs), dtype=bool), np.zeros(len(gs))

    def block(idx: list[int]) -> dict:
        n = len(idx)
        w = sum(gs[i].win for i in idx)
        out = {"games": n, "winrate": _pct(w, n)}
        valid = [i for i in idx if ok[i]]
        if valid:
            exp = float(np.mean(expect[valid])) * 100
            act = sum(gs[i].win for i in valid) / len(valid) * 100
            out.update(expected=round(exp, 1), diff=round(act - exp, 1))
        return out

    by = lambda key: _group(range(len(gs)), key)
    fmt = lambda ms: time.strftime("%Y-%m-%d", time.localtime(ms / 1000))

    positions = [{"position": p, **block(ix)} for p, ix in by(lambda i: gs[i].position).items()
                 if p in POSITIONS]
    positions.sort(key=lambda r: -r["games"])

    champs, rest = [], []
    for (c, p), ix in by(lambda i: (gs[i].champion, gs[i].position)).items():
        (champs if len(ix) >= TAG_MIN else rest).append(((c, p), ix))
    champions = sorted(({"championId": c, "position": p, **block(ix)} for (c, p), ix in champs),
                       key=lambda r: -r["games"])
    rest_ix = [i for _k, ix in rest for i in ix]

    months = [{"month": m, **block(ix)} for m, ix in sorted(
        by(lambda i: time.strftime("%Y-%m", time.localtime(gs[i].created_ms / 1000))).items())]

    lane = [{"championId": c, **block(ix)} for c, ix in by(lambda i: gs[i].foe).items()
            if c is not None and len(ix) >= 5]
    lane.sort(key=lambda r: (-r["games"], r["winrate"]))

    mate_ix: dict = defaultdict(list)
    for i, g in enumerate(gs):
        for m in g.mates:
            mate_ix[m].append(i)
    mates = sorted((block(ix) for ix in mate_ix.values() if len(ix) >= 3), key=lambda r: -r["games"])
    for k, r in enumerate(mates, 1):
        r["label"] = L(f"동료 {k}", f"Teammate {k}", lang)

    cuts = [(0, 25, L("25분 미만", "Under 25 min", lang)), (25, 30, L("25~30분", "25–30 min", lang)),
            (30, 35, L("30~35분", "30–35 min", lang)), (35, 999, L("35분 이상", "35+ min", lang))]
    duration = [{"label": lab, **block([i for i, g in enumerate(gs) if lo * 60 <= g.duration_s < hi * 60])}
                for lo, hi, lab in cuts]

    return {
        "available": True,
        "riotId": riot_id(),
        "from": fmt(gs[0].created_ms), "to": fmt(gs[-1].created_ms),
        "remakeNote": L(f"{REMAKE_S // 60}분 안에 끝난 다시하기는 뺐습니다",
                        f"Remakes ending within {REMAKE_S // 60} min are excluded", lang),
        "hasModel": rec is not None,
        "all": block(list(range(len(gs)))),
        "positions": positions,
        "champions": champions,
        "rest": {"champions": len(rest), **block(rest_ix)},
        "months": months,
        "lane": lane[:12],
        "mates": mates,
        "duration": duration,
        "accounts": sorted(({"riotId": a, **block(ix)} for a, ix in by(lambda i: gs[i].account).items()),
                           key=lambda r: -r["games"]) if len(_cache["accounts"]) > 1 else [],
    }


def _group(indices, key) -> dict:
    out: dict = defaultdict(list)
    for i in indices:
        out[key(i)].append(i)
    return out
