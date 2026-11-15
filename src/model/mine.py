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
FOE_MIN = 3             # 챔피언 줄을 펼쳤을 때 상대 라이너를 따로 세우는 최소 판 수
FOE_DETAIL_MIN = 5      # 그 상대로 지표(앞선 비율)까지 보이는 최소 판 수 — 이보다 적으면 거의 우연이다
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
    kda: tuple = (0, 0, 0)                       # 내 킬 · 데스 · 어시스트
    form: dict | None = None                     # 한눈에 지표 재료 — 솔로랭크 15분 넘고 타임라인 있는 판만
    gaps: dict | None = None                     # 시간대별 격차 — 맞은편과 CS·골드·경험치 차이, 5분마다

    @property
    def foe(self) -> int | None:
        """같은 포지션의 상대 챔피언 — 상대 라이너."""
        other = 200 if self.team == 100 else 100
        return self.lineup.get(other, {}).get(self.position)

    @property
    def full(self) -> bool:
        return all(len(self.lineup.get(t, {})) == 5 for t in (100, 200))


_cache: dict = {"mtime": None, "games": [], "me": {}, "accounts": [], "ranks": []}


def _load() -> None:
    if not DB.exists():
        _cache.update(mtime=None, games=[], me={}, accounts=[], ranks=[])
        return
    mtime = DB.stat().st_mtime
    if mtime == _cache["mtime"]:
        return
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)   # 내 기록 갱신이 쓰는 중이면 기다린다
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
        try:            # 5분마다 숫자를 받기 전 파일에는 없다 — 시간대별 격차 칸만 빈다
            curves = dict(conn.execute("SELECT match_id, curve FROM my_curve WHERE curve IS NOT NULL"))
        except sqlite3.OperationalError:
            curves = {}
        try:            # 티어 기록을 찍기 전 파일에는 없다 — 그래프가 추정만 그린다
            ranks = conn.execute("SELECT account, ts_ms, queue, tier, division, lp, wins, losses "
                                 "FROM rank_snapshot ORDER BY ts_ms").fetchall()
        except sqlite3.OperationalError:
            ranks = []
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
                g.kda = (int(p.get("kills", 0)), int(p.get("deaths", 0)), int(p.get("assists", 0)))
            if where in POSITIONS:
                g.lineup.setdefault(team, {})[where] = int(p.get("championId", 0))
            if p.get("mate"):
                g.mates.append(p["mate"])
        g.duel = _duel(g, info["participants"], timelines.get(match_id))   # 내 팀을 안 뒤에
        g.form = _form(g, info["participants"], timelines.get(match_id))
        g.gaps = _gaps(g, info["participants"], curves.get(match_id))
        if g.team:
            games.append(g)
    ranks = [{"a": label.get(a, ""), "t": t, "q": q, "tier": tier, "div": div, "lp": lp, "wins": w, "losses": l}
             for a, t, q, tier, div, lp, w, l in ranks if a in label]
    _cache.update(mtime=mtime, games=games, me=me, accounts=[r for _p, r in accounts], ranks=ranks)


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


def _form(g: Game, people: list[dict], frames: str | None) -> dict | None:
    """한눈에 · 지표 칸 재료. 사람마다 숫자는 src/riot/benchmark.py의 people()과 같은 꼴로 만들어 같은 식을 쓴다."""
    if g.duel is None or not frames:
        return None
    from src.riot.benchmark import end_stats, lane_share, mates_share, snowball, team_shares, tile_values
    at15 = json.loads(frames)
    minutes = g.duration_s / 60
    kills = {t: sum(p.get("kills", 0) for p in people if p.get("teamId") == t) for t in (100, 200)}
    seat, wins = {}, {}
    for p in people:
        line = at15.get(str(p.get("participantId")))
        pos = p.get("teamPosition") or ""
        if not line or pos not in POSITIONS:
            return None
        seat[p["teamId"], pos] = {**end_stats(p, kills[p["teamId"]], minutes), **line}
    if len(seat) != 10:
        return None
    me, foe = seat[g.team, g.position], seat[300 - g.team, g.position]
    team = [v for (t, _q), v in seat.items() if t == g.team]
    mates = [v for (t, q), v in seat.items() if t == g.team and q != g.position]
    rivals = [v for (t, q), v in seat.items() if t != g.team and q != g.position]
    return {"lane": lane_share(me, foe),
            "laneG": 100 * me["gold15"] / max(me["gold15"] + foe["gold15"], 1),
            "laneX": 100 * me["xp15"] / max(me["xp15"] + foe["xp15"], 1),
            "shares": team_shares(me, team),
            "behind": sum(v["gold15"] for v in mates) < sum(v["gold15"] for v in rivals),
            "tiles": {**tile_values(me, team, foe, minutes), "mates": mates_share(mates, rivals)},
            "snow": snowball(me, foe, g.win)}


def _gaps(g: Game, people: list[dict], curve: str | None) -> dict | None:
    """나 − 맞은편 라이너, 5분마다 (그 판이 간 데까지). 참가자 번호는 경기 원본에서 찾는다."""
    if not curve or g.queue != 420 or g.position not in POSITIONS:
        return None
    c = json.loads(curve)
    me = next((p for p in people if "puuid" in p), None)
    foe = next((p for p in people if p.get("teamId") != g.team and p.get("teamPosition") == g.position), None)
    if me is None or foe is None:
        return None
    a, b = c.get(str(me.get("participantId"))), c.get(str(foe.get("participantId")))
    if not a or not b:
        return None
    return {k: [x - y for x, y in zip(a[k], b[k])] for k in ("cs", "gold", "xp")}


def curve() -> dict:
    """시간대별 격차 — 판마다 나 − 맞은편, 그리고 티어·자리·챔피언 기준. 묶기는 화면이 한다."""
    gs = [g for g in games() if g.gaps]
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8")).get("curve") if BENCHMARK.exists() else None
    if not gs or not bench or not bench.get("tiers"):
        return {"available": False, "games": len(gs)}
    return {"available": True, "minutes": bench["minutes"], "sample": bench["games"],
            "games": [{"pos": g.position, "champ": g.champion, "win": g.win, "account": g.account, "gaps": g.gaps} for g in gs],
            "tiers": bench["tiers"], "posPooled": bench["posPooled"], "champs": bench["champs"]}


def form() -> dict:
    """한눈에 지표 (라인전 · 캐리력 · 인분 · 멘탈) — 판마다 재료와 티어별 기준. 묶기는 화면이 한다."""
    gs = [g for g in games() if g.form]
    if not gs or not BENCHMARK.exists():
        return {"available": False}
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8")).get("form")
    if not bench:
        return {"available": False}
    rows = []
    for g in gs:
        f, avg, q = g.form, bench["posShare"][g.position], bench["contribQ"][g.position]
        contrib = sum(f["shares"][k] / avg[k] for k in range(3)) / 3
        pctl = float(np.interp(contrib, q, np.linspace(0, 100, len(q))))
        rows.append({"pos": g.position, "win": g.win, "account": g.account, "champ": g.champion,
                     "lane": round(f["lane"], 2), "laneG": round(f["laneG"], 2), "laneX": round(f["laneX"], 2),
                     "t": g.created_ms, "dur": g.duration_s, "foe": g.foe, "gaps": g.gaps,
                     "contrib": round(contrib, 3), "pctl": round(pctl, 1), "behind": f["behind"], "snow": f["snow"],
                     "v": {k: round(v, 4) for k, v in f["tiles"].items()}})
    return {"available": True, "games": rows, "base": bench["tiers"], "q": bench.get("q", {}), "lower": bench.get("lower", [])}


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


# ---------------------------------------------------------------- 티어 그래프

LP_PER_GAME = 20        # 지난 티어를 승패로 거슬러 갈 때 한 판에 오르내린다고 보는 LP (실제는 MMR 따라 ±5쯤 다르다)


def tier_graph() -> dict:
    """솔로랭크 판마다 (계정, 시즌, 시각, 승패)와 찍어 둔 진짜 랭크. 선 긋기·시즌 고르기는 화면이 한다.

    지난 티어·LP는 Riot API에 없다 — 갱신할 때마다 찍은 랭크(rank_snapshot)만 진짜 점이고,
    그 사이와 그 전은 판마다 이기면 +LP_PER_GAME, 지면 −LP_PER_GAME으로 거슬러 간 추정이다.
    """
    gs = [g for g in games() if g.queue == 420 and g.season]
    if not gs:
        return {"available": False}
    return {"available": True, "lpPerGame": LP_PER_GAME,
            "games": [{"a": g.account, "s": g.season, "t": g.created_ms, "w": g.win} for g in gs],
            "snaps": [r for r in _cache["ranks"] if r["q"] == "RANKED_SOLO_5x5"]}


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
        out = {"games": n, "winrate": _pct(w, n), "kda": [sum(gs[i].kda[j] for i in idx) for j in range(3)]}
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
    def foes(ix: list[int]) -> dict:
        """이 챔피언으로 만난 상대 라이너 — FOE_MIN판 넘은 상대만 따로, 나머지는 한 줄로."""
        groups = _group([i for i in ix if gs[i].foe is not None], lambda i: gs[i].foe)
        rows, rest_ix = [], []
        for c, fix in groups.items():
            if len(fix) < FOE_MIN:
                rest_ix += fix
                continue
            r = {"championId": c, **block(fix)}
            duel = [gs[i] for i in fix if gs[i].duel]
            r["duelGames"] = len(duel)
            if len(duel) >= FOE_DETAIL_MIN:
                r["lead"] = {}
                for k in HEX_STATS:
                    xs = [_ahead(g.duel["me"].get(k), g.duel["foe"].get(k), k in LOWER_IS_BETTER) for g in duel]
                    xs = [x for x in xs if x is not None]
                    r["lead"][k] = round(sum(xs) / len(xs) * 100, 1) if xs else None
            rows.append(r)
        rows.sort(key=lambda r: (-r["games"], r["winrate"]))
        return {"rows": rows, "rest": {"champions": len(groups) - len(rows), **block(rest_ix)}}

    champions = sorted(({"championId": c, "position": p, **block(ix), "foes": foes(ix)} for (c, p), ix in champs),
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

    sides = [{"side": t, **block(ix)} for t, ix in sorted(by(lambda i: gs[i].team).items())]

    # 챔피언 풀 — 한 챔피언에 얼마나 기대나 (전체와 자리마다)
    def focus(ix: list[int]) -> dict:
        counts = sorted((len(cix) for cix in _group(ix, lambda i: gs[i].champion).values()), reverse=True)
        n = len(ix)
        top = sorted(_group(ix, lambda i: gs[i].champion).items(), key=lambda kv: -len(kv[1]))[:3]
        return {"games": n, "champs": len(counts),
                "top1": _pct(sum(counts[:1]), n), "top3": _pct(sum(counts[:3]), n), "top5": _pct(sum(counts[:5]), n),
                "top": [{"championId": c, **block(cix)} for c, cix in top]}
    pos_ix = by(lambda i: gs[i].position)
    champ_pool = {**focus(list(range(len(gs)))),
                  "positions": [{"position": p, **focus(pos_ix[p])} for p in POSITIONS if p in pos_ix]}

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
        "sides": sides,
        "pool": champ_pool,
        "foeMin": FOE_MIN, "foeDetailMin": FOE_DETAIL_MIN,
        "accounts": sorted(({"riotId": a, **block(ix)} for a, ix in by(lambda i: gs[i].account).items()),
                           key=lambda r: -r["games"]) if len(_cache["accounts"]) > 1 else [],
    }


def _group(indices, key) -> dict:
    out: dict = defaultdict(list)
    for i in indices:
        out[key(i)].append(i)
    return out
