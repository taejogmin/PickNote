"""게임 후 피드백 — 한 판의 타임라인에서 장면을 찾아 잘한 점 1개 · 아쉬운 점 2~3개를 고른다.

1 오브젝트 직전 1분 안에 죽음 (그 오브젝트를 누가 먹었나)
2 3명 이상에게 잡힌 데스 — 앞뒤 20초 킬 수 · 1분 안 오브젝트로 그 싸움을 이겼나 본다. 진 싸움만 아쉬운 점, 이긴 싸움은 잘한 점 (앞라인)
3 14분 전 갱 · 로밍 데스 (상대 정글, 또는 맞라인이 아닌 상대 라이너가 낀 데스)
4 강 · 상대 정글에서 죽은 횟수
6 오브젝트 싸움에서 넣은 딜 · 받은 피해의 팀 안 순위 (1분 단위)
7 정글만 — 상대가 오브젝트를 먹을 때 반대편에서 캠프를 돌고 있었나
9 15분 골드 · 경험치 — 맞라인 상대와 차이
10 6레벨 도달 시각 — 맞라인 상대와 차이
11 와드 공백 — 내가 와드를 안 박은 가장 긴 구간
13 첫 코어템 완성 시각 — 같은 챔피언 내 평균과 차이

위치는 1분 단위, 딜량도 1분 단위라 대략이다. 시야(보였는지)는 알 수 없다.
문장은 lang 에 맞춰 한국어 · 영어로 만든다. 음성은 "voice" 값으로 클립을 고른다 (feedback_live.voice_ids).
"""
from __future__ import annotations

import math

from src.i18n import L

MONSTER = {"DRAGON": ("용", "Dragon"), "RIFTHERALD": ("전령", "Herald"), "BARON_NASHOR": ("바론", "Baron"),
           "HORDE": ("유충", "Voidgrubs"), "ATAKHAN": ("아타칸", "Atakhan")}
FIGHT_MONSTERS = set(MONSTER)
POS_NAME = {"TOP": ("탑", "Top"), "JUNGLE": ("정글", "Jungle"), "MIDDLE": ("미드", "Mid"), "BOTTOM": ("원딜", "Bot"),
            "UTILITY": ("서폿", "Support")}
OBJ_BEFORE_MS = 60_000
FIGHT_HALF_MS = 20_000
FIGHT_MIN_DMG = 500                               # 오브젝트 싸움 순위 — 그 1분에 넣은 딜 + 받은 피해가 이만큼은 돼야 싸움에 있었다고 본다
EARLY_MS = 14 * 60_000
WARD_FROM_MS = 3 * 60_000
WARD_GAP = {"UTILITY": 240, "JUNGLE": 300}       # 초 — 나머지 자리는 420
WARD_GAP_OTHER = 420
LANE_GOLD = 500                                   # 15분 골드 차이가 이만큼 넘으면 말한다
LEVEL6_SEC = 30
CORE_SEC = 90


def mmss(ms: float) -> str:
    s = int(ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


def monster_name(e: dict | None, lang: str | None = None) -> str:
    if not e:
        return ""
    if e.get("monsterSubType") == "ELDER_DRAGON":
        return L("장로 용", "Elder Dragon", lang)
    return L(*MONSTER.get(e.get("monsterType"), ("오브젝트", "objective")), lang)


def zone(x: float, y: float, my_team: int) -> str:
    """대략의 구역 — base · lane · river · own_jungle · enemy_jungle (파란 팀 = 100 = 왼쪽 아래)"""
    if (x < 4200 and y < 4200) or (x > 10600 and y > 10600):
        return "base"
    if x < 2200 or y > 12600 or y < 2200 or x > 12600 or abs(x - y) < 1100:
        return "lane"
    if abs(x + y - 14800) < 1500:
        return "river"
    blue_side = x + y < 14800
    return "own_jungle" if blue_side == (my_team == 100) else "enemy_jungle"


class Game:
    def __init__(self, match: dict, line: dict, me_pid: int, lang: str | None = None):
        self.lang = lang
        self.info = match["info"]
        self.frames = line["info"]["frames"]
        self.me = me_pid
        self.p = {q["participantId"]: q for q in self.info["participants"]}
        self.team = self.p[me_pid]["teamId"]
        self.pos = self.p[me_pid].get("teamPosition") or ""
        self.events = [e for f in self.frames for e in f.get("events", [])]
        self.kills = [e for e in self.events if e["type"] == "CHAMPION_KILL"]
        objs = [e for e in self.events if e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") in FIGHT_MONSTERS]
        for o in objs:                    # 클라이언트(LCU) 타임라인에는 팀이 없을 수 있다 — 먹은 사람의 팀
            o.setdefault("killerTeamId", self.team_of(o.get("killerId")) or o.get("teamId"))
        # 유충은 세 마리가 따로 잡힌다 — 같은 종류가 1분 안에 이어지면 하나로 본다
        self.objs = [o for i, o in enumerate(objs) if not any(
            p["monsterType"] == o["monsterType"] and 0 <= o["timestamp"] - p["timestamp"] <= 60_000 for p in objs[:i])]
        self.deaths = [e for e in self.kills if e.get("victimId") == me_pid]

    def team_of(self, pid):
        return self.p.get(pid, {}).get("teamId")

    def foes(self, e: dict) -> set:
        who = {e.get("killerId")} | set(e.get("assistingParticipantIds") or [])
        return {x for x in who if self.team_of(x) and self.team_of(x) != self.team}

    def enemy(self, position):
        return next((pid for pid, q in self.p.items() if q["teamId"] != self.team and q.get("teamPosition") == position), None)

    def lane_opps(self) -> set:
        if self.pos in ("BOTTOM", "UTILITY"):
            return {self.enemy("BOTTOM"), self.enemy("UTILITY")}
        return {self.enemy(self.pos)}

    def frame(self, minute, pid):
        if pid and 0 <= minute < len(self.frames):
            return self.frames[minute]["participantFrames"].get(str(pid))
        return None

    def obj_after(self, t: int):
        return next((o for o in self.objs if 0 < o["timestamp"] - t <= OBJ_BEFORE_MS), None)

    def fight_result(self, t: int) -> dict:
        ours = theirs = 0
        for e in self.kills:
            if abs(e["timestamp"] - t) <= FIGHT_HALF_MS:
                if self.team_of(e["victimId"]) == self.team:
                    theirs += 1
                else:
                    ours += 1
        obj = self.obj_after(t)
        obj_ours = bool(obj and obj.get("killerTeamId") == self.team)
        obj_theirs = bool(obj) and not obj_ours
        won = ours > theirs or (ours == theirs and obj_ours)
        lost = theirs > ours or (ours == theirs and obj_theirs)      # 킬이 같고 오브젝트도 없으면 어느 쪽도 아니다
        return {"ours": ours, "theirs": theirs, "obj": monster_name(obj, self.lang), "objOurs": obj_ours, "won": won, "lost": lost}


def _side(ours: bool, lang: str | None) -> str:
    """오브젝트를 누가 먹었나 — 우리 / 상대"""
    return L("우리", "ours", lang) if ours else L("상대", "theirs", lang)


def _fight_line(x: dict, lang: str | None = None) -> str:
    s = L(f"{mmss(x['t'])} {x['n']}명에게 사망 · 앞뒤 20초 킬 {x['ours']}:{x['theirs']}",
          f"{mmss(x['t'])} died to {x['n']} · kills within 20s {x['ours']}:{x['theirs']}", lang)
    if x["obj"]:
        s += f" · {x['obj']} {_side(x['objOurs'], lang)}"
    return s


def analyze(match: dict, line: dict, me_pid: int, core_avg: float | None = None, items: dict | None = None,
            lang: str | None = None) -> dict:
    g = Game(match, line, me_pid, lang)
    T = lambda ko, en: L(ko, en, lang)      # noqa: E731
    out = []          # {k, good, w, text, detail, voice}

    # 1 오브젝트 직전 데스
    od = []
    for d in g.deaths:
        o = g.obj_after(d["timestamp"])
        if o:
            od.append({"t": d["timestamp"], "obj": monster_name(o, lang), "ours": o.get("killerTeamId") == g.team})
    lost = [x for x in od if not x["ours"]]
    used = {x["t"] for x in lost}          # 데스 하나는 아쉬운 점 하나에만 센다 (오브젝트 > 한타 > 갱 > 위치)
    if lost:
        out.append({"k": "obj_death", "good": False, "w": 3 * len(lost),
                    "text": (T(f"{lost[0]['obj']} 1분 전에 죽었고, {lost[0]['obj']}을 상대가 가져갔어요.",
                               f"You died within a minute before {lost[0]['obj']}, and the enemy took it.") if len(lost) == 1
                             else T(f"오브젝트 1분 전에 {len(lost)}번 죽었고, 그 오브젝트는 모두 상대가 가져갔어요.",
                                    f"You died {len(lost)} times within a minute before an objective, and the enemy took each one.")),
                    "detail": [T(f"{mmss(x['t'])} 사망 → {x['obj']} {_side(x['ours'], lang)}",
                                 f"{mmss(x['t'])} died → {x['obj']} {_side(x['ours'], lang)}") for x in od],
                    "voice": ["obj_death", len(lost)]})

    # 2 3명 이상에게 잡힘 — 싸움 결과로 나눈다
    lost_f, won_f = [], []
    for d in g.deaths:
        n = len(g.foes(d))
        if n >= 3 and d["timestamp"] not in used:
            r = g.fight_result(d["timestamp"])
            if r["won"] or r["lost"]:
                (won_f if r["won"] else lost_f).append({"t": d["timestamp"], "n": n, **r})
    used |= {x["t"] for x in lost_f} | {x["t"] for x in won_f}
    if lost_f:
        out.append({"k": "fight_lost", "good": False, "w": 2.5 * len(lost_f),
                    "text": T(f"상대 셋 이상에게 잡히고 그 싸움도 진 게 {len(lost_f)}번이에요.",
                              f"{len(lost_f)} times you were caught by three or more enemies and lost that fight."),
                    "detail": [_fight_line(x, lang) for x in lost_f], "voice": ["fight_lost", len(lost_f)]})
    if won_f:
        out.append({"k": "fight_won", "good": True, "w": 2 * len(won_f),
                    "text": T(f"앞에서 받아 주다 죽었지만 그 싸움은 이긴 게 {len(won_f)}번이에요.",
                              f"{len(won_f)} times you died holding the front line, but your team won that fight."),
                    "detail": [_fight_line(x, lang) for x in won_f], "voice": ["fight_won", len(won_f)]})

    # 3 초반 갱 · 로밍 데스
    if g.pos and g.pos != "JUNGLE":
        jg, opps, gank = g.enemy("JUNGLE"), g.lane_opps(), []
        for d in g.deaths:
            if d["timestamp"] > EARLY_MS or d["timestamp"] in used:
                continue
            who = g.foes(d)
            tags = ([T("정글", "Jungle")] if jg in who else []) + [T(*POS_NAME.get(g.p[x].get("teamPosition"), ("?", "?")))
                                                               for x in who if x != jg and x not in opps]
            if tags:
                gank.append({"t": d["timestamp"], "tags": tags})
        used |= {x["t"] for x in gank}
        if gank:
            out.append({"k": "gank", "good": False, "w": 1.5 * len(gank) if len(gank) >= 2 else 1,
                        "text": T(f"14분 전에 갱이나 로밍으로 {len(gank)}번 죽었어요.",
                                  f"You died to ganks or roams {len(gank)} times before 14 minutes."),
                        "detail": [T(f"{mmss(x['t'])} 상대 {'·'.join(x['tags'])}", f"{mmss(x['t'])} enemy {'·'.join(x['tags'])}")
                                   for x in gank], "voice": ["gank", len(gank)]})

    # 4 강 · 상대 정글 데스
    spot = []
    for d in g.deaths:
        z = zone(d["position"]["x"], d["position"]["y"], g.team) if d.get("position") and d["timestamp"] not in used else ""
        if z in ("river", "enemy_jungle"):
            spot.append({"t": d["timestamp"], "z": T("강", "river") if z == "river" else T("상대 정글", "enemy jungle")})
    if len(spot) >= 2:
        out.append({"k": "spot", "good": False, "w": 1.0 * len(spot),
                    "text": T(f"강이나 상대 정글에서 {len(spot)}번 죽었어요.", f"You died {len(spot)} times in the river or the enemy jungle."),
                    "detail": [f"{mmss(x['t'])} {x['z']}" for x in spot], "voice": ["spot", len(spot)]})

    # 6 오브젝트 싸움 딜 · 받은 피해 순위
    fights = []
    mates = [pid for pid, q in g.p.items() if q["teamId"] == g.team]
    for o in g.objs:
        t = o["timestamp"]
        if not any(-30_000 <= e["timestamp"] - t <= 15_000 for e in g.kills):
            continue                      # 싸움 없이 먹은 오브젝트
        m = t // 60_000
        dealt, taken = {}, {}
        for pid in mates:
            a, b = g.frame(m, pid), g.frame(m + 1, pid)
            if not a or not b or "damageStats" not in a or "damageStats" not in b:
                break                     # 클라이언트 타임라인에는 1분 딜량이 없을 수 있다
            dealt[pid] = b["damageStats"]["totalDamageDoneToChampions"] - a["damageStats"]["totalDamageDoneToChampions"]
            taken[pid] = b["damageStats"]["totalDamageTaken"] - a["damageStats"]["totalDamageTaken"]
        if len(dealt) != len(mates):
            continue
        if dealt[g.me] + taken[g.me] < FIGHT_MIN_DMG:
            continue                      # 그 싸움에 거의 없었다
        rank = lambda d: 1 + sum(v > d[g.me] for v in d.values())
        fights.append({"t": t, "obj": monster_name(o, lang), "ours": o.get("killerTeamId") == g.team,
                       "dealt": dealt[g.me], "dealtRank": rank(dealt), "taken": taken[g.me], "takenRank": rank(taken)})
    if fights:
        det = [T(f"{mmss(f['t'])} {f['obj']} {_side(f['ours'], lang)} · 딜 {f['dealtRank']}등 ({f['dealt']:,}) · 받은 피해 {f['takenRank']}등 ({f['taken']:,})",
                 f"{mmss(f['t'])} {f['obj']} {_side(f['ours'], lang)} · damage #{f['dealtRank']} ({f['dealt']:,}) · taken #{f['takenRank']} ({f['taken']:,})")
               for f in fights]
        front = g.pos == "UTILITY"
        size = lambda f: -(f["dealt"] + f["taken"])                 # 가장 큰 싸움부터
        top = sorted([f for f in fights if f["dealtRank"] == 1 or (front and f["takenRank"] <= 2 and f["ours"])], key=size)
        low = sorted([f for f in fights if f["dealtRank"] >= 4 and f["takenRank"] >= 4 and not f["ours"]], key=size)
        if top:
            f = top[0]
            text = (T(f"{f['obj']} 싸움에서 딜 {f['dealtRank']}등으로 가장 많이 넣었어요.",
                      f"You dealt the most damage on your team in the {f['obj']} fight.") if f["dealtRank"] == 1
                    else T(f"{f['obj']} 싸움에서 받은 피해 {f['takenRank']}등으로 앞에서 받아 줬고, 오브젝트도 가져왔어요.",
                           f"You were #{f['takenRank']} in damage taken in the {f['obj']} fight, holding the front, and your team got the objective."))
            out.append({"k": "obj_fight_top", "good": True, "w": 1.5 * len(top), "text": text,
                        "detail": det, "voice": ["obj_fight", f["dealtRank"], f["takenRank"]]})
        if low:
            f = low[0]
            out.append({"k": "obj_fight_low", "good": False, "w": 1.2 * len(low),
                        "text": T(f"{f['obj']} 싸움에서 딜 {f['dealtRank']}등, 받은 피해 {f['takenRank']}등이었고 오브젝트도 내줬어요.",
                                  f"In the {f['obj']} fight you were #{f['dealtRank']} in damage and #{f['takenRank']} in damage taken, and the objective was lost."),
                        "detail": det, "voice": ["obj_fight", f["dealtRank"], f["takenRank"]]})

    # 7 정글 — 상대 오브젝트 때 반대편 캠프
    if g.pos == "JUNGLE":
        away = []
        for o in g.objs:
            if o.get("killerTeamId") == g.team or not o.get("position"):
                continue
            m = o["timestamp"] // 60_000
            a, b = g.frame(m, g.me), g.frame(m + 1, g.me)
            if not a or not b:
                continue
            here = (o["position"]["x"], o["position"]["y"])
            dist = min(math.dist(here, (f["position"]["x"], f["position"]["y"])) for f in (a, b))
            farm = b["jungleMinionsKilled"] - a["jungleMinionsKilled"]
            if dist > 5000 and farm > 0:
                away.append({"t": o["timestamp"], "obj": monster_name(o, lang), "farm": farm})
        if away:
            out.append({"k": "jg_away", "good": False, "w": 2.5 * len(away),
                        "text": T(f"상대가 오브젝트를 먹을 때 반대편에서 캠프를 돈 게 {len(away)}번이에요.",
                                  f"{len(away)} times you were clearing camps on the other side while the enemy took an objective."),
                        "detail": [T(f"{mmss(x['t'])} {x['obj']} · 그 1분에 정글 몹 {x['farm']}",
                                     f"{mmss(x['t'])} {x['obj']} · {x['farm']} jungle monsters that minute") for x in away],
                        "voice": ["jg_away", len(away)]})

    # 9 15분 골드 · 경험치
    opp = g.enemy(g.pos)
    a, b = g.frame(15, g.me), g.frame(15, opp)
    if a and b:
        dg, dx = a["totalGold"] - b["totalGold"], a["xp"] - b["xp"]
        if abs(dg) >= LANE_GOLD:
            out.append({"k": "lane15", "good": dg > 0, "w": abs(dg) / 500,
                        "text": T(f"15분에 맞라인보다 골드가 {abs(dg):,} {'앞섰' if dg > 0 else '뒤졌'}어요.",
                                  f"At 15 minutes you were {abs(dg):,} gold {'ahead of' if dg > 0 else 'behind'} your lane opponent."),
                        "detail": [T(f"골드 {dg:+,} · 경험치 {dx:+,}", f"Gold {dg:+,} · XP {dx:+,}")], "voice": ["lane15", dg]})

    # 10 6레벨
    def lv6(pid):
        t = next((e["timestamp"] for e in g.events if e["type"] == "LEVEL_UP" and e.get("participantId") == pid and e.get("level") == 6), None)
        if t is None and not any(e["type"] == "LEVEL_UP" for e in g.events):     # 레벨업 기록이 없는 타임라인 — 1분 단위로
            t = next((m * 60_000 for m in range(len(g.frames)) if (g.frame(m, pid) or {}).get("level", 0) >= 6), None)
        return t
    m6, o6 = lv6(g.me), (lv6(opp) if opp else None)
    if m6 and o6 and abs(m6 - o6) >= LEVEL6_SEC * 1000:
        late = m6 > o6
        out.append({"k": "lv6", "good": not late, "w": abs(m6 - o6) / 60_000,
                    "text": T(f"6레벨이 맞라인보다 {abs(m6 - o6) // 1000}초 {'늦었' if late else '빨랐'}어요.",
                              f"You hit level 6 {abs(m6 - o6) // 1000}s {'later' if late else 'earlier'} than your lane opponent."),
                    "detail": [T(f"나 {mmss(m6)} · 상대 {mmss(o6)}", f"You {mmss(m6)} · opponent {mmss(o6)}")], "voice": ["lv6", (m6 - o6) // 1000]})

    # 11 와드 공백
    wards = sorted(e["timestamp"] for e in g.events if e["type"] == "WARD_PLACED" and e.get("creatorId") == g.me
                   and e.get("wardType") not in (None, "UNDEFINED"))
    end = g.frames[-1]["timestamp"]
    marks = [WARD_FROM_MS] + [t for t in wards if t > WARD_FROM_MS] + [end]
    gaps = sorted(((y - x, x, y) for x, y in zip(marks, marks[1:])), reverse=True)
    limit = WARD_GAP.get(g.pos, WARD_GAP_OTHER) * 1000
    if gaps and gaps[0][0] > limit:
        gap, s, e = gaps[0]
        out.append({"k": "ward_gap", "good": False, "w": (gap - limit) / 60_000 + 1,
                    "text": T(f"{mmss(s)}부터 {mmss(e)}까지 {gap // 60_000}분 넘게 와드를 안 박았어요.",
                              f"You placed no wards for over {gap // 60_000} minutes, from {mmss(s)} to {mmss(e)}."),
                    "detail": [T(f"{mmss(x)} ~ {mmss(y)} ({(y - x) // 1000}초)", f"{mmss(x)} – {mmss(y)} ({(y - x) // 1000}s)")
                               for n, x, y in gaps[:3] if n > limit],
                    "voice": ["ward_gap", gap // 60_000]})
    elif gaps and g.pos == "UTILITY":
        out.append({"k": "ward_ok", "good": True, "w": 0.8,
                    "text": T(f"와드 공백이 가장 길어도 {gaps[0][0] // 1000}초였어요.",
                              f"Your longest stretch without placing a ward was only {gaps[0][0] // 1000}s."),
                    "detail": [], "voice": ["ward_ok"]})

    # 13 첫 코어템
    t_core = first_core(g, items or {})
    if t_core and core_avg:
        d = (t_core - core_avg) / 1000
        if abs(d) >= CORE_SEC:
            out.append({"k": "core", "good": d < 0, "w": abs(d) / 120,
                        "text": T(f"첫 코어템이 이 챔피언 평소보다 {abs(d) / 60:.1f}분 {'늦었' if d > 0 else '빨랐'}어요.",
                                  f"Your first core item came {abs(d) / 60:.1f} min {'later' if d > 0 else 'earlier'} than usual on this champion."),
                        "detail": [T(f"이번 {mmss(t_core)} · 평소 {mmss(core_avg)}", f"This game {mmss(t_core)} · usual {mmss(core_avg)}")], "voice": ["core", round(d / 60)]})

    good = sorted((x for x in out if x["good"]), key=lambda x: -x["w"])[:1]
    bad = sorted((x for x in out if x["good"] is False), key=lambda x: -x["w"])[:3]
    return {"good": good, "bad": bad, "all": out, "fights": fights, "pos": g.pos, "core": t_core,
            "champ": g.p[g.me]["championId"], "win": bool(g.p[g.me]["win"]), "deaths": len(g.deaths)}


def first_core(g: Game, items: dict) -> int | None:
    undone = {(e.get("participantId"), e.get("beforeId")) for e in g.events if e["type"] == "ITEM_UNDO"}
    for e in g.events:
        if (e["type"] == "ITEM_PURCHASED" and e.get("participantId") == g.me
                and (items.get(e.get("itemId")) or {}).get("core") and (g.me, e.get("itemId")) not in undone):
            return e["timestamp"]
    return None
