"""로딩 화면 — 10명의 최근 판으로 연승·연패, 모스트, 최근 안 한 챔피언을 고르고 음성으로 말할 문장 세 개를 정한다.

자리는 게임이 알려 주면 그대로, 모르면 밴픽 때 본 우리 팀 자리 → 챔피언으로 추정한 자리 순서로 채운다.
추정이 불확실한 사람은 음성에서 뺀다 (엉뚱한 자리를 말하지 않도록). 나는 음성에서 뺀다.
"""
from __future__ import annotations

from collections import Counter

from src.model.positions import POSITIONS, infer_positions

SR = {400, 420, 430, 440, 480, 490, 700}     # 소환사의 협곡 (일반 · 솔로 · 자유 · 빠른 대전 · 격전)
REMAKE_SEC = 300
STREAK_MIN = 3
STREAK_MAX = 10                              # 10 은 "십연패 이상"
MOST_MIN = 5                                 # 최근 판에서 이 챔피언이 가장 많고, 이만큼은 했어야 모스트
NEW_MIN = 10                                 # 최근 판이 이만큼 있는데 이 챔피언이 0판이면 "최근에 안 했어요"
SAY_MAX = 3


def summarize(p: dict) -> dict:
    games = sorted((g for g in p.get("games") or [] if g.get("queue") in SR and g.get("dur", 0) >= REMAKE_SEC),
                   key=lambda g: -g["t"])
    streak = None
    if games:
        n = next((i for i, g in enumerate(games) if g["win"] != games[0]["win"]), len(games))
        streak = {"kind": "win" if games[0]["win"] else "lose", "n": n}
    on = [g for g in games if g.get("champ") == p.get("champ")]
    counts = Counter(g.get("champ") for g in games).most_common()
    top = counts[0][1] if counts else 0
    return {
        "recent": len(games), "wins": sum(g["win"] for g in games), "streak": streak,
        "champGames": len(on), "champWins": sum(g["win"] for g in on),
        "most": bool(on) and len(on) == top and top >= MOST_MIN,
        "new": len(games) >= NEW_MIN and not on,
        "mains": [c for c, _ in counts[:3]],
    }


def place(players: list[dict], ally_pos: dict[int, str], games_of) -> None:
    """자리가 비어 있는 사람을 채운다. p["posGuess"] = True 면 추정이 불확실하다."""
    for team in ("a", "e"):
        group = [p for p in players if p["team"] == team]
        for p in group:
            p["posGuess"] = False
            if p.get("pos") not in POSITIONS:
                p["pos"] = ally_pos.get(p["champ"], "") if team == "a" else ""
        taken = tuple(p["pos"] for p in group if p["pos"] in POSITIONS)
        rest = [p for p in group if p["pos"] not in POSITIONS]
        for p, guess in zip(rest, infer_positions([p["champ"] for p in rest], games_of, taken=taken)):
            p["pos"], p["posGuess"] = guess["position"], bool(guess["uncertain"])


def say(players: list[dict]) -> list[str]:
    """말할 클립 id — 연패 > 연승 > 모스트 > 최근 안 한 챔피언, 같은 순위면 긴 연속 · 상대 먼저. 한 사람에 한 문장."""
    cand = []
    for p in players:
        if p.get("me") or p.get("posGuess") or p.get("pos") not in POSITIONS:
            continue
        s, base = p["stat"], f"l_{p['team']}_{p['pos']}_"
        enemy_first = 0 if p["team"] == "e" else 1
        st = s["streak"]
        if st and st["n"] >= STREAK_MIN:
            n = min(st["n"], STREAK_MAX)
            cand.append(((0 if st["kind"] == "lose" else 1, -n, enemy_first), p["puuid"], f"{base}{st['kind']}{n}"))
        if s["most"]:
            cand.append(((2, -s["champGames"], enemy_first), p["puuid"], base + "most"))
        if s["new"]:
            cand.append(((3, 0, enemy_first), p["puuid"], base + "new"))
    out, used = [], set()
    for _, who, cid in sorted(cand):
        if who not in used and len(out) < SAY_MAX:
            out.append(cid)
            used.add(who)
    return out


def build(data: dict | None, ally_pos: dict[int, str], games_of) -> dict:
    if not data or not data.get("players"):
        return {"available": False}
    players = data["players"]
    for p in players:
        p["stat"] = summarize(p)
        p.pop("games", None)              # 화면에는 요약만
    place(players, ally_pos, games_of)
    order = {pos: i for i, pos in enumerate(POSITIONS)}
    players.sort(key=lambda p: (p["team"] != "a", order.get(p["pos"], 9)))
    for p in players:
        p.pop("puuid", None)
    ids = say([{**p, "puuid": i} for i, p in enumerate(players)])
    return {"available": True, "gameId": data["gameId"], "players": players, "say": ids}
