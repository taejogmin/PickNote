"""클라이언트의 밴픽 세션 JSON을 화면이 쓰는 모양으로 푼다.

    myTeam[]    cellId · championId · championPickIntent(호버) · assignedPosition
    theirTeam[] cellId · championId            (상대 포지션은 주지 않는다)
    actions[][] actorCellId · championId · type(pick/ban) · completed · isAllyAction
    localPlayerCellId · timer.phase
"""

from __future__ import annotations

POSITION_KEYS = {
    "top": "TOP", "jungle": "JUNGLE", "middle": "MIDDLE", "mid": "MIDDLE",
    "bottom": "BOTTOM", "bot": "BOTTOM", "utility": "UTILITY", "support": "UTILITY",
}


def _position(raw) -> str | None:
    return POSITION_KEYS.get(str(raw or "").strip().lower())


def parse_session(session: dict) -> dict:
    local = session.get("localPlayerCellId")
    actions = [a for group in (session.get("actions") or []) for a in (group or [])]

    locked: dict[int, int] = {}          # 칸 번호 → 확정 챔피언
    in_progress: dict[int, int] = {}     # 칸 번호 → 픽 차례에 올려 둔 챔피언
    cells_with_pick_action = set()
    bans: list[int] = []
    for a in actions:
        cell, champ, kind = a.get("actorCellId"), a.get("championId") or 0, a.get("type")
        if kind == "pick":
            cells_with_pick_action.add(cell)
            if champ and a.get("completed"):
                locked[cell] = champ
            elif champ and a.get("isInProgress"):
                in_progress[cell] = champ
        elif kind == "ban" and champ and a.get("completed"):
            bans.append(champ)
    old_bans = session.get("bans") or {}
    bans += list(old_bans.get("myTeamBans") or []) + list(old_bans.get("theirTeamBans") or [])

    def locked_champion(member) -> int:
        cell = member.get("cellId")
        if cell in locked:
            return locked[cell]
        # 행동 목록이 없는 모드(칼바람 등)는 championId만 채운다
        if cell not in cells_with_pick_action and member.get("championId"):
            return member["championId"]
        return 0

    my_role, my_pick, hover = None, 0, 0
    ally: dict[str, int] = {}
    ally_unplaced: list[int] = []
    for member in session.get("myTeam") or []:
        champ, pos = locked_champion(member), _position(member.get("assignedPosition"))
        if member.get("cellId") == local:
            my_role, my_pick = pos, champ
            hover = member.get("championPickIntent") or in_progress.get(local, 0)
            continue
        if not champ:
            continue
        if pos and pos not in ally:
            ally[pos] = champ
        else:
            ally_unplaced.append(champ)

    enemy: list[int] = []
    for member in sorted(session.get("theirTeam") or [], key=lambda m: m.get("cellId", 0)):
        champ = locked_champion(member) or member.get("championId") or 0
        if champ and champ not in enemy:
            enemy.append(champ)

    return {
        "phase": (session.get("timer") or {}).get("phase"),
        "myRole": my_role,
        "myPick": my_pick,
        "hover": hover if hover and hover != my_pick else 0,
        "ally": ally,
        "allyUnplaced": ally_unplaced,
        "enemy": enemy,
        "bans": sorted({b for b in bans if b}),
    }
