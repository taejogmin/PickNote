"""롤 클라이언트(LCU)의 밴픽 창을 읽는다. 읽기만 한다.

클라이언트는 켜져 있을 때 자기 PC에 HTTPS API를 연다. 포트와 비밀번호는
설치 폴더의 lockfile에 있고, 없으면 LeagueClientUx.exe 실행 인자에서 찾는다.
자동으로 픽·밴을 누르는 기능은 만들지 않는다 — 이 파일에는 GET만 있다.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import requests

try:
    from urllib3.exceptions import InsecureRequestWarning
    # 클라이언트는 자체 서명 인증서를 쓴다. 127.0.0.1로만 붙으므로 검증을 끄고 경고도 끈다.
    warnings.simplefilter("ignore", InsecureRequestWarning)
except Exception:  # pragma: no cover
    pass

INSTALL_DIRS = [
    Path(r"C:\Riot Games\League of Legends"),
    Path(r"D:\Riot Games\League of Legends"),
    Path(r"C:\Program Files\Riot Games\League of Legends"),
    Path(r"C:\Program Files (x86)\Riot Games\League of Legends"),
]
SESSION = "/lol-champ-select/v1/session"
FLOW = "/lol-gameflow/v1/gameflow-phase"
ME = "/lol-summoner/v1/current-summoner"
LOBBY = "/lol-lobby/v2/lobby"
HISTORY = "/lol-match-history/v1/products/lol/current-summoner/matches?begIndex=0&endIndex=20"
GAME = "/lol-gameflow/v1/session"                                       # 로딩 화면부터 10명 (puuid · 챔피언 · 자리)
HISTORY_OF = "/lol-match-history/v1/products/lol/{}/matches?begIndex=0&endIndex=20"
RANKED_OF = "/lol-ranked/v1/ranked-stats/{}"
# 실행 인자 조회는 PowerShell을 띄워 1초 가까이 걸린다. 클라이언트가 꺼져 있을 때
# 1초마다 폴링하면서 매번 띄우면 PC가 무거워지므로 간격을 둔다.
PROCESS_LOOKUP_INTERVAL = 10.0


@dataclass
class Credentials:
    port: int
    password: str


def _from_lockfile() -> Credentials | None:
    """lockfile 형식: LeagueClient:pid:port:password:https"""
    for directory in INSTALL_DIRS:
        lock = directory / "lockfile"
        try:
            parts = lock.read_text(encoding="utf-8").strip().split(":")
        except OSError:
            continue
        if len(parts) >= 4 and parts[2].isdigit():
            return Credentials(int(parts[2]), parts[3])
    return None


def _from_process() -> Credentials | None:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='LeagueClientUx.exe'\").CommandLine"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    port = re.search(r"--app-port=(\d+)", out or "")
    token = re.search(r"--remoting-auth-token=([\w-]+)", out or "")
    if port and token:
        return Credentials(int(port.group(1)), token.group(1))
    return None


class FileClient:
    """개발용 — 클라이언트 대신 JSON 파일을 읽는다.

    파일이 있으면 밴픽 중, 없으면 대기로 본다. 파일을 고쳐 픽 공개를 흉내 낸다.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def champ_select(self) -> tuple[str, dict | None]:
        try:
            return "champ_select", json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "idle", None

    def recent_games(self) -> list[dict]:
        return []

    def gameflow(self) -> str:
        return ""

    def game_id(self) -> int | None:
        return (self.loading() or {}).get("gameId")

    def loading(self) -> dict | None:
        """개발용 — 가짜 세션 파일의 "loading": loading() 과 같은 모양"""
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("loading")
        except (OSError, ValueError):
            return None

    def party(self, session: dict | None = None) -> list[dict]:
        """개발용 — 가짜 세션 파일의 "party": [{"puuid", "name"}]"""
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("party") or []
        except (OSError, ValueError):
            return []


def _games(body) -> list[dict]:
    """최근 전적 응답에서 판마다 {id, t, dur, queue, win, champ}. 이 응답의 participants 는 그 사람 하나뿐이다."""
    out = []
    for g in ((body or {}).get("games") or {}).get("games") or []:
        try:
            p = g["participants"][0]
            out.append({"id": int(g["gameId"]), "t": int(g["gameCreation"]), "dur": int(g["gameDuration"]),
                        "queue": int(g.get("queueId") or 0), "win": int(bool(p["stats"]["win"])),
                        "champ": int(p.get("championId") or 0)})
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


class LeagueClient:
    """champ_select()는 (상태, 세션 JSON)을 돌려준다.

    상태: "offline"(클라이언트 꺼짐) · "idle"(켜져 있지만 밴픽 중 아님) · "champ_select"
    """

    def __init__(self, timeout: float = 1.5):
        self.timeout = timeout
        self.creds: Credentials | None = None
        self._last_process_lookup = 0.0
        self.http = requests.Session()
        self.http.verify = False

    def _credentials(self) -> Credentials | None:
        if self.creds:
            return self.creds
        self.creds = _from_lockfile()
        if not self.creds and time.monotonic() - self._last_process_lookup > PROCESS_LOOKUP_INTERVAL:
            self._last_process_lookup = time.monotonic()
            self.creds = _from_process()
        return self.creds

    def champ_select(self) -> tuple[str, dict | None]:
        creds = self._credentials()
        if not creds:
            return "offline", None
        try:
            response = self.http.get(
                f"https://127.0.0.1:{creds.port}{SESSION}",
                auth=("riot", creds.password), timeout=self.timeout,
            )
        except requests.RequestException:
            self.creds = None            # 클라이언트를 껐거나 다시 켜서 포트가 바뀌었다
            return "offline", None
        if response.status_code == 200:
            return "champ_select", response.json()
        if response.status_code in (401, 403):
            self.creds = None            # 비밀번호가 바뀌었다 (클라이언트 재시작)
            return "offline", None
        return "idle", None              # 404 — 밴픽 중이 아니다

    def recent_games(self) -> list[dict]:
        """클라이언트의 최근 전적 (기울어짐 · 오늘 판 수). 내 기록(me.db)은 손으로 갱신해야 쌓여서 오늘 판은 여기서 본다."""
        creds = self._credentials()
        if not creds:
            return []
        try:
            response = self.http.get(f"https://127.0.0.1:{creds.port}{HISTORY}",
                                     auth=("riot", creds.password), timeout=self.timeout)
            return _games(response.json()) if response.status_code == 200 else []
        except (requests.RequestException, ValueError):
            return []

    def gameflow(self) -> str:
        """게임 진행 단계 — Lobby · Matchmaking · ChampSelect · InProgress · EndOfGame … 모르면 ""."""
        creds = self.creds           # 밴픽 조회가 이미 찾아 둔 것만 쓴다 (꺼져 있으면 다시 찾지 않는다)
        if not creds:
            return ""
        try:
            response = self.http.get(f"https://127.0.0.1:{creds.port}{FLOW}", auth=("riot", creds.password), timeout=self.timeout)
            return str(response.json()) if response.status_code == 200 else ""
        except (requests.RequestException, ValueError):
            return ""

    def _get(self, path: str, timeout: float | None = None):
        creds = self.creds or self._credentials()
        if not creds:
            return None
        try:
            r = self.http.get(f"https://127.0.0.1:{creds.port}{path}", auth=("riot", creds.password), timeout=timeout or self.timeout)
            return r.json() if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None

    def party(self, session: dict | None = None) -> list[dict]:
        """같이 큐를 잡은 파티원 [{puuid, name}] (나 빼고). 랭크 밴픽에서 모르는 팀원은 가려지니 파티원만 보인다.
        로비 멤버를 먼저 보고, 밴픽 세션의 우리 팀에서 puuid 가 보이는 사람을 더한다."""
        me = (self._get(ME) or {}).get("puuid")
        out = {}
        for m in (self._get(LOBBY) or {}).get("members") or []:
            if m.get("puuid") and m["puuid"] != me:
                out[m["puuid"]] = {"puuid": m["puuid"], "name": m.get("gameName") or m.get("summonerName") or ""}
        for m in (session or {}).get("myTeam") or []:
            if m.get("puuid") and m["puuid"] != me and m.get("nameVisibilityType", "VISIBLE") != "HIDDEN":
                out.setdefault(m["puuid"], {"puuid": m["puuid"], "name": m.get("gameName") or ""})
        return list(out.values())

    def game_id(self) -> int | None:
        """지금 게임 번호 — 로딩 화면부터 있다"""
        return ((self._get(GAME) or {}).get("gameData") or {}).get("gameId") or None

    def loading(self) -> dict | None:
        """로딩 화면 — 10명의 {puuid, name, champ, pos, team: "a"|"e", me, games: 최근 20판, rank}.
        게임이 시작된 뒤(gameflow 세션에 팀이 생긴 뒤)에만 나온다. 다른 사람의 최근 전적과 랭크도 클라이언트에서 읽는다."""
        g = (self._get(GAME) or {}).get("gameData") or {}
        game_id = g.get("gameId")
        if not game_id:
            return None
        me = (self._get(ME) or {}).get("puuid")
        champ_of = {c.get("puuid"): c.get("championId") for c in g.get("playerChampionSelections") or [] if c.get("puuid")}
        people = []
        for team_key in ("teamOne", "teamTwo"):
            for m in g.get(team_key) or []:
                if not m.get("puuid"):
                    continue
                name = m.get("gameName") or m.get("riotIdGameName") or m.get("summonerName") or ""
                people.append({"puuid": m["puuid"], "name": name.split("#")[0], "side": team_key,
                               "champ": int(m.get("championId") or champ_of.get(m["puuid"]) or 0),
                               "pos": (m.get("selectedPosition") or "").upper(), "me": m["puuid"] == me})
        mine = next((p["side"] for p in people if p["me"]), None)
        if not mine:
            return None

        def fill(p):
            p["games"] = _games(self._get(HISTORY_OF.format(p["puuid"]), 10))  # 남의 전적은 클라이언트가 서버에서 받아 와 느리다
            q = ((self._get(RANKED_OF.format(p["puuid"]), 10) or {}).get("queueMap") or {}).get("RANKED_SOLO_5x5") or {}
            p["rank"] = {"tier": q.get("tier") or "", "div": q.get("division") or "", "lp": q.get("leaguePoints")} if q.get("tier") not in (None, "", "NONE") else None
            return p

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(5) as pool:
            people = list(pool.map(fill, people))
        for p in people:
            p["team"] = "a" if p.pop("side") == mine else "e"
        return {"gameId": int(game_id), "players": people}
