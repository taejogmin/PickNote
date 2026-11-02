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
