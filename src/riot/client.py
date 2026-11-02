"""Riot API 클라이언트.

개발용 키의 제한(초당 20회 / 2분당 100회)을 넘지 않도록 스스로 속도를 조절한다.
제한을 넘기면 키가 정지될 수 있어서, 실제 한도보다 조금 낮게 잡아 여유를 둔다.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path

import requests

from src.i18n import L
from src import region as rg

ROOT = Path(__file__).resolve().parents[2]


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """.env를 읽어서 돌려주기만 한다. python-dotenv를 의존성으로 더하지 않으려고 직접 읽는다.

    os.environ을 건드리지 않는 이유: 키가 만료됐을 때 파일을 다시 읽어
    새 키를 집어와야 하는데, setdefault로는 이미 들어간 옛 값을 못 덮는다.
    """
    path = path or ROOT / ".env"
    values: dict[str, str] = {}
    if path.exists():
        raw = path.read_bytes()
        # 메모장이 BOM을 붙이거나(UTF-8 BOM) 한글 주석을 ANSI(cp949)로 저장해도 읽는다.
        # 여기서 실패하면 새 키를 넣어도 수집기가 알아채지 못한다.
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp949", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def load_env(path: Path | None = None) -> dict[str, str]:
    values = read_env_file(path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


class RateLimitError(RuntimeError):
    pass


class RiotClient:
    # (요청 수, 초). 개발용 키 실제 한도는 20/1s, 100/120s.
    # 여러 프로세스가 같은 키를 쓰거나 시계가 어긋날 때를 대비해 낮춰 잡았다.
    DEFAULT_LIMITS = ((18, 1.0), (95, 122.0))

    def __init__(
        self,
        api_key: str | None = None,
        limits=None,
        verbose: bool = True,
        wait_for_new_key: bool = False,
        server: str | None = None,
    ):
        load_env()
        self.server = rg.pick(server)      # 리그 목록 · 매치 ID 목록을 어느 서버에 묻나
        self.api_key = api_key or os.environ.get("RIOT_API_KEY", "")
        # True면 키가 만료돼도 죽지 않고 .env에 새 키가 들어올 때까지 기다린다.
        # 며칠씩 돌리는 수집기용. 개발용 키는 24시간마다 만료된다.
        self.wait_for_new_key = wait_for_new_key
        if not self.api_key:
            raise SystemExit(
                L(".env에 RIOT_API_KEY가 없습니다. 1_ApiKey.cmd 로 키를 넣어 주세요.",
                  "RIOT_API_KEY is missing from .env. Add your key with 1_ApiKey.cmd.")
            )
        self.limits = limits or self.DEFAULT_LIMITS
        self.calls: deque[float] = deque()
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = self.api_key
        self.verbose = verbose
        self.request_count = 0
        self._waiting_key = False     # 키 만료 안내를 한 번만 찍으려고

    # ------------------------------------------------------------------ 속도 조절

    def _throttle(self) -> None:
        """가장 빡빡한 창을 기준으로, 다음 요청이 가능해질 때까지 기다린다."""
        while True:
            now = time.monotonic()
            longest = max(seconds for _, seconds in self.limits)
            while self.calls and now - self.calls[0] > longest:
                self.calls.popleft()

            wait = 0.0
            for count, seconds in self.limits:
                recent = [t for t in self.calls if now - t <= seconds]
                if len(recent) >= count:
                    # 이 창에서 가장 오래된 요청이 빠져나갈 때까지
                    wait = max(wait, seconds - (now - recent[0]) + 0.02)
            if wait <= 0:
                self.calls.append(now)
                return
            time.sleep(min(wait, 5.0))

    # ------------------------------------------------------------------ 요청

    def _reload_key(self) -> bool:
        """.env를 다시 읽어 키가 바뀌었으면 갈아 끼운다."""
        fresh = read_env_file().get("RIOT_API_KEY", "")
        if fresh and fresh != self.api_key:
            self.api_key = fresh
            self.session.headers["X-Riot-Token"] = fresh
            return True
        return False

    def _wait_for_new_key(self, code: int) -> None:
        """키가 만료됐을 때 죽지 않고 새 키를 기다린다.

        수집은 며칠에 걸쳐 도는데 개발용 키는 24시간마다 만료된다.
        여기서 죽으면 사람이 돌아올 때까지 통째로 노는 시간이 된다.

        새 키를 찾지 못해도 2분마다 돌아가 같은 키로 다시 시도한다.
        갓 발급한 키는 라이엇 쪽에 퍼지는 데 1분쯤 걸려서, 갈아 끼운 직후
        401을 한 번 더 받고 "다른 키"만 기다리며 멈춘 적이 있다 (2026-09-14).
        """
        if not self._waiting_key:
            self._waiting_key = True
            self._wait_started = time.monotonic()
            self._log(L(
                f"{code} 인증 실패 — 개발용 키가 만료된 것 같습니다. "
                "1_ApiKey.cmd 로 새 키를 넣어 주세요. "
                "고치면 여기서 알아서 이어받습니다 (다시 켤 필요 없음).",
                f"{code} auth failed — the development key has probably expired. "
                "Put in a new key with 1_ApiKey.cmd. "
                "Collection resumes here on its own (no restart needed).",
            ))
        for _ in range(6):
            time.sleep(20)
            if self._reload_key():
                self._log(L("새 키를 찾았습니다. 이어서 받습니다.", "Found a new key. Resuming."))
                return
        minutes = int((time.monotonic() - self._wait_started) // 60)
        if minutes and minutes % 10 < 2:
            self._log(L(f"{minutes}분째 새 키를 기다리는 중", f"Waiting for a new key ({minutes} min)"))

    def get(self, url: str, params: dict | None = None, retries: int = 4):
        """성공하면 JSON, 404면 None. 그 외 실패는 재시도 후 예외."""
        attempt = 0
        while attempt < retries:
            self._throttle()
            self.request_count += 1
            try:
                response = self.session.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                self._log(L(f"네트워크 오류 {exc} — {2 ** attempt}초 후 재시도", f"Network error {exc} — retrying in {2 ** attempt}s"))
                time.sleep(2**attempt)
                attempt += 1
                continue

            code = response.status_code
            if code == 200:
                if self._waiting_key:
                    self._waiting_key = False
                    self._log(L("키가 통합니다. 이어서 받습니다.", "Key works. Resuming."))
                return response.json()
            if code == 404:
                return None
            if code == 429:
                # 한도 초과는 실패가 아니라 "너무 빨랐다"는 신호다.
                # 재시도 횟수를 깎지 않고 기다렸다 그대로 다시 보낸다.
                delay = int(response.headers.get("Retry-After", "10")) + 1
                self._log(L(f"429 한도 초과 — {delay}초 대기 (요청 {self.request_count}건째)",
                             f"429 rate limited — waiting {delay}s (request #{self.request_count})"))
                time.sleep(delay)
                continue
            if code in (401, 403):
                if not self.wait_for_new_key:
                    raise SystemExit(L(
                        f"{code} 인증 실패. 개발용 키는 24시간마다 만료됩니다. 1_ApiKey.cmd 로 새 키를 넣어 주세요.",
                        f"{code} auth failed. Development keys expire every 24 hours. Put in a new key with 1_ApiKey.cmd.",
                    ))
                # 키가 갈릴 때까지 기다린다. 이것도 실패로 세지 않는다.
                self._wait_for_new_key(code)
                continue
            if 500 <= code < 600:
                self._log(L(f"{code} 서버 오류 — {2 ** attempt}초 후 재시도", f"{code} server error — retrying in {2 ** attempt}s"))
                time.sleep(2**attempt)
                attempt += 1
                continue
            raise RateLimitError(f"예상 못 한 응답 {code}: {response.text[:200]}")

        raise RateLimitError(f"{retries}번 시도했지만 실패: {url}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [api] {message}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ 엔드포인트

    def league_entries(self, tier: str, division: str, page: int, platform: str | None = None):
        """티어·디비전별 소환사 목록. 한 페이지에 최대 205명."""
        url = (
            f"https://{platform or rg.platform(self.server)}.api.riotgames.com"
            f"/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
        )
        return self.get(url, {"page": page}) or []

    def apex_league(self, tier: str, platform: str | None = None):
        """마스터·그랜드마스터·챌린저는 디비전이 없어 경로가 따로다. 한 번에 전원이 온다."""
        path = {"MASTER": "masterleagues", "GRANDMASTER": "grandmasterleagues",
                "CHALLENGER": "challengerleagues"}[tier]
        url = f"https://{platform or rg.platform(self.server)}.api.riotgames.com/lol/league/v4/{path}/by-queue/RANKED_SOLO_5x5"
        return (self.get(url) or {}).get("entries", [])

    def match_ids(
        self,
        puuid: str,
        *,
        queue: int = 420,
        count: int = 100,
        start_time: int | None = None,
        start: int = 0,
        region: str | None = None,
    ):
        """최근 매치 ID. start_time은 초 단위 epoch, start는 몇 번째부터 (거슬러 받을 때 넘긴다)."""
        url = f"https://{region or rg.region(self.server)}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params: dict = {"queue": queue, "type": "ranked", "start": start, "count": count}
        if start_time:
            params["startTime"] = start_time
        return self.get(url, params) or []

    def account_by_riot_id(self, game_name: str, tag_line: str, region: str | None = None):
        """Riot ID(이름#태그) → 계정 (puuid 포함). 내 경기 기록을 받을 때 쓴다."""
        from urllib.parse import quote
        url = (f"https://{region or rg.account_region(self.server)}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"
               f"{quote(game_name)}/{quote(tag_line)}")
        return self.get(url)

    def match(self, match_id: str, region: str | None = None):
        # 매치 ID 앞부분이 서버다 — 다른 서버 계정의 경기도 맞는 곳에 묻는다
        url = f"https://{region or rg.region_of_match(match_id)}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self.get(url)
