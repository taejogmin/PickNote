"""API 키가 아직 통하는지, 수집기가 돌고 있는지 본다.

    python -m tools.check_key

개발용 키는 24시간마다 만료된다. 수집기는 만료돼도 죽지 않고 2분마다 같은 키를 다시 시도하며
기다리므로(`src/riot/client.py`의 `_wait_for_new_key`), 까만 창을 안 보면 멈춘 줄 모른다.

세 가지를 갈라 본다:

1. **키 만료** — 요청이 401/403으로 돌아온다. 새 키를 받아 `.env`를 고쳐야 한다.
2. **수집기가 안 돈다** — 키는 통하는데 프로세스가 없거나 DB를 안 건드린다.
3. **정상**

**매치 수가 느는지로 살아있음을 재면 안 된다.** 수집기는 매치를 묶어서 커밋해서(배치 300)
한참 아무 일도 없다가 한 번에 들어온다 — 1분을 봐도 0건이 흔하다 (10-10에 실제로 오탐했다).
대신 **프로세스가 있는지**와 **WAL 파일을 최근에 건드렸는지**를 본다. 기다릴 필요도 없다.

키 값은 읽기만 하고 **어디에도 찍지 않는다.** 종류·길이도 적지 않는다.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

from src.i18n import L
from src.region import db_path, platform
from src.riot.client import read_env_file

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
WAL = db_path().with_name("matches.db-wal")
ENV = ROOT / ".env"
KEY_HOURS = 24          # 개발용 키 수명
STALE_MINUTES = 10      # 이보다 오래 DB를 안 건드렸으면 멈춘 것으로 본다


def collector_running() -> bool:
    """2_Collect.cmd의 중복 실행 막기와 같은 방식으로 본다."""
    query = (
        "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Where-Object { $_.CommandLine -like '*src.riot.collect run*' }) { exit 1 }"
    )
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                              capture_output=True, timeout=30)
        return done.returncode == 1
    except Exception:
        return False     # 못 세면 없는 것으로 치지 않고, 아래에서 WAL로 한 번 더 본다


def key_alive(key: str) -> tuple[bool, int]:
    """수집기가 실제로 쓰는 엔드포인트로 한 번만 물어본다. (통하나, 응답 코드)

    새 키는 엔드포인트마다 붙는 시각이 다르다 — 상태 엔드포인트가 먼저 통하기도 한다.
    그래서 상태가 아니라 **수집기가 쓰는 리그 목록**으로 묻는다.
    """
    url = f"https://{platform()}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/IV"
    try:
        r = requests.get(url, params={"page": 1}, headers={"X-Riot-Token": key}, timeout=15)
    except requests.RequestException as exc:
        print(L(f"네트워크 오류로 확인하지 못했습니다: {type(exc).__name__}", f"Couldn't check because of a network error: {type(exc).__name__}"))
        raise SystemExit(0)
    return r.status_code == 200, r.status_code


def main() -> None:
    key = read_env_file(ENV).get("RIOT_API_KEY", "")
    if not key:
        print(L("[!] .env에서 RIOT_API_KEY를 읽지 못했습니다. 1_ApiKey.cmd 로 넣어 주세요.", "[!] Couldn't read RIOT_API_KEY from .env. Add it with 1_ApiKey.cmd."))
        return

    age_h = (time.time() - ENV.stat().st_mtime) / 3600
    quiet_m = (time.time() - WAL.stat().st_mtime) / 60 if WAL.exists() else 1e9
    running = collector_running()
    alive, code = key_alive(key)

    print(L(f"키를 넣은 지 {age_h:.1f}시간 (개발용 키는 {KEY_HOURS}시간)", f"Key added {age_h:.1f}h ago (development keys last {KEY_HOURS}h)"))
    print(L(f"수집기 {'돌고 있음' if running else '꺼져 있음'} · DB를 건드린 지 {quiet_m:.1f}분",
            f"Collector {'running' if running else 'not running'} · DB last touched {quiet_m:.1f} min ago"))

    if not alive:
        print(L(f"\n[!] 키가 만료됐습니다 (응답 {code}).\n"
                "  1_ApiKey.cmd 로 새 키를 넣어 주세요.\n"
                "  수집기는 켜 둔 채로 두면 됩니다 — 2분 안에 새 키를 집어 이어서 받습니다.",
                f"\n[!] The key has expired (response {code}).\n"
                "  Put in a new key with 1_ApiKey.cmd.\n"
                "  Leave the collector running — it picks up the new key within 2 minutes."))
        return

    if not running:
        print(L("\n[!] 키는 통하는데 수집기가 안 돌고 있습니다.\n  2_Collect.cmd 를 실행하면 이어서 받습니다.",
                "\n[!] The key works but the collector isn't running.\n  Run 2_Collect.cmd to resume."))
        return

    if quiet_m > STALE_MINUTES:
        print(L(f"\n[!] 수집기는 떠 있는데 {quiet_m:.0f}분째 DB를 안 건드립니다.\n  수집기 창에 무슨 오류가 찍혀 있는지 봐 주세요.",
                f"\n[!] The collector is up but hasn't touched the DB for {quiet_m:.0f} min.\n  Check the collector window for errors."))
        return

    left = max(KEY_HOURS - age_h, 0)
    print(L(f"\n정상입니다. 키 만료까지 대략 {left:.1f}시간 남았습니다 (넣은 시각 기준 어림).",
            f"\nAll good. Roughly {left:.1f}h until the key expires (estimated from when it was added)."))


if __name__ == "__main__":
    main()
