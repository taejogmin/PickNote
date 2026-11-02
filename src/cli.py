"""더블클릭용 .cmd 파일이 부르는 입구. 안내 문구는 0_Setup.cmd에서 고른 언어로 나온다.

    python -m src.cli setup        처음 한 번 — 언어 · 서버 · .env · 챔피언 목록
    python -m src.cli settings     언어 · 서버 바꾸기
    python -m src.cli apikey       Riot 개발자 사이트와 .env 메모장을 연다
    python -m src.cli collect      수집기 (이미 돌고 있으면 또 켜지 않는다)
    python -m src.cli open         화면을 연다
    python -m src.cli myrecord     my_accounts.txt의 계정들 경기 기록 받기
    python -m src.cli update       새 버전 확인 · 다시 학습 · 내 기록 갱신
    python -m src.cli status       수집 진행 상황 · 키 상태
    python -m src.cli autostart on|off   로그인할 때 수집기 자동 시작
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests

from src import region as rg
from src.i18n import L, set_default_lang

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
ACCOUNTS = ROOT / "my_accounts.txt"
DB = rg.db_path()                 # 고른 서버의 수집 DB (한국은 data/matches.db)
PORT = 8124
REPO = "taejogmin/PickNote"
API = f"https://api.github.com/repos/{REPO}"
# 수집 데이터는 공개하지 않는다 (Riot API 약관 — 받은 정보를 남에게 넘길 수 없다).
# 그래서 처음 학습은 각자 모은 경기가 이만큼 쌓인 뒤에 한다. 그 전에는 들어 있는 모델을 쓴다.
FIRST_TRAIN_GAMES = 300_000
RETRAIN_NEW_GAMES = 30_000                    # 그 뒤로는 지난 학습 뒤 이만큼 쌓였거나
RETRAIN_NEW_PATCH = 10_000                    # 새 패치 경기가 이만큼 보이면 다시 학습
TRAIN_STATE = rg.data_dir() / "last_train.json"   # 이 PC에서 마지막으로 학습을 돌린 시점 (채택 여부와 무관)
LOW_TIERS = "('IRON','BRONZE','SILVER','GOLD','PLATINUM')"
# 업데이트할 때 건드리지 않는 것 — 사용자가 만든 것들
KEEP = {".env", "my_accounts.txt", "data", "models", ".venv", ".git"}

ACCOUNTS_TEMPLATE = """\
# 내 Riot ID를 한 줄에 하나씩 적고, 저장(Ctrl+S)한 뒤 메모장을 닫으세요.
# Write one Riot ID per line, save (Ctrl+S), then close Notepad.
#
# 예 / Example:
#   이름#KR1
#
# 계정이 설정한 서버와 다른 서버에 있으면 뒤에 @서버를 붙이세요.
# If an account is on a different server than your setting, add @server after it.
#   Name#EUW @euw1
# 서버 / Servers: kr jp1 na1 br1 la1 la2 euw1 eun1 tr1 ru me1 oc1 sg2 tw2 vn2
#
# '#'으로 시작하는 줄은 무시합니다. / Lines starting with '#' are ignored.

"""


def ask(question: str) -> bool:
    try:
        answer = input(f"{question} (Y/N) ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "ㅛ", "예", "네")


def run_module(*args: str) -> int:
    """같은 파이썬으로 모듈을 돌린다. 출력은 그대로 이 창에 나온다."""
    return subprocess.call([sys.executable, "-m", *args], cwd=ROOT)


def version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    return "0"


def parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.lstrip("v").split(".") if x.isdigit())


# ---------------------------------------------------------------- 수집기 프로세스

COLLECTOR_QUERY = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                   "| Where-Object { $_.CommandLine -like '*src.riot.collect run*' }")


def collector_running() -> bool:
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-Command", f"if ({COLLECTOR_QUERY}) {{ exit 1 }}"],
                              capture_output=True, timeout=30)
        return done.returncode == 1
    except Exception:
        return False


def stop_collector() -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"{COLLECTOR_QUERY} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"],
                   capture_output=True, timeout=60)


def start_collector_window() -> None:
    """수집기를 새 창에서 다시 켠다."""
    subprocess.Popen(["cmd", "/c", "start", "", str(ROOT / "2_Collect.cmd")], cwd=ROOT)


# ---------------------------------------------------------------- 명령

def choose_lang() -> None:
    print("\n  1. 한국어\n  2. English\n")
    try:
        choice = input("  언어를 고르세요 / Choose a language (1/2): ").strip()
    except EOFError:
        choice = "1"
    set_default_lang("en" if choice == "2" else "ko")


def choose_server() -> None:
    """어느 서버의 경기를 모으고 학습할지. 엔터만 치면 지금 값(처음엔 한국)."""
    codes = list(rg.SERVERS)
    now = rg.server()
    print(L("\n  어느 서버에서 하나요? 수집 · 학습 · 내 기록이 이 서버 기준입니다.",
            "\n  Which server do you play on? Collecting, training and your record use it."))
    for i, code in enumerate(codes, 1):
        mark = " *" if code == now else ""
        print(f"  {i:>2}. {rg.name(code)} ({code}){mark}")
    try:
        raw = input(L(f"  번호 (엔터 = {rg.name(now)}): ", f"  Number (Enter = {rg.name(now)}): ")).strip()
    except EOFError:
        raw = ""
    if raw.isdigit() and 1 <= int(raw) <= len(codes):
        rg.set_server(codes[int(raw) - 1])
    elif not rg.SERVER_FILE.exists():
        rg.set_server(now)
    s = rg.server()
    print(L(f"  서버: {rg.name(s)} ({s})", f"  Server: {rg.name(s)} ({s})"))
    if s != rg.DEFAULT:
        print(L("  들어 있는 모델은 한국 데이터입니다. 이 서버 경기를 30만 판 모아 Update.cmd를 돌리면 이 서버 모델로 바뀝니다.",
                "  The bundled model is from Korean games. Collect 300k games on this server and run Update.cmd to get a model for it."))


def cmd_settings(_args) -> None:
    choose_lang()                      # L()은 부를 때마다 파일을 읽으므로 바로 고른 언어로 이어 묻는다
    choose_server()
    print(L("\n  바꿨습니다. 켜져 있는 창(수집기 · 화면)은 닫고 다시 켜야 적용됩니다.",
            "\n  Saved. Close and reopen any running windows (collector, app) to apply."))


def cmd_setup(_args) -> None:
    choose_lang()
    choose_server()

    if not ENV.exists():
        shutil.copyfile(ROOT / ".env.example", ENV)
    if not ACCOUNTS.exists():
        ACCOUNTS.write_text(ACCOUNTS_TEMPLATE, encoding="utf-8")

    print(L("\n챔피언 목록을 최신으로 받습니다…", "\nUpdating the champion list…"))
    if run_module("src.data.champions") != 0:
        print(L("  받지 못했습니다. 들어 있는 목록을 그대로 씁니다.", "  Couldn't download it. Using the bundled list."))

    print(L("\n준비가 끝났습니다. 다음은 1_ApiKey.cmd (수집·내 기록에 필요) 또는 3_Open.cmd (화면) 입니다.",
            "\nAll set. Next: 1_ApiKey.cmd (needed for collecting and your record) or 3_Open.cmd (the app)."))


def cmd_apikey(_args) -> None:
    if not ENV.exists():
        shutil.copyfile(ROOT / ".env.example", ENV)
    print(L("""
  [1] Riot 개발용 API 키 넣기
  ------------------------------------------------------------
  개발용 키는 발급 24시간 뒤 만료됩니다. 수집할 때는 매일 새로 넣어야 합니다.

  1. 열리는 사이트에서 Riot 계정으로 로그인 -> REGENERATE API KEY
  2. RGAPI- 로 시작하는 키를 복사
  3. 같이 열리는 메모장에서 RIOT_API_KEY= 뒤를 새 키로 바꾸기
  4. 저장하고 닫기 (Ctrl+S)

  수집기가 돌고 있으면 끄지 않아도 됩니다. 저장하면 20초 안에 새 키로 이어받습니다.
""", """
  [1] Add your Riot development API key
  ------------------------------------------------------------
  Development keys expire 24 hours after they're issued. Renew it daily while collecting.

  1. Sign in with your Riot account on the page that opens -> REGENERATE API KEY
  2. Copy the key that starts with RGAPI-
  3. In the Notepad window that opens, replace the text after RIOT_API_KEY= with it
  4. Save and close (Ctrl+S)

  No need to stop the collector. It picks up the new key within 20 seconds of saving.
"""))
    webbrowser.open("https://developer.riotgames.com/")
    subprocess.Popen(["notepad", str(ENV)])


def has_key() -> bool:
    from src.riot.client import read_env_file

    key = read_env_file(ENV).get("RIOT_API_KEY", "")
    return key.startswith("RGAPI-") and "paste" not in key


def cmd_collect(args) -> None:
    if not has_key():
        sys.exit(L("\n  [!] API 키가 없습니다. 먼저 1_ApiKey.cmd 를 실행하세요.",
                   "\n  [!] No API key yet. Run 1_ApiKey.cmd first."))
    if collector_running():
        print(L("\n  수집기가 이미 돌고 있습니다. 이 창은 닫으셔도 됩니다.",
                "\n  The collector is already running. You can close this window."))
        return
    apex = args.tiers == "apex"
    print(L(f"""
  [2] {rg.name()} 솔로랭크 수집기{' (마스터 이상 · 실험용)' if apex else ''}
  ------------------------------------------------------------
  최근 30일 경기를 받습니다. 개발용 키로 하루 약 5만 경기입니다.
  켜 두고 잊으셔도 됩니다.

  * 멈추려면 Ctrl+C (다시 켜면 이어서 받습니다)
  * 키가 만료돼도 죽지 않습니다. 1_ApiKey.cmd 로 새 키만 넣으면 이어받습니다.
""", f"""
  [2] {rg.name()} Solo/Duo collector{' (Master+ · experimental)' if apex else ''}
  ------------------------------------------------------------
  Collects games from the last 30 days — about 50,000 a day with a development key.
  Leave it running and forget about it.

  * Ctrl+C to stop (starting it again resumes)
  * An expired key doesn't stop it. Put in a new key with 1_ApiKey.cmd and it carries on.
"""))
    run_module("src.riot.collect", "run", "--tiers", args.tiers)


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def cmd_open(_args) -> None:
    url = f"http://127.0.0.1:{PORT}/"
    if port_open():
        print(L("  이미 켜져 있어서 브라우저만 엽니다.", "  Already running — just opening the browser."))
        webbrowser.open(url)
        return
    print(L(f"""
  [3] PickNote 화면
  ------------------------------------------------------------
  주소: {url}
  끄려면 이 창에서 Ctrl+C
""", f"""
  [3] PickNote
  ------------------------------------------------------------
  Address: {url}
  Press Ctrl+C in this window to stop
"""))
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.web.app:app", "--port", str(PORT)], cwd=ROOT)
    for _ in range(60):
        if port_open():
            webbrowser.open(url)
            break
        if server.poll() is not None:
            break
        time.sleep(0.5)
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()


def read_accounts() -> list[tuple[str, str]]:
    """my_accounts.txt → [(Riot ID, 서버)]. 서버는 줄 끝의 @서버, 없으면 설정한 서버."""
    if not ACCOUNTS.exists():
        return []
    raw = ACCOUNTS.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949", errors="replace")      # 메모장이 ANSI로 저장한 경우
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "#" not in line:
            continue
        server = rg.server()
        if " @" in line:
            line, _, tail = line.rpartition(" @")
            server = rg.pick(tail)
        ids.append((line.strip(), server))
    return list(dict.fromkeys(ids))


def cmd_myrecord(_args) -> None:
    from src.riot import me

    if not has_key():
        sys.exit(L("\n  [!] API 키가 없습니다. 먼저 1_ApiKey.cmd 를 실행하세요.",
                   "\n  [!] No API key yet. Run 1_ApiKey.cmd first."))
    if not ACCOUNTS.exists():
        ACCOUNTS.write_text(ACCOUNTS_TEMPLATE, encoding="utf-8")
    ids = read_accounts()
    if not ids:
        print(L("\n  메모장에 내 Riot ID(이름#태그)를 적고 저장한 뒤 닫아 주세요…",
                "\n  Write your Riot ID (Name#Tag) in Notepad, save, and close it…"))
        subprocess.call(["notepad", str(ACCOUNTS)])
        ids = read_accounts()
    if not ids:
        sys.exit(L("  Riot ID가 없어 멈춥니다. 다시 실행해 주세요.", "  No Riot ID found — stopping. Run this again."))

    conn = me.connect()
    known = {riot_id.lower() for _p, riot_id in me.accounts(conn)}
    new = [(i, s) for i, s in ids if i.lower() not in known]

    if new:
        names = ", ".join(f"{i} ({s})" for i, s in new)
        print(L(f"\n  처음 받는 계정: {names}\n  지난 경기를 거슬러 받습니다 (수천 경기면 1~2시간).",
                f"\n  New accounts: {names}\n  Downloading past games (an hour or two for a few thousand)."))
        restart = False
        if collector_running():
            # 키 한도를 같이 쓰면 둘 다 막힌다 — 잠시 멈췄다가 끝나면 다시 켠다
            if not ask(L("  수집기를 잠시 멈추고 받을까요? 끝나면 다시 켭니다.",
                         "  Pause the collector while downloading? It restarts afterwards.")):
                sys.exit(L("  나중에 다시 실행해 주세요.", "  Run this again later."))
            stop_collector()
            restart = True
        try:
            for riot_id, server in new:
                try:
                    me.cmd_fetch(argparse.Namespace(riot_id=riot_id, server=server, force=True))
                except SystemExit as exc:          # 한 계정이 틀려도 나머지는 받는다
                    print(f"  [!] {exc}")
        finally:
            if restart:
                start_collector_window()

    print(L("\n  등록한 계정의 새 경기를 받습니다…", "\n  Fetching new games for your accounts…"))
    try:
        me.cmd_update(argparse.Namespace())
    except SystemExit as exc:
        print(f"  [!] {exc}")
    me.cmd_stats(argparse.Namespace())
    print(L("\n  화면의 '내 기록'을 새로고침하면 반영됩니다.", "\n  Refresh 'My record' in the app to see it."))


# ---------------------------------------------------------------- 업데이트

def github(path: str):
    r = requests.get(f"{API}{path}", timeout=15, headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    return r.json()


def latest_release() -> dict | None:
    """가장 새 정식 릴리스 (v로 시작하는 태그)."""
    for rel in github("/releases?per_page=20"):
        if not rel.get("draft") and not rel.get("prerelease") and rel["tag_name"].startswith("v"):
            return rel
    return None


def apply_release(rel: dict) -> None:
    """새 버전의 코드만 덮어쓴다. 키 · 계정 목록 · 수집 데이터 · 학습한 모델은 그대로 둔다."""
    r = requests.get(rel["zipball_url"], timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        top = next(Path(tmp).iterdir())          # GitHub zip은 "owner-repo-커밋" 폴더 하나로 싸여 있다
        for src in top.rglob("*"):
            rel_path = src.relative_to(top)
            if rel_path.parts[0] in KEEP or src.is_dir():
                continue
            dest = ROOT / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)


def model_meta() -> dict:
    try:
        return json.loads((rg.usable_model_dir() / "draft.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def collected() -> tuple[int, int]:
    """(아이언~플래 경기 수, 마지막 경기 시각 ms). 수집 DB가 없거나 비었으면 (0, 0)."""
    if not DB.exists():
        return 0, 0
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{quote(DB.as_posix())}?mode=ro", uri=True)
        n, last = conn.execute(f"SELECT COUNT(*), MAX(created_ms) FROM match WHERE tier IN {LOW_TIERS}").fetchone()
        return n, last or 0
    except sqlite3.Error:
        return 0, 0


def needs_retrain() -> tuple[bool, str]:
    total, _last = collected()
    try:
        state = json.loads(TRAIN_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None
    if state is None:                      # 아직 이 PC에서 학습한 적이 없다 — 들어 있는 모델을 쓰는 중
        if total >= FIRST_TRAIN_GAMES:
            return True, L(f"모은 경기 {total:,} — 처음으로 직접 학습합니다",
                           f"{total:,} games collected — training on them for the first time")
        return False, L(f"모은 경기 {total:,} / {FIRST_TRAIN_GAMES:,} — 그때까지는 들어 있는 모델을 씁니다",
                        f"{total:,} / {FIRST_TRAIN_GAMES:,} games collected — using the bundled model until then")

    import sqlite3

    conn = sqlite3.connect(f"file:{quote(DB.as_posix())}?mode=ro", uri=True)
    new, = conn.execute(f"SELECT COUNT(*) FROM match WHERE tier IN {LOW_TIERS} AND created_ms > ?",
                        (state["until_ms"],)).fetchone()
    patch = conn.execute("SELECT patch FROM match ORDER BY created_ms DESC LIMIT 1").fetchone()
    new_patch = patch and patch[0] not in state.get("patches", [])
    if new_patch and new >= RETRAIN_NEW_PATCH:
        return True, L(f"새 패치 {patch[0]} · 지난 학습 뒤 새 경기 {new:,}", f"new patch {patch[0]} · {new:,} games since last training")
    if new >= RETRAIN_NEW_GAMES:
        return True, L(f"지난 학습 뒤 새 경기 {new:,}", f"{new:,} games since last training")
    return False, L(f"지난 학습 뒤 새 경기 {new:,} — {RETRAIN_NEW_GAMES:,}경기가 쌓이면 다시 학습합니다",
                    f"{new:,} games since last training — retrains once {RETRAIN_NEW_GAMES:,} have built up")


def train() -> None:
    """다시 학습하고, 채택 여부와 상관없이 이번에 본 데이터까지를 적어 둔다 — 같은 데이터로 또 돌지 않게."""
    _total, until = collected()
    import sqlite3

    conn = sqlite3.connect(f"file:{quote(DB.as_posix())}?mode=ro", uri=True)
    patches = [r[0] for r in conn.execute("SELECT DISTINCT patch FROM match")]
    conn.close()
    if run_module("src.model.train") != 0:
        print(L("  학습이 실패했습니다. 지금 모델을 그대로 씁니다.", "  Training failed. Keeping the current model."))
        return
    TRAIN_STATE.write_text(json.dumps({"until_ms": until, "patches": patches}), encoding="utf-8")


def cmd_update(args) -> None:
    print(L("\n  [업데이트] 1/3 새 버전 확인", "\n  [Update] 1/3 Checking for a new version"))
    try:
        rel = latest_release()
    except requests.RequestException:
        rel = None
        print(L("  확인하지 못했습니다 (인터넷 연결?). 건너뜁니다.", "  Couldn't check (no internet?). Skipping."))
    if rel and parse_version(rel["tag_name"]) > parse_version(version()):
        print(L(f"  새 버전 {rel['tag_name']} 이 있습니다 (지금 v{version()}).",
                f"  A new version {rel['tag_name']} is available (you have v{version()})."))
        if rel.get("body"):
            print("  " + rel["body"].strip().replace("\n", "\n  ")[:1500])
        if ask(L("  설치할까요? 키 · 내 계정 · 수집 데이터 · 모델은 그대로 둡니다.",
                 "  Install it? Your key, accounts, collected data and models are kept.")):
            apply_release(rel)
            print(L("\n  설치했습니다. 이 창이 패키지를 맞춘 뒤, Update.cmd 를 한 번 더 실행해 나머지를 이어 하세요.",
                    "\n  Installed. This window will sync packages; then run Update.cmd once more to finish."))
            sys.exit(10)                     # Update.cmd가 패키지를 다시 맞춘다
    else:
        print(L(f"  최신입니다 (v{version()}).", f"  Up to date (v{version()})."))

    run_module("src.data.champions")         # 새 챔피언 — 키가 필요 없다

    print(L("\n  [업데이트] 2/3 다시 학습", "\n  [Update] 2/3 Retraining"))
    retrain, why = needs_retrain()
    print(f"  {why}")
    if retrain or args.force_train:
        train()
        if port_open():
            print(L("  화면이 켜져 있으면 3_Open.cmd 창을 닫고 다시 켜야 새 모델을 씁니다.",
                    "  If the app is open, close the 3_Open.cmd window and start it again to load the new model."))

    print(L("\n  [업데이트] 3/3 내 기록", "\n  [Update] 3/3 My record"))
    if read_accounts() and has_key():
        cmd_myrecord(args)
    else:
        print(L("  등록한 계정이나 API 키가 없어 건너뜁니다 (4_MyRecord.cmd).",
                "  No accounts or API key yet — skipping (see 4_MyRecord.cmd)."))


def cmd_status(_args) -> None:
    s = rg.server()
    print(L(f"서버: {rg.name(s)} ({s})\n", f"Server: {rg.name(s)} ({s})\n"))
    run_module("src.riot.collect", "stats")
    meta = model_meta()
    if meta:
        print(L(f"\n모델: {meta.get('matches', 0):,}경기 · {meta.get('trained_at', '?')} 학습",
                f"\nModel: {meta.get('matches', 0):,} games · trained {meta.get('trained_at', '?')}"))
    print()
    run_module("tools.check_key")


def startup_link() -> Path:
    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "PickNote collector.lnk"


def cmd_autostart(args) -> None:
    link = startup_link()
    if args.state == "off":
        link.unlink(missing_ok=True)
        print(L("  로그인할 때 수집기를 켜지 않습니다.", "  The collector will no longer start at sign-in."))
        return
    target = ROOT / "2_Collect.cmd"
    script = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{link}');"
              f"$s.TargetPath='{target}';$s.WorkingDirectory='{ROOT}';$s.WindowStyle=7;$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    print(L(f"  로그인하면 수집기가 최소화된 창으로 켜집니다.\n  끄려면 advanced\\Autostart_Off.cmd",
            f"  The collector will start minimized when you sign in.\n  To undo: advanced\\Autostart_Off.cmd"))


def main() -> None:
    parser = argparse.ArgumentParser(description="PickNote")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(func=cmd_setup)
    sub.add_parser("settings").set_defaults(func=cmd_settings)
    sub.add_parser("apikey").set_defaults(func=cmd_apikey)
    c = sub.add_parser("collect")
    c.add_argument("--tiers", default="low")
    c.set_defaults(func=cmd_collect)
    sub.add_parser("open").set_defaults(func=cmd_open)
    sub.add_parser("myrecord").set_defaults(func=cmd_myrecord)
    u = sub.add_parser("update")
    u.add_argument("--force-train", action="store_true")
    u.set_defaults(func=cmd_update)
    sub.add_parser("status").set_defaults(func=cmd_status)
    a = sub.add_parser("autostart")
    a.add_argument("state", choices=["on", "off"])
    a.set_defaults(func=cmd_autostart)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
