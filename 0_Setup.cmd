@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem zip 안에서 바로 실행하면 이 파일 하나만 임시 폴더에 풀려서 옆 파일이 없다.
if not exist "%~dp0pyproject.toml" (
    echo.
    echo   [!] 압축을 먼저 풀어 주세요. zip 파일 오른쪽 클릭 -^> 압축 풀기 뒤 그 폴더에서 실행하세요.
    echo   [!] Extract the zip first: right-click it -^> Extract All, then run this from that folder.
    echo.
    pause
    exit /b 1
)

echo.
echo   PickNote 설치 / Setup
echo   ------------------------------------------------------------
echo   파이썬과 필요한 패키지를 이 폴더 안에 설치합니다. 처음 한 번만, 몇 분 걸립니다.
echo   Installs Python and the required packages inside this folder. Once only, takes a few minutes.
echo.

call "%~dp0tools\find_uv.cmd"
if defined UV goto :have_uv
echo   uv 설치 중 (파이썬 설치 도구, astral.sh) / Installing uv (Python installer, astral.sh)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
call "%~dp0tools\find_uv.cmd"
if not defined UV (
    echo.
    echo   [!] uv 를 설치하지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.
    echo   [!] Could not install uv. Check your internet connection and run this again.
    pause
    exit /b 1
)

:have_uv
"%UV%" sync
if errorlevel 1 (
    echo.
    echo   [!] 패키지 설치에 실패했습니다. 위 메시지를 확인해 주세요.
    echo   [!] Package install failed. See the messages above.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m src.cli setup
echo.
pause
