@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   [!] 먼저 0_Setup.cmd 를 실행하세요. / Run 0_Setup.cmd first.
    echo.
    pause
    exit /b 1
)
call "%~dp0tools\find_uv.cmd"
rem 새 버전을 설치하면 이 파일도 바뀐다. cmd는 실행 중인 파일을 줄마다 다시 읽으므로
rem 마지막 줄 하나에 전부 담아 둔다 — 한 줄은 미리 다 읽고 나서 실행된다.
".venv\Scripts\python.exe" -m src.cli update & if errorlevel 10 (if defined UV ("%UV%" sync)) & echo. & pause & exit /b
