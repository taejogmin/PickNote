@echo off
chcp 65001 > nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   [!] 먼저 0_Setup.cmd 를 실행하세요. / Run 0_Setup.cmd first.
    echo.
    pause
    exit /b 1
)
rem 마스터 이상 경기 — 실험용. 2_Collect.cmd 와 같이 돌지 않는다 (키 한도를 나눠 쓴다).
".venv\Scripts\python.exe" -m src.cli collect --tiers apex
echo.
pause
