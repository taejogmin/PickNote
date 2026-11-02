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
rem 새 경기가 적어도 지금 바로 다시 학습한다. 보통은 Update.cmd 가 알아서 한다.
".venv\Scripts\python.exe" -m src.model.train
echo.
pause
