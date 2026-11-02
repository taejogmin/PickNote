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
rem 승패 육각형 기준 데이터를 새로 받는다 (약 4시간). 들어 있는 models\benchmark.json 으로 충분하면 안 해도 된다.
rem 키 한도를 같이 쓰므로 수집기를 멈추고 받고, 끝나면 수집기를 다시 켠다.
echo.
echo   승패 육각형 기준 받기 - 약 4시간 / Downloading win/loss hexagon baselines - about 4 hours
echo.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*src.riot.collect run*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
".venv\Scripts\python.exe" -m src.riot.benchmark all
start "" "%~dp0..\2_Collect.cmd"
pause
