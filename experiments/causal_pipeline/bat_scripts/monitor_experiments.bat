@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Monitor Experiment Progress
REM ========================================

echo ========================================
echo Experiment Progress Monitor
echo ========================================
echo.

:monitor_loop
cls
echo ========================================
echo Experiment Progress Monitor
echo ========================================
echo Current Time: !time! !date!
echo.

REM Check for running Python processes
set PYTHON_RUNNING=0
for /f %%i in ('tasklist /fi "imagename eq python.exe" ^| find /c "python.exe"') do set PYTHON_RUNNING=%%i

if %PYTHON_RUNNING% gtr 0 (
    echo Status: EXPERIMENTS RUNNING ^(%PYTHON_RUNNING% Python processes^)
) else (
    echo Status: NO EXPERIMENTS RUNNING
)
echo.

REM Show recent experiment outputs
echo Recent Experiment Directories:
if exist "..\..\outputs\causal\sim_study\runs\" (
    for /f "tokens=*" %%d in ('dir "..\..\outputs\causal\sim_study\runs\" /b /ad /o-d 2^>nul') do (
        echo   %%d
    )
) else (
    echo   No experiment outputs found yet
)
echo.

REM Show recent log files
echo Recent Log Files:
if exist "run_*.log" (
    for /f "tokens=*" %%f in ('dir "run_*.log" /b /o-d 2^>nul') do (
        echo   %%f
    )
) else (
    echo   No log files found
)
echo.

echo Press Ctrl+C to exit, or wait for auto-refresh...
timeout /t 10 /nobreak >nul
goto :monitor_loop
