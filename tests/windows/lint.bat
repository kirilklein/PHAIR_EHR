@echo off
REM run flake8
flake8 corebehrt tests --count --select=E9,F63,F7,F82,U100,E711,E712,E713,E714,E721,F401,F402,F405,F811,F821,F822,F823,F831,F841,F901, --show-source --statistics
REM Pause to allow you to see the output before the window closes
pause