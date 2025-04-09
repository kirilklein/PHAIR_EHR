@echo off
REM Run unittests
python -m unittest discover -s tests
REM Pause to allow you to see the output before the window closes
pause