@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if exist "%~dp0unins000.exe" (
  start "" "%~dp0unins000.exe"
  exit /b
)
"%~dp0python\python.exe" -B -m src.phase_j.distribution_uninstall --root "%CD%" --interactive
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
