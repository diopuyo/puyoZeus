@echo off
cd /d "%~dp0"
start "" "python\pythonw.exe" -m src.phase_j.distribution_app --root "%~dp0." %*
