@echo off
if not defined PUYO_DEV_PYTHON set "PUYO_DEV_PYTHON=%~dp0..\.local\build_20261019a_stage\PuyoLive\python\python.exe"
setlocal
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
"%PUYO_DEV_PYTHON%" -B "%~dp0live_gate_coef.py" %*
exit /b %errorlevel%
