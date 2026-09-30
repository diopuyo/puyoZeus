@echo off
rem ぷよぷよ有利不利オーバーレイ 起動用 (ダブルクリック可)
chcp 65001 >nul
cd /d "%~dp0"
rem バイトコードは書かせる (初回のコンパイルを 2 回目以降に持ち越さない。フォルダは書き込み可の前提)
set PYTHONUTF8=1
"python\python.exe" -m src.phase_j.launcher --config "%~dp0puyo_live.json" --require-manifest %*
if errorlevel 1 (
  echo.
  echo 起動に失敗しました。上のメッセージを確認してください。
  pause
)
