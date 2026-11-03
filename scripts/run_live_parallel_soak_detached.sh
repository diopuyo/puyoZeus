#!/usr/bin/env bash
# Windows実機の試験をWSL側setsid配下から起動し、セッション終了後も待機を継続する。
set -eu
cd <work>/wt_live_par
/mnt/c/PuyoLive_20261012a/PuyoLive/python/python.exe -X utf8 -B <work>/wt_live_par/scripts/run_live_parallel_soak.py > logs/live_parallel/camera_wait.log 2>&1
