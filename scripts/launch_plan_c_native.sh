#!/usr/bin/env bash
# 共有venvを変更せず、このworktree専用の拡張だけをビルドする。
set -eu
cd /mnt/d/puyo_analyzer/wt_prefire
out=/mnt/d/puyo_analyzer/wt_prefire/logs/prefire_prediction/plan_c
mkdir -p "$out"
export PATH="$HOME/.cargo/bin:$PATH"
export PYO3_PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1
cd native/puyo_core
set +e
nice -n 10 cargo build --release -j 1 > "$out/native_build.log" 2>&1
code=$?
if [ "$code" -eq 0 ]; then
    cp target/release/libpuyo_core.so ../../puyo_core.so
    code=$?
fi
printf '%s\n' "$code" > "$out/native_build.exit"
exit "$code"
