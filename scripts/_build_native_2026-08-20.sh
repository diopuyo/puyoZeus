#!/bin/bash
# native/puyo_core を maturin でビルドして venv に入れる (2026-08-20)。
# cargo は ~/.cargo/bin にあり PATH に無いので明示的に足す。
# PATH に空白/カッコを含むため wsl 直書きは MSYS に壊される → スクリプト化。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/native/puyo_core || exit 1
export PATH="$HOME/.cargo/bin:$PATH"
# maturin develop は VIRTUAL_ENV を要求する (activate 相当を手で与える)
export VIRTUAL_ENV=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "=== build start $(date +%T) ==="
"$VIRTUAL_ENV/bin/maturin" develop --release 2>&1 | tail -30
rc=$?
echo "=== build end $(date +%T) rc=$rc ==="
