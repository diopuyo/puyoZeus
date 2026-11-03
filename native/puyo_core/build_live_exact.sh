#!/bin/sh
# C:上の作業木から単盤面DLLを再生成する。WSLの既設Rust/Zigを使用する。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
: "${LIVE_BUILD_ROOT:=<work>/proof}"
: "${LIVE_RUSTC:=<home>/.cargo/bin/rustc}"
: "${LIVE_ZIG_PYTHON:=<home>/zigvenv/bin/python}"
: "${LIVE_DLL_DEST:=$ROOT/src/phase_j/puyo_live_exact.dll}"
export LIVE_ZIG_PYTHON
export ZIG_GLOBAL_CACHE_DIR="$LIVE_BUILD_ROOT/zig-cache"
export TMPDIR="$LIVE_BUILD_ROOT"
mkdir -p "$LIVE_BUILD_ROOT/native"
"$LIVE_RUSTC" --edition=2021 --crate-type=cdylib --target x86_64-pc-windows-gnu \
  -C opt-level=3 -C panic=abort -C linker="$ROOT/native/puyo_core/live_zig_link.py" \
  "$ROOT/native/puyo_core/src/live_exact.rs" -o "$LIVE_BUILD_ROOT/native/puyo_live_exact.dll"
cp "$LIVE_BUILD_ROOT/native/puyo_live_exact.dll" "$LIVE_DLL_DEST"
