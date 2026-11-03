#!/bin/bash
# WSL cargo/Zigから独立検証先へ出力する。本番配布ディレクトリは変更しない。
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_TARGET_DIR="${AMA_TARGET_DIR:-<home>/ama_rust_windows}"
export PYO3_CONFIG_FILE="${AMA_PYO3_CONFIG:-<home>/b3_rt_windows_target/maturin/pyo3-config-x86_64-pc-windows-gnu-3.12.txt}"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$PWD/ama_zig_link.py"
# 既設配布ビルドで生成済みのCPython 3.12 import libraryを使う。
export RUSTFLAGS="-L ${AMA_PYTHON_LIB:-<home>/b3_rt_windows_target/x86_64-pc-windows-gnu/release/build/pyo3-ffi-ee1ae34af4dda785/out/lib}"
<home>/.cargo/bin/cargo build --release --target x86_64-pc-windows-gnu -j 4
mkdir -p target/ama
cp "$CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/puyo_core.dll" target/ama/puyo_core.pyd
