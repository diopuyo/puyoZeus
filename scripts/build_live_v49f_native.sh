#!/usr/bin/env bash
# 既存wheel・targetを触らず、Windows用の候補を新しい出力先へ作る。
set -euo pipefail
cd <work>/wt_live_v49f
out=/mnt/d/puyo_analyzer/packaging/build_20261019a/native
test ! -e "$out" || exit 2
mkdir -p "$out"
export CARGO_TARGET_DIR=<home>/live_v49f_windows_20261019
export PYO3_CONFIG_FILE=<home>/b3_rt_windows_target/maturin/pyo3-config-x86_64-pc-windows-gnu-3.12.txt
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$PWD/native/puyo_core/ama_zig_link.py"
export RUSTFLAGS='-C target-cpu=x86-64 -L <home>/b3_rt_windows_target/x86_64-pc-windows-gnu/release/build/pyo3-ffi-ee1ae34af4dda785/out/lib'
features=ama-fast-avx2,ama-fast-connected,ama-fast-gravity,ama-reuse-identical-queues,b3-fast-prefilter,b3-fast-chain,compact-chain-summary
<home>/.cargo/bin/cargo build --locked --release --target x86_64-pc-windows-gnu -j 1 --manifest-path native/puyo_core/Cargo.toml --features "$features" > "$out/build.log" 2>&1
cp "$CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/puyo_core.dll" "$out/puyo_core.pyd"
printf '%s\n' "$features" > "$out/features.txt"
sha256sum "$out/puyo_core.pyd" > "$out/SHA256SUMS.txt"
