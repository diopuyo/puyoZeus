#!/usr/bin/env bash
# 配信用 puyo_core (Windows 向け .pyd) を、WSL の Rust + Zig リンカで作る。
# 既存の配布 pyd (build_20261019a) と同じ機能旗・同じ CPU 前提 (x86-64 基準、AVX 等は使わない) で、
# 出力先は常に新しいフォルダ (既存の wheel・target・配布物には触れない)。
#
# 使い方 (PC1 の Git Bash から。MSYS のパス変換を止める):
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash <work>/wt_live_recog_speed_rust/scripts/build_live_recog_native.sh \
#       <work>/recog_speed_rust_20261025/native_build_N
#
# 前例: scripts/build_live_v49f_native.sh (2026-10-19)。違いは、作業木を自分の位置から決める点と、
# 出力先・target を引数/環境変数で変えられる点だけ。
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out=${1:?出力先フォルダ (存在しないこと) を指定}
test ! -e "$out" || { echo "出力先が既にあります: $out" >&2; exit 2; }
mkdir -p "$out"
cd "$ROOT"
export CARGO_TARGET_DIR=${LIVE_RECOG_TARGET_DIR:-<home>/live_recog_rust_windows_20261025}
export PYO3_CONFIG_FILE=<home>/b3_rt_windows_target/maturin/pyo3-config-x86_64-pc-windows-gnu-3.12.txt
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$ROOT/native/puyo_core/ama_zig_link.py"
# target-cpu=x86-64: 低スペックの PC でも動くよう SSE2 までの基準命令だけ (既存の配布ビルドと同じ)。
export RUSTFLAGS='-C target-cpu=x86-64 -L <home>/b3_rt_windows_target/x86_64-pc-windows-gnu/release/build/pyo3-ffi-ee1ae34af4dda785/out/lib'
features=ama-fast-avx2,ama-fast-connected,ama-fast-gravity,ama-reuse-identical-queues,b3-fast-prefilter,b3-fast-chain,compact-chain-summary
<home>/.cargo/bin/cargo build --locked --release --target x86_64-pc-windows-gnu -j 4 \
    --manifest-path native/puyo_core/Cargo.toml --features "$features" > "$out/build.log" 2>&1
cp "$CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/puyo_core.dll" "$out/puyo_core.pyd"
printf '%s\n' "$features" > "$out/features.txt"
# WSL の git は Windows 側の worktree (.git が C:/ のパスを指す) を読めないので、コミットは呼び出し側が渡す。
printf '%s\n' "${LIVE_RECOG_SOURCE_COMMIT:-unknown}" > "$out/source_commit.txt"
sha256sum "$out/puyo_core.pyd" > "$out/SHA256SUMS.txt"
echo "ビルド完了: $out/puyo_core.pyd"
cat "$out/SHA256SUMS.txt"
