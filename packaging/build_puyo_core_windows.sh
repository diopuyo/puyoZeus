#!/bin/bash
# puyo_core (Rust / PyO3) を Windows 用 wheel (cp312, windows-gnu) としてビルドする手順 (Git Bash から)。
# 2026-09-30 に下記を個別に実行して成功した手順を 1 本にまとめたもの (一括実行での再検証はしていない)。
# MSVC (Visual Studio Build Tools) も管理者権限も要らない: Rust の windows-gnu ターゲット + zig をリンカにする。
# 出力: D:/puyo_analyzer/packaging/native_build/wheels/puyo_core-0.1.0-cp312-cp312-win_amd64.whl (build_bundle.py が同梱)
set -euo pipefail
WORK=/d/puyo_analyzer/packaging
PY314=/c/Users/ryouj/AppData/Local/Python/pythoncore-3.14-64/python.exe   # ビルド用ホスト Python (pip 用)
REPO="${1:?リポジトリのルート (native/puyo_core を含む)}"

# 1) Rust を D: に導入 (ユーザー権限、PATH は汚さない)
mkdir -p "$WORK/rust" && cd "$WORK/rust"
curl -sSL -o rustup-init.exe https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe
RUSTUP_HOME=D:/puyo_analyzer/packaging/rust/rustup CARGO_HOME=D:/puyo_analyzer/packaging/rust/cargo \
  ./rustup-init.exe -y --default-host x86_64-pc-windows-gnu --default-toolchain stable --profile minimal --no-modify-path

# 2) maturin (exe は wheel の .data/scripts にある。pip --target ではスクリプトが落ちるため wheel を直接展開) と zig
mkdir -p "$WORK/native_build" && cd "$WORK/native_build"
"$PY314" -m pip download maturin --no-deps --only-binary=:all: -d mw
"$PY314" -m zipfile -e mw/maturin-*.whl mwx
"$PY314" -m pip download ziglang --no-deps --only-binary=:all: --platform win_amd64 -d zw
"$PY314" -c "import zipfile,glob; zipfile.ZipFile(glob.glob('zw/ziglang-*.whl')[0]).extractall('zx')"   # 19.5 千ファイル。数分かかる

# 3) ソースをビルド用コピーへ (target/ を作業ツリーに作らない)。実行環境に Python 開発ファイルが無いため
#    pyo3 に generate-import-lib feature を足す (ビルド用コピーのみ。リポジトリの Cargo.toml は無改変)。
rm -rf puyo_core && cp -r "$REPO/native/puyo_core" . && rm -rf puyo_core/target
sed -i 's/features = \["extension-module"\]/features = ["extension-module", "generate-import-lib"]/' puyo_core/Cargo.toml

# 4) ビルド (約 5 分、AV が効いている環境ではさらに遅い)。-i python3.12 は maturin 内蔵の sysconfig を使う疑似指定。
cd puyo_core
export RUSTUP_HOME=D:/puyo_analyzer/packaging/rust/rustup CARGO_HOME=D:/puyo_analyzer/packaging/rust/cargo
export PATH="$WORK/rust/cargo/bin:$WORK/native_build/zx/ziglang:$PATH" PYTHONPATH="$WORK/native_build/zx"
"$WORK/native_build/mwx/maturin-"*.data/scripts/maturin.exe build --release --target x86_64-pc-windows-gnu --zig \
  -i python3.12 -o "$WORK/native_build/wheels"
