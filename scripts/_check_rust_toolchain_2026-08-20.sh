#!/bin/bash
# Rust ツールチェーンの有無を確認する (2026-08-20)。
# HSV 分類の Rust 化に着手する前に、ビルドできる状態かを確定させる。
# PATH に空白やカッコを含むため wsl 直書きでは MSYS に壊される
# (memory feedback_msys_pipe_escape) — スクリプト化する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

export PATH="$HOME/.cargo/bin:$PATH"

echo "=== cargo ==="
if command -v cargo > /dev/null 2>&1; then
  cargo --version
else
  echo "  cargo が PATH に無い"
  echo "  ~/.cargo/bin の中身:"
  ls -1 "$HOME/.cargo/bin" 2>/dev/null | head -10 || echo "    (ディレクトリ自体が無い)"
fi

echo "=== rustc ==="
rustc --version 2>/dev/null || echo "  rustc なし"

echo "=== maturin ==="
./venv/bin/maturin --version 2>/dev/null || echo "  maturin なし"

echo "=== 既存ビルド成果物 ==="
ls -l venv/lib/python3.12/site-packages/puyo_core/*.so 2>/dev/null | head -2

echo "=== ビルドキャッシュ (過去にビルドできた証拠) ==="
ls -d native/puyo_core/target/release 2>/dev/null && \
  ls -1 native/puyo_core/target/release/*.so 2>/dev/null | head -2
echo "  target/release のタイムスタンプ:"
stat -c "%y" native/puyo_core/target/release 2>/dev/null || echo "    なし"
