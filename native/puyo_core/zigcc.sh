#!/bin/bash
# cargo linker 用の zig cc ラッパー (root権限なしでCリンカを確保する手段、2026-08-12)。
# WSL に build-essential (gcc) が未導入 + sudo がパスワード要求で使えないため、
# pip 配布の ziglang (zig の cc 互換ドライバ) を linker として代用する。
exec /home/ryouj/zigvenv/bin/python -m ziglang cc "$@"
