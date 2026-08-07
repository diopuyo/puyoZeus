#!/bin/bash
# 走行中 collect_boards_lean の一覧 (2026-08-05、使い捨て)
pgrep -a -f collect_boards_lean | sed 's/--enable-chain-tracker.*//' | head -12
echo "---count---"
pgrep -c -f collect_boards_lean
