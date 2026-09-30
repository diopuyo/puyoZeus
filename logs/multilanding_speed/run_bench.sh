#!/bin/bash
# 保存した証明入力の単体再実行 (基準の全文 SHA と照合)。nice19・1 プロセス。
bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed/py.sh -m scripts.bench_multilanding_prove \
  logs/multilanding_speed/base_head/profile_fcXG83vInDY.jsonl logs/multilanding_speed/base_head/profile_q_7gc4TgFig.jsonl \
  logs/multilanding_speed/base_head/profile_review.jsonl logs/multilanding_speed/base_head/profile_zenchi.jsonl \
  logs/multilanding_speed/base_head/profile_live_b20_on.jsonl logs/multilanding_speed/base_head/profile_live_b18_run5.jsonl \
  logs/multilanding_speed/base_head/profile_live_b18_stall.jsonl --out logs/multilanding_speed/bench_all.json > logs/multilanding_speed/bench_all.log 2>&1
