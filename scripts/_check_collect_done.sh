#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

echo "=== alive python collect workers ==="
n=$(pgrep -cf '\.venv/bin/python.*phase_i_collect_pseudo_labels')
echo "count: $n"
pgrep -af '\.venv/bin/python.*phase_i_collect_pseudo_labels' || echo "(none)"

echo ""
echo "=== done lines per video ==="
done_count=0
for v in 29 30 31 32 33 40 51 57 70 89; do
  log="logs/phase_i_collect_v${v}.log"
  if [ -f "$log" ]; then
    line=$(grep "done video_id" "$log" | tail -1)
    if [ -n "$line" ]; then
      echo "v${v}: DONE - $line"
      done_count=$((done_count + 1))
    else
      lastline=$(tail -1 "$log")
      echo "v${v}: not done - last: $lastline"
    fi
  fi
done
echo "DONE: $done_count / 10"

echo ""
echo "=== fine_tune ==="
ps -eo pid,args | grep -E 'phase_i_fine_tune' | grep -v grep || echo "(none)"

echo ""
echo "=== visualize ==="
ps -eo pid,args | grep -E 'visualize_recognition' | grep -v grep || echo "(none)"

echo ""
echo "=== model + viz videos ==="
ls -la models/cnn_phase_b_finetuned.pt 2>/dev/null || echo "model: NO"
ls -la data/evaluation_videos/v29_match2_phase_i_viz.mp4 2>/dev/null || echo "v29 viz: NO"
ls -la data/evaluation_videos/v89_match3_phase_i_viz.mp4 2>/dev/null || echo "v89 viz: NO"
