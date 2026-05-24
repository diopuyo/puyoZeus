#!/bin/bash
# Step 2: 38 動画 union 版 _merged_default.json の効果評価.
# 8 動画 + 4 動画 (= ユーザー目視) 並列 3 で測定。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/verify/merged38_viz
mkdir -p data/verify/merged38_eval
mkdir -p logs/merged38_eval

CNN_MODEL="models/cnn_phase_b_large_v2.pt"
HSV_STATE="data/per_video_hsv_ranges/_merged_default.json"
MAX_PARALLEL=3

run_one() {
  local key="$1" input="$2" output="$3" report="$4" board_log="$5"
  if [ -f "$report" ]; then return; fi
  if [ ! -f "$input" ]; then return; fi
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state "$HSV_STATE" \
    --dump-board-log "$board_log" \
    > "logs/merged38_eval/run_${key}.log" 2>&1
  if [ ! -f "$board_log" ]; then return; fi
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    >> "logs/merged38_eval/run_${key}.log" 2>&1
  echo "[done] $key"
}

TASKS=(
  "v89m7|data/phase_l/cut/v89m7_buf15s.mp4|data/verify/merged38_viz/v89m7_merged38.mp4|data/verify/merged38_viz/v89m7.json|logs/merged38_eval/viz_v89m7.jsonl"
  "v30_match11|data/holdout_videos/v30_match11_buf15s.mp4|data/verify/merged38_viz/v30_match11_merged38.mp4|data/verify/merged38_viz/v30_match11.json|logs/merged38_eval/viz_v30_match11.jsonl"
  "v30_5min|data/holdout_videos/v30_5min_90s.mp4|data/verify/merged38_viz/v30_5min_merged38.mp4|data/verify/merged38_viz/v30_5min.json|logs/merged38_eval/viz_v30_5min.jsonl"
  "v97_match11|data/holdout_videos/v97_match11_buf15s.mp4|data/verify/merged38_viz/v97_match11_merged38.mp4|data/verify/merged38_viz/v97_match11.json|logs/merged38_eval/viz_v97_match11.jsonl"
  "v29m2|data/baseline_videos_v3/v29m2_buf15s.mp4|data/verify/merged38_eval/v29m2.mp4|data/verify/merged38_eval/v29m2.json|logs/merged38_eval/eval_v29m2.jsonl"
  "v40m7|data/baseline_videos_v3/v40m7_buf15s.mp4|data/verify/merged38_eval/v40m7.mp4|data/verify/merged38_eval/v40m7.json|logs/merged38_eval/eval_v40m7.jsonl"
  "v51m2|data/baseline_videos_v3/v51m2_buf15s.mp4|data/verify/merged38_eval/v51m2.mp4|data/verify/merged38_eval/v51m2.json|logs/merged38_eval/eval_v51m2.jsonl"
  "v57m2|data/baseline_videos_v3/v57m2_buf15s.mp4|data/verify/merged38_eval/v57m2.mp4|data/verify/merged38_eval/v57m2.json|logs/merged38_eval/eval_v57m2.jsonl"
  "v70m2|data/baseline_videos_v3/v70m2_buf15s.mp4|data/verify/merged38_eval/v70m2.mp4|data/verify/merged38_eval/v70m2.json|logs/merged38_eval/eval_v70m2.jsonl"
  "v89m3|data/baseline_videos_v3/v89m3_buf15s.mp4|data/verify/merged38_eval/v89m3.mp4|data/verify/merged38_eval/v89m3.json|logs/merged38_eval/eval_v89m3.jsonl"
  "v95m15|data/baseline_videos_v3/v95m15_buf15s.mp4|data/verify/merged38_eval/v95m15.mp4|data/verify/merged38_eval/v95m15.json|logs/merged38_eval/eval_v95m15.jsonl"
  "v97m11|data/baseline_videos_v3/v97m11_buf15s.mp4|data/verify/merged38_eval/v97m11.mp4|data/verify/merged38_eval/v97m11.json|logs/merged38_eval/eval_v97m11.jsonl"
)

pids=()
running=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r key input output report board_log <<< "$task"
  run_one "$key" "$input" "$output" "$report" "$board_log" &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"; pids=("${pids[@]:1}"); ((running--)) || true
  fi
done
wait

echo "=== all done @ $(date) ==="

# 集計 (= 既存 baseline 1512 と比較)
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
def load(p): return json.load(open(p, encoding='utf-8')) if Path(p).exists() else None
def calc_flicker(d): return sum(v.get('extra', {}).get('total_flips', 0) for v in d.get('violations', []) if v.get('metric') == 'static_color_flicker')

videos8 = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'flicker': 0}
per_video = {}
for v in videos8:
    d = load(f'data/verify/merged38_eval/{v}.json')
    if not d: continue
    s = d.get('summary', {})
    flicker = calc_flicker(d)
    per_video[v] = {'critical': s.get('critical', 0), 'flicker': flicker}
    totals['critical'] += s.get('critical', 0)
    totals['flicker'] += flicker

user_videos = ['v89m7', 'v30_match11', 'v30_5min', 'v97_match11']
user_summary = {}
for v in user_videos:
    d = load(f'data/verify/merged38_viz/{v}.json')
    if not d: continue
    s = d.get('summary', {})
    flicker = calc_flicker(d)
    ojama_fired = [vi.get('side') for vi in d.get('violations', []) if vi.get('metric') == 'ojama_global_scarcity']
    user_summary[v] = {'critical': s.get('critical'), 'flicker': flicker, 'ojama_fired': ojama_fired}

baseline = load('data/verify/baseline_v3_eval/_summary.json')
existing_critical = baseline['totals']['critical'] if baseline else 1512

cmp = {
    'baseline_existing_critical': existing_critical,
    'merged38_critical': totals['critical'],
    'merged38_flicker': totals['flicker'],
    'diff_vs_baseline': totals['critical'] - existing_critical,
    'pct_vs_baseline': round((totals['critical'] - existing_critical) / max(1, existing_critical) * 100, 1),
    'per_video': per_video,
    'user_summary': user_summary,
}
Path('data/verify/merged38_eval/_comparison.json').write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
" 2>&1 | tee data/verify/merged38_eval/_summary.log
