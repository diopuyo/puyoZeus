#!/bin/bash
# 案 K: per-video HSV 直接 inject 評価スクリプト (2026-05-24)
# 動画名から動画 ID を自動抽出 → 該当 per-video JSON を inject (resolve_hsv_path)
# --hsv-state 省略で自動選択 (= 案 K の本体は visualize_recognition.py 側)
# 8 動画 eval + 4 動画 viz を並列 3 で実行。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/verify/per_video_inject_viz
mkdir -p data/verify/per_video_inject_eval
mkdir -p logs/per_video_inject

CNN_MODEL="models/cnn_phase_b_large_v2.pt"
MAX_PARALLEL=3

run_one() {
  local key="$1" input="$2" output="$3" report="$4" board_log="$5"
  if [ -f "$report" ]; then
    echo "[skip] $key (report exists)"
    return
  fi
  if [ ! -f "$input" ]; then
    echo "[skip] $key (input missing: $input)"
    return
  fi
  # --hsv-state 省略 = resolve_hsv_path() が動画 ID から自動選択 (案 K)
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --dump-board-log "$board_log" \
    > "logs/per_video_inject/run_${key}.log" 2>&1
  if [ ! -f "$board_log" ]; then
    echo "[warn] no board_log: $key"
    return
  fi
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    >> "logs/per_video_inject/run_${key}.log" 2>&1
  echo "[done] $key"
}

# viz 4 動画 (= ユーザー目視用) + eval 8 動画
TASKS=(
  "v89m7|data/phase_l/cut/v89m7_buf15s.mp4|data/verify/per_video_inject_viz/v89m7.mp4|data/verify/per_video_inject_viz/v89m7.json|logs/per_video_inject/viz_v89m7.jsonl"
  "v30_match11|data/holdout_videos/v30_match11_buf15s.mp4|data/verify/per_video_inject_viz/v30_match11.mp4|data/verify/per_video_inject_viz/v30_match11.json|logs/per_video_inject/viz_v30_match11.jsonl"
  "v30_5min|data/holdout_videos/v30_5min_90s.mp4|data/verify/per_video_inject_viz/v30_5min.mp4|data/verify/per_video_inject_viz/v30_5min.json|logs/per_video_inject/viz_v30_5min.jsonl"
  "v97_match11|data/holdout_videos/v97_match11_buf15s.mp4|data/verify/per_video_inject_viz/v97_match11.mp4|data/verify/per_video_inject_viz/v97_match11.json|logs/per_video_inject/viz_v97_match11.jsonl"
  "v29m2|data/baseline_videos_v3/v29m2_buf15s.mp4|data/verify/per_video_inject_eval/v29m2.mp4|data/verify/per_video_inject_eval/v29m2.json|logs/per_video_inject/eval_v29m2.jsonl"
  "v40m7|data/baseline_videos_v3/v40m7_buf15s.mp4|data/verify/per_video_inject_eval/v40m7.mp4|data/verify/per_video_inject_eval/v40m7.json|logs/per_video_inject/eval_v40m7.jsonl"
  "v51m2|data/baseline_videos_v3/v51m2_buf15s.mp4|data/verify/per_video_inject_eval/v51m2.mp4|data/verify/per_video_inject_eval/v51m2.json|logs/per_video_inject/eval_v51m2.jsonl"
  "v57m2|data/baseline_videos_v3/v57m2_buf15s.mp4|data/verify/per_video_inject_eval/v57m2.mp4|data/verify/per_video_inject_eval/v57m2.json|logs/per_video_inject/eval_v57m2.jsonl"
  "v70m2|data/baseline_videos_v3/v70m2_buf15s.mp4|data/verify/per_video_inject_eval/v70m2.mp4|data/verify/per_video_inject_eval/v70m2.json|logs/per_video_inject/eval_v70m2.jsonl"
  "v89m3|data/baseline_videos_v3/v89m3_buf15s.mp4|data/verify/per_video_inject_eval/v89m3.mp4|data/verify/per_video_inject_eval/v89m3.json|logs/per_video_inject/eval_v89m3.jsonl"
  "v95m15|data/baseline_videos_v3/v95m15_buf15s.mp4|data/verify/per_video_inject_eval/v95m15.mp4|data/verify/per_video_inject_eval/v95m15.json|logs/per_video_inject/eval_v95m15.jsonl"
  "v97m11|data/baseline_videos_v3/v97m11_buf15s.mp4|data/verify/per_video_inject_eval/v97m11.mp4|data/verify/per_video_inject_eval/v97m11.json|logs/per_video_inject/eval_v97m11.jsonl"
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

# 集計 (baseline 1512 と比較)
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def calc_flicker(d):
    return sum(
        v.get('extra', {}).get('total_flips', 0)
        for v in d.get('violations', [])
        if v.get('metric') == 'static_color_flicker'
    )

videos8 = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'flicker': 0}
per_video = {}
for v in videos8:
    d = load(f'data/verify/per_video_inject_eval/{v}.json')
    if not d:
        continue
    s = d.get('summary', {})
    flicker = calc_flicker(d)
    per_video[v] = {'critical': s.get('critical', 0), 'flicker': flicker}
    totals['critical'] += s.get('critical', 0)
    totals['flicker'] += flicker

user_videos = ['v89m7', 'v30_match11', 'v30_5min', 'v97_match11']
user_summary = {}
for v in user_videos:
    d = load(f'data/verify/per_video_inject_viz/{v}.json')
    if not d:
        continue
    s = d.get('summary', {})
    flicker = calc_flicker(d)
    ojama_fired = [vi.get('side') for vi in d.get('violations', []) if vi.get('metric') == 'ojama_global_scarcity']
    user_summary[v] = {'critical': s.get('critical'), 'flicker': flicker, 'ojama_fired': ojama_fired}

baseline = load('data/verify/baseline_v3_eval/_summary.json')
existing_critical = baseline['totals']['critical'] if baseline else 1512

merged38 = load('data/verify/merged38_eval/_comparison.json')
merged38_critical = merged38['merged38_critical'] if merged38 else None

cmp = {
    'baseline_existing_critical': existing_critical,
    'per_video_inject_critical': totals['critical'],
    'per_video_inject_flicker': totals['flicker'],
    'diff_vs_baseline': totals['critical'] - existing_critical,
    'pct_vs_baseline': round((totals['critical'] - existing_critical) / max(1, existing_critical) * 100, 1),
    'merged38_critical_ref': merged38_critical,
    'diff_vs_merged38': (totals['critical'] - merged38_critical) if merged38_critical is not None else None,
    'per_video': per_video,
    'user_summary': user_summary,
}
out = Path('data/verify/per_video_inject_eval/_comparison.json')
out.write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
" 2>&1 | tee data/verify/per_video_inject_eval/_summary.log
