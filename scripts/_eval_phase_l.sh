#!/bin/bash
# Phase L 評価: 4 動画 viz + 8 動画 eval + 集計 (= cycle 57b と同フォーマット).
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health phase_l_eval

mkdir -p data/verify/phase_l_viz
mkdir -p data/verify/phase_l_eval
mkdir -p logs/phase_l_eval

CNN_MODEL="models/cnn_phase_l.pt"

declare -A USER_VIDEOS=(
  [v89m7]="data/phase_l/cut/v89m7_buf15s.mp4"
  [v30_match11]="data/holdout_videos/v30_match11_buf15s.mp4"
  [v30_5min]="data/holdout_videos/v30_5min_90s.mp4"
  [v97_match11]="data/holdout_videos/v97_match11_buf15s.mp4"
)
for key in "${!USER_VIDEOS[@]}"; do
  input="${USER_VIDEOS[$key]}"
  output="data/verify/phase_l_viz/${key}_phase_l.mp4"
  board_log="logs/phase_l_eval/viz_${key}.jsonl"
  report="data/verify/phase_l_viz/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] viz @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/phase_l_eval/viz_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/phase_l_eval/eval_${key}.log" 2>&1
  echo "[done] $key"
done

VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)
for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/phase_l_eval/eval_${key}.jsonl"
  report="data/verify/phase_l_eval/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "data/verify/phase_l_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/phase_l_eval/viz8_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/phase_l_eval/eval8_${key}.log" 2>&1
  echo "[done eval] $key"
done

PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

def load(p): return json.load(open(p, encoding='utf-8')) if Path(p).exists() else None
def calc_flicker(d): return sum(v.get('extra', {}).get('total_flips', 0) for v in d.get('violations', []) if v.get('metric') == 'static_color_flicker')

# 4 動画 viz サマリ
user_videos = ['v89m7', 'v30_match11', 'v30_5min', 'v97_match11']
pair_names = {'1-2':'赤→青','2-1':'青→赤','1-3':'赤→緑','3-1':'緑→赤','1-4':'赤→黄','4-1':'黄→赤','1-5':'赤→紫','5-1':'紫→赤','2-3':'青→緑','3-2':'緑→青','2-4':'青→黄','4-2':'黄→青','2-5':'青→紫','5-2':'紫→青','3-4':'緑→黄','4-3':'黄→緑','3-5':'緑→紫','5-3':'紫→緑'}
user_summary = {}
for v in user_videos:
    d = load(f'data/verify/phase_l_viz/{v}.json')
    if not d: continue
    s = d.get('summary', {})
    flicker = calc_flicker(d)
    pair_counts = {}
    ojama_fired = False
    for vi in d.get('violations', []):
        if vi.get('metric') == 'static_color_flicker':
            for k, n in vi.get('extra', {}).get('pair_counts', {}).items():
                pair_counts[k] = pair_counts.get(k, 0) + n
        if vi.get('metric') == 'ojama_global_scarcity':
            ojama_fired = True
    top5 = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:5]
    user_summary[v] = {
        'critical': s.get('critical'), 'warning': s.get('warning'), 'flicker': flicker,
        'ojama_scarcity_fired': ojama_fired,
        'top5': {pair_names.get(k, k): n for k, n in top5},
    }

# 8 動画 eval 集計
videos8 = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'flicker': 0}
per_video = {}
for v in videos8:
    d = load(f'data/verify/phase_l_eval/{v}.json')
    if not d: continue
    s = d.get('summary', {})
    flicker = calc_flicker(d)
    per_video[v] = {'critical': s.get('critical', 0), 'flicker': flicker}
    totals['critical'] += s.get('critical', 0)
    totals['flicker'] += flicker

baseline = load('data/verify/baseline_v3_eval/_summary.json')
cmp = {
    'baseline_critical': baseline['totals']['critical'] if baseline else 1512,
    'phase_l_critical': totals['critical'],
    'phase_l_flicker': totals['flicker'],
    'diff_vs_baseline': totals['critical'] - (baseline['totals']['critical'] if baseline else 1512),
    'per_video': per_video,
    'user_summary': user_summary,
}
Path('data/verify/phase_l_eval/_comparison.json').write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
" 2>&1 | tee data/verify/phase_l_eval/_summary.log

finalize_health 0
