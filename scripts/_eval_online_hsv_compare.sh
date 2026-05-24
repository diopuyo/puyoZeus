#!/bin/bash
# Step 0 (2026-05-24): OnlineHsvCalibrator 効果定量評価.
# 8 動画 baseline_v3 + baseline model で 2 設定比較:
#   A: --no-online-hsv (= 純粋 baseline)
#   B: 通常 (= online_hsv 学習 active)
# 既存 baseline_v3_eval (= --hsv-state あり + online_hsv suppressed) を参考値とする。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/verify/online_hsv_off_eval
mkdir -p data/verify/online_hsv_on_eval
mkdir -p logs/online_hsv_eval

CNN_MODEL="models/cnn_phase_b_large_v2.pt"
MAX_PARALLEL=3

run_one() {
  local mode="$1"
  local key="$2"
  local input="$3"
  local output="$4"
  local report="$5"
  local board_log="$6"
  if [ -f "$report" ]; then return; fi
  if [ ! -f "$input" ]; then return; fi
  local extra_flag=""
  if [ "$mode" = "off" ]; then
    extra_flag="--no-online-hsv"
  fi
  # 共通: --hsv-state は渡さない (= 純粋比較、 DB inject なし)
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --dump-board-log "$board_log" \
    $extra_flag \
    > "logs/online_hsv_eval/${mode}_${key}.log" 2>&1
  if [ ! -f "$board_log" ]; then return; fi
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    >> "logs/online_hsv_eval/${mode}_${key}.log" 2>&1
  echo "[done] $mode/$key"
}

VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)

# 設定 A: online_hsv OFF
pids=()
running=0
for key in "${VIDEOS[@]}"; do
  run_one "off" "$key" \
    "data/baseline_videos_v3/${key}_buf15s.mp4" \
    "data/verify/online_hsv_off_eval/${key}.mp4" \
    "data/verify/online_hsv_off_eval/${key}.json" \
    "logs/online_hsv_eval/off_${key}.jsonl" &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"; pids=("${pids[@]:1}"); ((running--)) || true
  fi
done
wait

# 設定 B: online_hsv ON
pids=()
running=0
for key in "${VIDEOS[@]}"; do
  run_one "on" "$key" \
    "data/baseline_videos_v3/${key}_buf15s.mp4" \
    "data/verify/online_hsv_on_eval/${key}.mp4" \
    "data/verify/online_hsv_on_eval/${key}.json" \
    "logs/online_hsv_eval/on_${key}.jsonl" &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"; pids=("${pids[@]:1}"); ((running--)) || true
  fi
done
wait

# 集計
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
def load(p): return json.load(open(p, encoding='utf-8')) if Path(p).exists() else None
def calc_flicker(d): return sum(v.get('extra', {}).get('total_flips', 0) for v in d.get('violations', []) if v.get('metric') == 'static_color_flicker')

videos = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'off': {'critical': 0, 'flicker': 0}, 'on': {'critical': 0, 'flicker': 0}}
per_video = {}
for v in videos:
    per_video[v] = {}
    for mode in ['off', 'on']:
        d = load(f'data/verify/online_hsv_{mode}_eval/{v}.json')
        if not d: continue
        s = d.get('summary', {})
        flicker = calc_flicker(d)
        per_video[v][mode] = {'critical': s.get('critical', 0), 'flicker': flicker}
        totals[mode]['critical'] += s.get('critical', 0)
        totals[mode]['flicker'] += flicker

# 既存 baseline (= hsv-state あり + online_hsv suppressed) も比較
baseline_existing = load('data/verify/baseline_v3_eval/_summary.json')
existing_critical = baseline_existing['totals']['critical'] if baseline_existing else 1512

cmp = {
    'config_descriptions': {
        'A_off': 'hsv-state なし + online_hsv 完全 OFF (= 純粋 baseline)',
        'B_on': 'hsv-state なし + online_hsv 動画中学習 ON',
        'existing_baseline': 'hsv-state あり (= DB inject) + online_hsv suppressed (= 既存 baseline_v3_eval)',
    },
    'totals': {
        'A_off_critical': totals['off']['critical'],
        'A_off_flicker': totals['off']['flicker'],
        'B_on_critical': totals['on']['critical'],
        'B_on_flicker': totals['on']['flicker'],
        'existing_baseline_critical': existing_critical,
    },
    'online_hsv_effect': {
        'critical_diff_B_minus_A': totals['on']['critical'] - totals['off']['critical'],
        'flicker_diff_B_minus_A': totals['on']['flicker'] - totals['off']['flicker'],
        'critical_pct': round((totals['on']['critical'] - totals['off']['critical']) / max(1, totals['off']['critical']) * 100, 1),
    },
    'db_inject_effect': {
        'critical_diff_existing_minus_A': existing_critical - totals['off']['critical'],
        'pct': round((existing_critical - totals['off']['critical']) / max(1, totals['off']['critical']) * 100, 1),
    },
    'per_video': per_video,
}
Path('data/verify/online_hsv_compare.json').write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
" 2>&1 | tee data/verify/online_hsv_compare.log

echo "=== done @ $(date) ==="
