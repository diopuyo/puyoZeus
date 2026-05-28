#!/bin/bash
# η + Stage 1 採用後 (HEAD b42a0c9) の 12 動画 eval スクリプト。
# cnn_phase_l.pt (= load_default) + per-video HSV inject (--hsv-state 省略) で実行。
# baseline_v3 critical 合計 1512 との比較基準で退行有無を判定する。
# 並列上限 3 (= CLAUDE.md / CYCLE_FINDINGS.md ルール準拠)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health eta_stage1_eval

mkdir -p data/verify/eta_stage1_eval
mkdir -p logs/eta_stage1_eval

CNN_MODEL="models/cnn_phase_l.pt"
MAX_PARALLEL=3

# 1 動画分の viz + evaluate を実行する関数
run_one() {
  local key="$1"
  local input="$2"
  local output="$3"
  local report="$4"
  local board_log="$5"
  if [ -f "$report" ]; then
    echo "[skip] $key (report exists)"
    return
  fi
  if [ ! -f "$input" ]; then
    echo "[skip] $key (input missing: $input)"
    return
  fi
  echo "=== [$key] viz @ $(date) ==="
  # --hsv-state 省略 = resolve_hsv_path() が動画 ID から per-video JSON を自動選択
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --dump-board-log "$board_log" \
    > "logs/eta_stage1_eval/run_${key}.log" 2>&1
  if [ ! -f "$board_log" ]; then
    echo "[fail viz] $key (board_log not generated)"
    return
  fi
  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    >> "logs/eta_stage1_eval/run_${key}.log" 2>&1
  echo "[done] $key"
}

# タスク一覧 (= "key|input|output|report|board_log")
# 4 動画: viz 用 + 8 動画: eval 用
TASKS=(
  "v89m7|data/phase_l/cut/v89m7_buf15s.mp4|data/verify/eta_stage1_eval/v89m7.mp4|data/verify/eta_stage1_eval/v89m7.json|logs/eta_stage1_eval/board_v89m7.jsonl"
  "v30_match11|data/holdout_videos/v30_match11_buf15s.mp4|data/verify/eta_stage1_eval/v30_match11.mp4|data/verify/eta_stage1_eval/v30_match11.json|logs/eta_stage1_eval/board_v30_match11.jsonl"
  "v30_5min|data/holdout_videos/v30_5min_90s.mp4|data/verify/eta_stage1_eval/v30_5min.mp4|data/verify/eta_stage1_eval/v30_5min.json|logs/eta_stage1_eval/board_v30_5min.jsonl"
  "v97_match11|data/holdout_videos/v97_match11_buf15s.mp4|data/verify/eta_stage1_eval/v97_match11.mp4|data/verify/eta_stage1_eval/v97_match11.json|logs/eta_stage1_eval/board_v97_match11.jsonl"
  "v29m2|data/baseline_videos_v3/v29m2_buf15s.mp4|data/verify/eta_stage1_eval/v29m2.mp4|data/verify/eta_stage1_eval/v29m2.json|logs/eta_stage1_eval/board_v29m2.jsonl"
  "v40m7|data/baseline_videos_v3/v40m7_buf15s.mp4|data/verify/eta_stage1_eval/v40m7.mp4|data/verify/eta_stage1_eval/v40m7.json|logs/eta_stage1_eval/board_v40m7.jsonl"
  "v51m2|data/baseline_videos_v3/v51m2_buf15s.mp4|data/verify/eta_stage1_eval/v51m2.mp4|data/verify/eta_stage1_eval/v51m2.json|logs/eta_stage1_eval/board_v51m2.jsonl"
  "v57m2|data/baseline_videos_v3/v57m2_buf15s.mp4|data/verify/eta_stage1_eval/v57m2.mp4|data/verify/eta_stage1_eval/v57m2.json|logs/eta_stage1_eval/board_v57m2.jsonl"
  "v70m2|data/baseline_videos_v3/v70m2_buf15s.mp4|data/verify/eta_stage1_eval/v70m2.mp4|data/verify/eta_stage1_eval/v70m2.json|logs/eta_stage1_eval/board_v70m2.jsonl"
  "v89m3|data/baseline_videos_v3/v89m3_buf15s.mp4|data/verify/eta_stage1_eval/v89m3.mp4|data/verify/eta_stage1_eval/v89m3.json|logs/eta_stage1_eval/board_v89m3.jsonl"
  "v95m15|data/baseline_videos_v3/v95m15_buf15s.mp4|data/verify/eta_stage1_eval/v95m15.mp4|data/verify/eta_stage1_eval/v95m15.json|logs/eta_stage1_eval/board_v95m15.jsonl"
  "v97m11|data/baseline_videos_v3/v97m11_buf15s.mp4|data/verify/eta_stage1_eval/v97m11.mp4|data/verify/eta_stage1_eval/v97m11.json|logs/eta_stage1_eval/board_v97m11.jsonl"
)

pids=()
running=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r key input output report board_log <<< "$task"
  run_one "$key" "$input" "$output" "$report" "$board_log" &
  pids+=($!)
  ((running++)) || true
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
    ((running--)) || true
  fi
done
wait

echo "=== all done @ $(date) ==="

# 集計 (baseline_v3 critical 1512 と比較)
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

BASELINE_CRITICAL = 1512

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def calc_flicker(d):
    return sum(
        v.get('extra', {}).get('total_flips', 0)
        for v in d.get('violations', [])
        if v.get('metric') == 'static_color_flicker'
    )

all_videos = [
    'v89m7', 'v30_match11', 'v30_5min', 'v97_match11',
    'v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11',
]
totals = {'critical': 0, 'flicker': 0}
per_video = {}
flags = []

for v in all_videos:
    d = load(f'data/verify/eta_stage1_eval/{v}.json')
    if not d:
        per_video[v] = {'critical': None, 'flicker': None, 'status': 'MISSING'}
        continue
    s = d.get('summary', {})
    c = s.get('critical', 0)
    flicker = calc_flicker(d)
    per_video[v] = {'critical': c, 'flicker': flicker, 'verdict': d.get('verdict')}
    totals['critical'] += c
    totals['flicker'] += flicker

# baseline_v3_eval の per-video critical (= 8 動画分のみ存在)
baseline_data = load('data/verify/baseline_v3_eval/_summary.json')
baseline_per = baseline_data.get('per_video', {}) if baseline_data else {}

# 退行 flag (= +20 以上悪化した動画)
for v, info in per_video.items():
    if info.get('critical') is None:
        continue
    b = baseline_per.get(v, {}).get('critical')
    if b is not None and info['critical'] - b >= 20:
        flags.append({'video': v, 'current': info['critical'], 'baseline': b, 'diff': info['critical'] - b})

cmp = {
    'baseline_critical_total': BASELINE_CRITICAL,
    'eta_stage1_critical_total': totals['critical'],
    'diff_vs_baseline': totals['critical'] - BASELINE_CRITICAL,
    'pct_vs_baseline': round((totals['critical'] - BASELINE_CRITICAL) / max(1, BASELINE_CRITICAL) * 100, 1),
    'flicker_total': totals['flicker'],
    'per_video': per_video,
    'regression_flags': flags,
    'verdict': 'ACCEPT' if totals['critical'] <= BASELINE_CRITICAL else 'REJECT',
}
out = Path('data/verify/eta_stage1_eval/_comparison.json')
out.write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
" 2>&1 | tee logs/eta_stage1_eval/_summary.log

finalize_health 0
