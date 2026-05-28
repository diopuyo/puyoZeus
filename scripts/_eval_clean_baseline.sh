#!/bin/bash
# clean_baseline eval: η 撤回後 (revert-eta branch HEAD) の 8 動画評価。
# baseline_v3_eval と同一 8 動画を新規測定し、今後の比較 anchor とする。
# cnn_phase_l.pt (= load_default) + per-video HSV inject (--hsv-state 省略) で実行。
# 並列上限 3 (= CLAUDE.md / CYCLE_FINDINGS.md ルール準拠)。
# 出力: data/verify/clean_baseline_eval/ (= 既存 eval ディレクトリを絶対上書きしない)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health clean_baseline_eval

mkdir -p data/verify/clean_baseline_eval
mkdir -p logs/clean_baseline_eval

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
    > "logs/clean_baseline_eval/run_${key}.log" 2>&1
  if [ ! -f "$board_log" ]; then
    echo "[fail viz] $key (board_log not generated)"
    return
  fi
  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    >> "logs/clean_baseline_eval/run_${key}.log" 2>&1
  echo "[done] $key"
}

# タスク一覧 (= baseline_v3_eval と同一 8 動画)
# フォーマット: "key|input|output|report|board_log"
TASKS=(
  "v29m2|data/baseline_videos_v3/v29m2_buf15s.mp4|data/verify/clean_baseline_eval/v29m2.mp4|data/verify/clean_baseline_eval/v29m2.json|logs/clean_baseline_eval/board_v29m2.jsonl"
  "v40m7|data/baseline_videos_v3/v40m7_buf15s.mp4|data/verify/clean_baseline_eval/v40m7.mp4|data/verify/clean_baseline_eval/v40m7.json|logs/clean_baseline_eval/board_v40m7.jsonl"
  "v51m2|data/baseline_videos_v3/v51m2_buf15s.mp4|data/verify/clean_baseline_eval/v51m2.mp4|data/verify/clean_baseline_eval/v51m2.json|logs/clean_baseline_eval/board_v51m2.jsonl"
  "v57m2|data/baseline_videos_v3/v57m2_buf15s.mp4|data/verify/clean_baseline_eval/v57m2.mp4|data/verify/clean_baseline_eval/v57m2.json|logs/clean_baseline_eval/board_v57m2.jsonl"
  "v70m2|data/baseline_videos_v3/v70m2_buf15s.mp4|data/verify/clean_baseline_eval/v70m2.mp4|data/verify/clean_baseline_eval/v70m2.json|logs/clean_baseline_eval/board_v70m2.jsonl"
  "v89m3|data/baseline_videos_v3/v89m3_buf15s.mp4|data/verify/clean_baseline_eval/v89m3.mp4|data/verify/clean_baseline_eval/v89m3.json|logs/clean_baseline_eval/board_v89m3.jsonl"
  "v95m15|data/baseline_videos_v3/v95m15_buf15s.mp4|data/verify/clean_baseline_eval/v95m15.mp4|data/verify/clean_baseline_eval/v95m15.json|logs/clean_baseline_eval/board_v95m15.jsonl"
  "v97m11|data/baseline_videos_v3/v97m11_buf15s.mp4|data/verify/clean_baseline_eval/v97m11.mp4|data/verify/clean_baseline_eval/v97m11.json|logs/clean_baseline_eval/board_v97m11.jsonl"
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

# 集計 (= これ自体が新 baseline になるため baseline 比較なし)
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

VIDEOS = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def calc_flicker(d):
    return sum(
        v.get('extra', {}).get('total_flips', 0)
        for v in d.get('violations', [])
        if v.get('metric') == 'static_color_flicker'
    )

totals = {'critical': 0, 'warning': 0, 'flicker': 0}
per_video = {}

for v in VIDEOS:
    d = load(f'data/verify/clean_baseline_eval/{v}.json')
    if not d:
        per_video[v] = {'critical': None, 'warning': None, 'flicker': None, 'status': 'MISSING'}
        continue
    s = d.get('summary', {})
    c = s.get('critical', 0)
    w = s.get('warning', 0)
    flicker = calc_flicker(d)
    per_video[v] = {
        'critical': c,
        'warning': w,
        'flicker': flicker,
        'verdict': d.get('verdict'),
    }
    totals['critical'] += c
    totals['warning'] += w
    totals['flicker'] += flicker

result = {
    'description': 'η 撤回後 clean state の 8 動画 eval (新 baseline anchor)',
    'totals': totals,
    'per_video': per_video,
}
out = Path('data/verify/clean_baseline_eval/_summary.json')
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
" 2>&1 | tee logs/clean_baseline_eval/_summary.log

finalize_health 0
