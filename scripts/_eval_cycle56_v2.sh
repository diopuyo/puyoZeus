#!/bin/bash
# cycle 56_v2 model (= 真 fine-tune 軽量) の baseline_v3 8 動画 eval.
# F + E (= state machine + HSV 修正) は既適用、 model のみ cnn_cycle56_v2.pt に変更。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle56_v2_eval

mkdir -p data/verify/cycle56_v2_eval
mkdir -p logs/cycle56_v2_eval

CNN_MODEL="models/cnn_cycle56_v2.pt"

if [ ! -f "$CNN_MODEL" ]; then
  echo "[fail] CNN model not found: $CNN_MODEL"
  finalize_health 1
  exit 1
fi

VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)

for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/cycle56_v2_eval/viz_${key}.jsonl"
  report="data/verify/cycle56_v2_eval/${key}.json"

  if [ ! -f "$input" ]; then
    echo "[skip] $key (no input)"
    continue
  fi
  if [ -f "$report" ]; then
    echo "[skip] $key (report exists)"
    continue
  fi

  echo "=== [$key] viz @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "data/verify/cycle56_v2_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle56_v2_eval/viz_${key}.log" 2>&1

  if [ ! -f "$board_log" ]; then
    echo "[fail] viz did not produce board_log for $key"
    continue
  fi

  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    > "logs/cycle56_v2_eval/eval_${key}.log" 2>&1

  echo "[done] $key"
done

# critical 集計 + baseline 比較 + F_v1 比較
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
videos = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'warning': 0}
per_video = {}
for v in videos:
    p = Path(f'data/verify/cycle56_v2_eval/{v}.json')
    if not p.exists():
        continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    per_video[v] = {'critical': s.get('critical', 0), 'warning': s.get('warning', 0), 'verdict': d.get('verdict')}
    totals['critical'] += s.get('critical', 0)
    totals['warning'] += s.get('warning', 0)
out = {'totals': totals, 'per_video': per_video}
Path('data/verify/cycle56_v2_eval/_summary.json').write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8'
)
baseline_p = Path('data/verify/baseline_v3_eval/_summary.json')
F_v1_p = Path('data/verify/F_v1_eval/_summary.json')
cmp = {'cycle56_v2_critical': totals['critical']}
if baseline_p.exists():
    base = json.loads(baseline_p.read_text(encoding='utf-8'))
    cmp['baseline_critical'] = base['totals']['critical']
    cmp['diff_vs_baseline'] = totals['critical'] - base['totals']['critical']
if F_v1_p.exists():
    fv1 = json.loads(F_v1_p.read_text(encoding='utf-8'))
    cmp['F_v1_critical'] = fv1['totals']['critical']
    cmp['diff_vs_F_v1'] = totals['critical'] - fv1['totals']['critical']
cmp['per_video'] = per_video
Path('data/verify/cycle56_v2_eval/_comparison.json').write_text(
    json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8'
)
print(json.dumps(cmp, indent=2, ensure_ascii=False))
"

finalize_health 0
