#!/bin/bash
# F 検証 (= STABLE 復帰ゲート、 cycle 56)。
# 既存 default model (= cnn_phase_b_large_v2.pt) に F 適用済の state machine で評価。
# F は model でなく state machine 挙動修正なので model 不変、 認識結果のみ変化。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health F_v1_eval

mkdir -p data/verify/F_v1_eval
mkdir -p logs/F_v1_eval

CNN_MODEL="models/cnn_phase_b_large_v2.pt"

VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)

for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/F_v1_eval/viz_${key}.jsonl"
  report="data/verify/F_v1_eval/${key}.json"

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
    --output "data/verify/F_v1_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/F_v1_eval/viz_${key}.log" 2>&1

  if [ ! -f "$board_log" ]; then
    echo "[fail] viz did not produce board_log for $key"
    continue
  fi

  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    > "logs/F_v1_eval/eval_${key}.log" 2>&1

  echo "[done] $key"
done

# critical 集計 + baseline 比較
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
videos = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'warning': 0}
per_video = {}
for v in videos:
    p = Path(f'data/verify/F_v1_eval/{v}.json')
    if not p.exists():
        continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    per_video[v] = {'critical': s.get('critical', 0), 'warning': s.get('warning', 0), 'verdict': d.get('verdict')}
    totals['critical'] += s.get('critical', 0)
    totals['warning'] += s.get('warning', 0)
out = {'totals': totals, 'per_video': per_video}
Path('data/verify/F_v1_eval/_summary.json').write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8'
)
baseline_p = Path('data/verify/baseline_v3_eval/_summary.json')
if baseline_p.exists():
    base = json.loads(baseline_p.read_text(encoding='utf-8'))
    cmp = {
        'baseline_critical': base['totals']['critical'],
        'F_v1_critical': totals['critical'],
        'diff_critical': totals['critical'] - base['totals']['critical'],
        'pct_change': round((totals['critical'] - base['totals']['critical']) / max(1, base['totals']['critical']) * 100, 1),
        'per_video_comparison': {},
    }
    for v in videos:
        b = base.get('per_video', {}).get(v, {})
        c = per_video.get(v, {})
        cmp['per_video_comparison'][v] = {
            'baseline_critical': b.get('critical', 0),
            'F_v1_critical': c.get('critical', 0),
            'diff': c.get('critical', 0) - b.get('critical', 0),
            'baseline_verdict': b.get('verdict'),
            'F_v1_verdict': c.get('verdict'),
        }
    Path('data/verify/F_v1_eval/_comparison.json').write_text(
        json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(cmp, indent=2, ensure_ascii=False))
else:
    print(json.dumps(out, indent=2, ensure_ascii=False))
"

finalize_health 0
