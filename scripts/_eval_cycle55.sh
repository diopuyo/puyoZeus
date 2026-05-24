#!/bin/bash
# cycle 55 (= cnn_cycle55.pt) を baseline_videos_v3 8 動画で評価。
# baseline (= cnn_phase_b_large_v2.pt) との比較 anchor を取得。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle55_eval

mkdir -p data/verify/cycle55_eval
mkdir -p logs/cycle55_eval

CNN_MODEL="models/cnn_cycle55.pt"

VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)

# 順次実行 (= 並列は CPU 負荷上限超過時に viz が path 化けるリスク)
for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/cycle55_eval/viz_${key}.jsonl"
  report="data/verify/cycle55_eval/${key}.json"

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
    --output "data/verify/cycle55_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle55_eval/viz_${key}.log" 2>&1

  if [ ! -f "$board_log" ]; then
    echo "[fail] viz did not produce board_log for $key"
    continue
  fi

  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    > "logs/cycle55_eval/eval_${key}.log" 2>&1

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
    p = Path(f'data/verify/cycle55_eval/{v}.json')
    if not p.exists():
        continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    per_video[v] = {'critical': s.get('critical', 0), 'warning': s.get('warning', 0), 'verdict': d.get('verdict')}
    totals['critical'] += s.get('critical', 0)
    totals['warning'] += s.get('warning', 0)
out = {'totals': totals, 'per_video': per_video}
Path('data/verify/cycle55_eval/_summary.json').write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8'
)
# baseline 比較
baseline_p = Path('data/verify/baseline_v3_eval/_summary.json')
if baseline_p.exists():
    base = json.loads(baseline_p.read_text(encoding='utf-8'))
    cmp = {
        'baseline_critical': base['totals']['critical'],
        'cycle55_critical': totals['critical'],
        'diff_critical': totals['critical'] - base['totals']['critical'],
        'pct_change': round((totals['critical'] - base['totals']['critical']) / max(1, base['totals']['critical']) * 100, 1),
        'per_video_comparison': {},
    }
    for v in videos:
        b = base.get('per_video', {}).get(v, {})
        c = per_video.get(v, {})
        cmp['per_video_comparison'][v] = {
            'baseline_critical': b.get('critical', 0),
            'cycle55_critical': c.get('critical', 0),
            'diff': c.get('critical', 0) - b.get('critical', 0),
            'baseline_verdict': b.get('verdict'),
            'cycle55_verdict': c.get('verdict'),
        }
    Path('data/verify/cycle55_eval/_comparison.json').write_text(
        json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(cmp, indent=2, ensure_ascii=False))
else:
    print(json.dumps(out, indent=2, ensure_ascii=False))
"

finalize_health 0
