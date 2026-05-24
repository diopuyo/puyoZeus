#!/bin/bash
# cycle 56_v3b (= KA2、 最終 2 層 baseline 復元) の baseline_v3 8 動画 eval.
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle56_v3b_eval

mkdir -p data/verify/cycle56_v3b_eval
mkdir -p logs/cycle56_v3b_eval

CNN_MODEL="models/cnn_cycle56_v3b.pt"
VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)

for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/cycle56_v3b_eval/viz_${key}.jsonl"
  report="data/verify/cycle56_v3b_eval/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] viz @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "data/verify/cycle56_v3b_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle56_v3b_eval/viz_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/cycle56_v3b_eval/eval_${key}.log" 2>&1
  echo "[done] $key"
done

PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
videos = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'warning': 0}
per_video = {}
for v in videos:
    p = Path(f'data/verify/cycle56_v3b_eval/{v}.json')
    if not p.exists(): continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    per_video[v] = {'critical': s.get('critical', 0), 'warning': s.get('warning', 0), 'verdict': d.get('verdict'),
                    'oj_disap': s.get('by_metric', {}).get('ojama_disappearance', 0),
                    'oj_scarce': s.get('by_metric', {}).get('ojama_global_scarcity', 0)}
    totals['critical'] += s.get('critical', 0)
    totals['warning'] += s.get('warning', 0)
out = {'totals': totals, 'per_video': per_video}
Path('data/verify/cycle56_v3b_eval/_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
base = json.loads(Path('data/verify/baseline_v3_eval/_summary.json').read_text(encoding='utf-8'))
cmp = {'baseline_critical': base['totals']['critical'], 'cycle56_v3b_critical': totals['critical'],
       'diff': totals['critical'] - base['totals']['critical'],
       'pct': round((totals['critical'] - base['totals']['critical']) / max(1, base['totals']['critical']) * 100, 1),
       'per_video': per_video}
Path('data/verify/cycle56_v3b_eval/_comparison.json').write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
"

finalize_health 0
