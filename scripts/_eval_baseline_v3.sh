#!/bin/bash
# baseline_videos_v3 8 動画で current default モデルの強化アナリスト評価を取得。
# 自律 cycle で改善幅算出の anchor になる。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health baseline_v3_eval

mkdir -p data/verify/baseline_v3_eval
mkdir -p logs/baseline_v3_eval

DEFAULT_MODEL="models/cnn_phase_b_large_v2.pt"

VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)

# 順次実行 (= 並列は CPU 負荷上限超過時に viz が path 化けるリスク)
for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/baseline_v3_eval/viz_${key}.jsonl"
  report="data/verify/baseline_v3_eval/${key}.json"

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
    --output "data/verify/baseline_v3_eval/${key}.mp4" \
    --cnn-model "$DEFAULT_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/baseline_v3_eval/viz_${key}.log" 2>&1

  if [ ! -f "$board_log" ]; then
    echo "[fail] viz did not produce board_log for $key"
    continue
  fi

  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    > "logs/baseline_v3_eval/eval_${key}.log" 2>&1

  echo "[done] $key"
done

# critical 集計
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
videos = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
totals = {'critical': 0, 'warning': 0}
per_video = {}
for v in videos:
    p = Path(f'data/verify/baseline_v3_eval/{v}.json')
    if not p.exists():
        continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    per_video[v] = {'critical': s.get('critical', 0), 'warning': s.get('warning', 0), 'verdict': d.get('verdict')}
    totals['critical'] += s.get('critical', 0)
    totals['warning'] += s.get('warning', 0)
out = {'totals': totals, 'per_video': per_video}
Path('data/verify/baseline_v3_eval/_summary.json').write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8'
)
print(json.dumps(out, indent=2, ensure_ascii=False))
"

finalize_health 0
