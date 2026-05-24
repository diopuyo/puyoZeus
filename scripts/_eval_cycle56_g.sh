#!/bin/bash
# cycle 56 G: 小手先 (= state machine 定数 N=5→8、 votes=3→5) + KC metric (= 静止中色ブレ) 効果確認.
# c56_v3b model + 定数変更を反映した viz を 3 動画 + 8 動画 eval で生成。
# 比較対象: c56_v3b 単独 (= 定数 N=5 時代の eval) vs cycle 56 G (= 定数拡張)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle56_g_eval

mkdir -p data/verify/cycle56_g_viz
mkdir -p data/verify/cycle56_g_eval
mkdir -p logs/cycle56_g_eval

CNN_MODEL="models/cnn_cycle56_v3b.pt"

# Phase 1: ユーザー目視 3 動画 viz (= v89m7 + v30_match11 + v97_match11)
declare -A USER_VIDEOS=(
  [v89m7]="data/phase_l/cut/v89m7_buf15s.mp4"
  [v30_match11]="data/holdout_videos/v30_match11_buf15s.mp4"
  [v97_match11]="data/holdout_videos/v97_match11_buf15s.mp4"
)
for key in "${!USER_VIDEOS[@]}"; do
  input="${USER_VIDEOS[$key]}"
  output="data/verify/cycle56_g_viz/${key}_cycle56_g.mp4"
  board_log="logs/cycle56_g_eval/viz_${key}.jsonl"
  report="data/verify/cycle56_g_viz/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] G viz @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle56_g_eval/viz_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/cycle56_g_eval/eval_${key}.log" 2>&1
  echo "[done viz] $key"
done

# Phase 2: 8 動画 critical eval
VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)
for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/cycle56_g_eval/eval_${key}.jsonl"
  report="data/verify/cycle56_g_eval/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] G eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "data/verify/cycle56_g_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle56_g_eval/viz8_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/cycle56_g_eval/eval8_${key}.log" 2>&1
  echo "[done eval] $key"
done

# Phase 3: 集計
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

# 8 動画 critical 比較 (= baseline_v3 vs c56_v3b vs c56_g)
videos8 = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
g_totals = {'critical': 0, 'warning': 0, 'flicker': 0}
g_per_video = {}
for v in videos8:
    p = Path(f'data/verify/cycle56_g_eval/{v}.json')
    if not p.exists(): continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    by_m = s.get('by_metric', {})
    flicker_critical = 0
    flicker_total = 0
    for viol in d.get('violations', []):
        if viol.get('metric') == 'static_color_flicker':
            flicker_total += viol.get('extra', {}).get('total_flips', 0)
    g_per_video[v] = {
        'critical': s.get('critical', 0),
        'warning': s.get('warning', 0),
        'flicker_total': flicker_total,
        'verdict': d.get('verdict'),
    }
    g_totals['critical'] += s.get('critical', 0)
    g_totals['warning'] += s.get('warning', 0)
    g_totals['flicker'] += flicker_total

# 比較
baseline = json.loads(Path('data/verify/baseline_v3_eval/_summary.json').read_text(encoding='utf-8'))
v3b = json.loads(Path('data/verify/cycle56_v3b_eval/_summary.json').read_text(encoding='utf-8'))
cmp = {
    'baseline_critical': baseline['totals']['critical'],
    'c56_v3b_critical': v3b['totals']['critical'],
    'c56_g_critical': g_totals['critical'],
    'c56_g_flicker_total': g_totals['flicker'],
    'diff_vs_baseline': g_totals['critical'] - baseline['totals']['critical'],
    'diff_vs_v3b': g_totals['critical'] - v3b['totals']['critical'],
    'pct_vs_baseline': round((g_totals['critical'] - baseline['totals']['critical']) / max(1, baseline['totals']['critical']) * 100, 1),
    'per_video': g_per_video,
}
Path('data/verify/cycle56_g_eval/_comparison.json').write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))

# 3 動画 viz 集計 (= flicker 数 + critical)
user_videos = ['v89m7', 'v30_match11', 'v97_match11']
viz_summary = {}
for v in user_videos:
    p = Path(f'data/verify/cycle56_g_viz/{v}.json')
    if not p.exists(): continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    flicker_total = 0
    pair_counts = {}
    for viol in d.get('violations', []):
        if viol.get('metric') == 'static_color_flicker':
            flicker_total += viol.get('extra', {}).get('total_flips', 0)
            pcs = viol.get('extra', {}).get('pair_counts', {})
            for k, n in pcs.items():
                pair_counts[k] = pair_counts.get(k, 0) + n
    viz_summary[v] = {
        'critical': s.get('critical', 0),
        'warning': s.get('warning', 0),
        'flicker_total': flicker_total,
        'pair_counts_top5': dict(sorted(pair_counts.items(), key=lambda kv: -kv[1])[:5]),
    }
Path('data/verify/cycle56_g_viz/_summary.json').write_text(json.dumps(viz_summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('=== 3 動画 viz ===')
print(json.dumps(viz_summary, indent=2, ensure_ascii=False))
"

finalize_health 0
