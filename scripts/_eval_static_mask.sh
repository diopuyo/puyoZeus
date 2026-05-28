#!/bin/bash
# T4: StaticBoardMask AND ガード 評価スクリプト (2026-05-28)。
# pixel-level diff による背景判定 + fail-silent 自動検知 (PuyoErasureMonitor)
# を有効にした状態で 12 動画を評価する。
#
# 出力: data/verify/static_mask_eval/
# log:  logs/static_mask_eval/
# 比較:
#   baseline_v3_eval (critical 1512)
#   axis3b_revert_eval (直近 baseline)
#   patch_fp_eval (案 d 比較)
# 並列上限 3 (CLAUDE.md / CYCLE_FINDINGS.md ルール準拠)。
# p_to_e_count 増加があれば fail-silent 自動 REJECT。

cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health static_mask_eval

mkdir -p data/verify/static_mask_eval
mkdir -p logs/static_mask_eval

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
    > "logs/static_mask_eval/run_${key}.log" 2>&1
  if [ ! -f "$board_log" ]; then
    echo "[fail viz] $key (board_log not generated)"
    return
  fi
  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    >> "logs/static_mask_eval/run_${key}.log" 2>&1
  echo "[done] $key"
}

# タスク一覧 (= 12 動画、 axis3b_revert_eval + patch_fp_eval と同一セット)
TASKS=(
  "v89m7|data/phase_l/cut/v89m7_buf15s.mp4|data/verify/static_mask_eval/v89m7.mp4|data/verify/static_mask_eval/v89m7.json|logs/static_mask_eval/board_v89m7.jsonl"
  "v30_match11|data/holdout_videos/v30_match11_buf15s.mp4|data/verify/static_mask_eval/v30_match11.mp4|data/verify/static_mask_eval/v30_match11.json|logs/static_mask_eval/board_v30_match11.jsonl"
  "v30_5min|data/holdout_videos/v30_5min_90s.mp4|data/verify/static_mask_eval/v30_5min.mp4|data/verify/static_mask_eval/v30_5min.json|logs/static_mask_eval/board_v30_5min.jsonl"
  "v97_match11|data/holdout_videos/v97_match11_buf15s.mp4|data/verify/static_mask_eval/v97_match11.mp4|data/verify/static_mask_eval/v97_match11.json|logs/static_mask_eval/board_v97_match11.jsonl"
  "v29m2|data/baseline_videos_v3/v29m2_buf15s.mp4|data/verify/static_mask_eval/v29m2.mp4|data/verify/static_mask_eval/v29m2.json|logs/static_mask_eval/board_v29m2.jsonl"
  "v40m7|data/baseline_videos_v3/v40m7_buf15s.mp4|data/verify/static_mask_eval/v40m7.mp4|data/verify/static_mask_eval/v40m7.json|logs/static_mask_eval/board_v40m7.jsonl"
  "v51m2|data/baseline_videos_v3/v51m2_buf15s.mp4|data/verify/static_mask_eval/v51m2.mp4|data/verify/static_mask_eval/v51m2.json|logs/static_mask_eval/board_v51m2.jsonl"
  "v57m2|data/baseline_videos_v3/v57m2_buf15s.mp4|data/verify/static_mask_eval/v57m2.mp4|data/verify/static_mask_eval/v57m2.json|logs/static_mask_eval/board_v57m2.jsonl"
  "v70m2|data/baseline_videos_v3/v70m2_buf15s.mp4|data/verify/static_mask_eval/v70m2.mp4|data/verify/static_mask_eval/v70m2.json|logs/static_mask_eval/board_v70m2.jsonl"
  "v89m3|data/baseline_videos_v3/v89m3_buf15s.mp4|data/verify/static_mask_eval/v89m3.mp4|data/verify/static_mask_eval/v89m3.json|logs/static_mask_eval/board_v89m3.jsonl"
  "v95m15|data/baseline_videos_v3/v95m15_buf15s.mp4|data/verify/static_mask_eval/v95m15.mp4|data/verify/static_mask_eval/v95m15.json|logs/static_mask_eval/board_v95m15.jsonl"
  "v97m11|data/baseline_videos_v3/v97m11_buf15s.mp4|data/verify/static_mask_eval/v97m11.mp4|data/verify/static_mask_eval/v97m11.json|logs/static_mask_eval/board_v97m11.jsonl"
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

# 集計: baseline_v3 / axis3b_revert / patch_fp と 3 種比較 + p_to_e_count
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

BASELINE_V3_CRITICAL = 1512

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
p_to_e_total = 0

for v in all_videos:
    d = load(f'data/verify/static_mask_eval/{v}.json')
    if not d:
        per_video[v] = {'critical': None, 'flicker': None, 'status': 'MISSING'}
        continue
    s = d.get('summary', {})
    c = s.get('critical', 0)
    flicker = calc_flicker(d)
    # p_to_e_count: PuyoErasureMonitor が board_log に埋め込んだ値 (将来)
    p_to_e = d.get('p_to_e_count', 0)
    p_to_e_total += p_to_e
    per_video[v] = {
        'critical': c,
        'flicker': flicker,
        'p_to_e_count': p_to_e,
        'verdict': d.get('verdict'),
    }
    totals['critical'] += c
    totals['flicker'] += flicker

# 比較用: axis3b_revert / patch_fp の per-video critical
axis3b_per = {}
patch_fp_per = {}
for v in all_videos:
    d = load(f'data/verify/axis3b_revert_eval/{v}.json')
    if d:
        axis3b_per[v] = {'critical': d.get('summary', {}).get('critical', 0)}
    d2 = load(f'data/verify/patch_fp_eval/{v}.json')
    if d2:
        patch_fp_per[v] = {'critical': d2.get('summary', {}).get('critical', 0)}

axis3b_total = sum(v.get('critical', 0) for v in axis3b_per.values() if v.get('critical') is not None)
patch_fp_total = sum(v.get('critical', 0) for v in patch_fp_per.values() if v.get('critical') is not None)

# 退行 flag (= axis3b_revert 比 +20 以上悪化した動画)
regression_flags = []
for v, info in per_video.items():
    if info.get('critical') is None:
        continue
    b = axis3b_per.get(v, {}).get('critical')
    if b is not None and info['critical'] - b >= 20:
        regression_flags.append({
            'video': v,
            'current': info['critical'],
            'axis3b_revert': b,
            'diff': info['critical'] - b,
        })

# verdict 判定:
#   critical 改善 AND p_to_e_count 増加なし → ACCEPT
#   critical 改善 BUT p_to_e_count 増加あり → fail-silent 自動 REJECT
improved = totals['critical'] <= axis3b_total
p_to_e_clean = p_to_e_total == 0
if improved and p_to_e_clean:
    verdict = 'ACCEPT'
elif improved and not p_to_e_clean:
    verdict = 'REJECT_P_TO_E'  # fail-silent 自動 REJECT
else:
    verdict = 'REJECT'

cmp = {
    'static_mask_critical_total': totals['critical'],
    'baseline_v3_critical_total': BASELINE_V3_CRITICAL,
    'axis3b_revert_critical_total': axis3b_total,
    'patch_fp_critical_total': patch_fp_total,
    'diff_vs_baseline_v3': totals['critical'] - BASELINE_V3_CRITICAL,
    'diff_vs_axis3b_revert': totals['critical'] - axis3b_total,
    'diff_vs_patch_fp': totals['critical'] - patch_fp_total,
    'pct_vs_baseline_v3': round(
        (totals['critical'] - BASELINE_V3_CRITICAL) / max(1, BASELINE_V3_CRITICAL) * 100, 1
    ),
    'p_to_e_count_total': p_to_e_total,
    'p_to_e_per_video': {v: per_video[v].get('p_to_e_count', 0) for v in all_videos},
    'flicker_total': totals['flicker'],
    'per_video': per_video,
    'axis3b_per_video': axis3b_per,
    'regression_flags': regression_flags,
    'verdict': verdict,
    'note': 'T4 StaticBoardMask AND ガード + PuyoErasureMonitor fail-silent 検知',
}
out = Path('data/verify/static_mask_eval/_comparison.json')
out.write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(cmp, indent=2, ensure_ascii=False))
" 2>&1 | tee logs/static_mask_eval/_summary.log

finalize_health 0
