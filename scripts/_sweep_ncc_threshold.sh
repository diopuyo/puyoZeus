#!/bin/bash
# PATCH_NCC_EMPTY_THRESHOLD sweep スクリプト (2026-05-28)
# 目的: 案 d の NCC 閾値 sweet spot 探索。
# v95m15 で 3 秒幻ぷよ確認 (user 目視) のため、 0.92 → [0.85, 0.88, 0.90, 0.92] を評価。
#
# 対象動画 3 本:
#   v40m7  (真因動画)
#   v95m15 (退行動画、 3 秒幻ぷよ確認済)
#   v89m7  (改善動画 = -65)
# 出力: data/verify/ncc_sweep/<thresh>/<video>.json
# 並列上限 3 (= CLAUDE.md ルール)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health ncc_sweep

# ============================
# 定数
# ============================
CNN_MODEL="models/cnn_phase_l.pt"
MAX_PARALLEL=3
# sweep 候補 (小数点をファイル名に使えるよう _ に置換してディレクトリ名にする)
THRESHOLDS=("0.85" "0.88" "0.90" "0.92")

# 動画パス (key|path)
declare -A VIDEO_PATHS
VIDEO_PATHS["v40m7"]="data/baseline_videos_v3/v40m7_buf15s.mp4"
VIDEO_PATHS["v95m15"]="data/baseline_videos_v3/v95m15_buf15s.mp4"
VIDEO_PATHS["v89m7"]="data/phase_l/cut/v89m7_buf15s.mp4"
VIDEO_KEYS=("v40m7" "v95m15" "v89m7")

mkdir -p logs/ncc_sweep

# ============================
# 1 動画分の eval を実行する関数 (viz mp4 は /tmp に廃棄してディスク節約)
# ============================
run_one() {
  local thresh="$1"
  local key="$2"
  local input="${VIDEO_PATHS[$key]}"
  local thresh_dir="data/verify/ncc_sweep/${thresh}"
  local report="${thresh_dir}/${key}.json"
  local board_log="logs/ncc_sweep/board_${thresh}_${key}.jsonl"
  local log_file="logs/ncc_sweep/${thresh}_${key}.log"
  local tmp_mp4="/tmp/sweep_${thresh}_${key}.mp4"

  if [ -f "$report" ]; then
    echo "[skip] thresh=${thresh} ${key} (report exists)"
    return
  fi
  if [ ! -f "$input" ]; then
    echo "[skip] thresh=${thresh} ${key} (input missing: ${input})"
    return
  fi

  mkdir -p "$thresh_dir"
  echo "=== [thresh=${thresh} ${key}] viz @ $(date) ===" | tee -a "$log_file"

  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "$tmp_mp4" \
    --cnn-model "$CNN_MODEL" \
    --dump-board-log "$board_log" \
    --patch-ncc-threshold "$thresh" \
    >> "$log_file" 2>&1

  if [ ! -f "$board_log" ]; then
    echo "[fail viz] thresh=${thresh} ${key} (board_log not generated)" | tee -a "$log_file"
    return
  fi
  # 廃棄 mp4 を削除してディスク節約
  rm -f "$tmp_mp4"

  echo "=== [thresh=${thresh} ${key}] eval @ $(date) ===" | tee -a "$log_file"
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" \
    --report-out "$report" \
    >> "$log_file" 2>&1

  echo "[done] thresh=${thresh} ${key} @ $(date)" | tee -a "$log_file"
}

# ============================
# 全組み合わせを並列制御しながら実行
# ============================
pids=()
running=0

for thresh in "${THRESHOLDS[@]}"; do
  for key in "${VIDEO_KEYS[@]}"; do
    run_one "$thresh" "$key" &
    pids+=($!)
    ((running++)) || true
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
      ((running--)) || true
    fi
  done
done
wait

echo "=== all viz+eval done @ $(date) ===" | tee -a logs/ncc_sweep_master.log

# ============================
# 集計スクリプト
# ============================
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path

THRESHOLDS = ['0.85', '0.88', '0.90', '0.92']
VIDEO_KEYS = ['v40m7', 'v95m15', 'v89m7']

# 参照: patch_fp_eval (= NCC 0.92 で評価済み結果がある動画)
PATCH_FP_EVAL_DIR = Path('data/verify/patch_fp_eval')
# axis3b_revert_eval (= 直近 baseline)
AXIS3B_DIR = Path('data/verify/axis3b_revert_eval')

def load_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def get_critical(d: dict | None) -> int | None:
    if d is None:
        return None
    return d.get('summary', {}).get('critical', None)

# baseline として patch_fp_eval の各動画の critical を参照 (= NCC=0.92 既存評価)
baseline_per = {}
for key in VIDEO_KEYS:
    d = load_json(PATCH_FP_EVAL_DIR / f'{key}.json')
    baseline_per[key] = get_critical(d)

# sweep 結果集計
sweep = {}
for thresh in THRESHOLDS:
    thresh_dir = Path(f'data/verify/ncc_sweep/{thresh}')
    per = {}
    total = 0
    for key in VIDEO_KEYS:
        d = load_json(thresh_dir / f'{key}.json')
        c = get_critical(d)
        per[key] = c
        if c is not None:
            total += c
    sweep[thresh] = {'per_video': per, 'total': total}

# diff vs baseline (NCC=0.92 patch_fp_eval)
baseline_total = sum(
    v for v in baseline_per.values() if v is not None
)

rows = []
for thresh in THRESHOLDS:
    s = sweep[thresh]
    diff = s['total'] - baseline_total if baseline_total else None
    rows.append({
        'thresh': thresh,
        'total': s['total'],
        'diff_vs_baseline_0_92': diff,
        'per_video': s['per_video'],
    })

summary = {
    'baseline_patch_fp_eval_ncc_0_92': baseline_per,
    'baseline_total': baseline_total,
    'sweep_rows': rows,
    'note': 'NCC sweep: [0.85, 0.88, 0.90, 0.92] x [v40m7, v95m15, v89m7]',
}

out = Path('data/verify/ncc_sweep/_sweep_summary.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
" 2>&1 | tee logs/ncc_sweep/_summary.log

finalize_health 0
