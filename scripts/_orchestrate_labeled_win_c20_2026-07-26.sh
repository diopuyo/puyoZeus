#!/bin/bash
# #43 本番第1弾(段階1): c系20本の labeled_win 収集を自動チェインで実行する。
# 1. winners_panel_diff_2026-07-26/*.json の完了を監視しつつ品質ゲート+選定を再実行
# 2. ティア確認済み・正常20本が揃ったら (または上限時間に達したら) 選定を確定
# 3. 指標収集ジョブ生成 -> scripts/_run_safe.sh で熱対策済み実行 (既定 MAXPAR=3)
# 4. 完了後、label_win_from_winners.py でラベル結合し labeled_win_c20.csv を確定
#
# setsid -f 経由で detach 起動する前提 (CLAUDE.md プロセス管理ルール)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

TARGET_N="${1:-20}"
# 選定待ちの上限 (秒)。これを超えたら現時点の本数で妥協して先へ進む。
MAX_WAIT_SEC="${2:-14400}"   # 既定4時間
POLL_INTERVAL_SEC=180
# 収集ジョブの並列数 (既定3: winners_panel_diff_all93 抽出ジョブとの熱共存を優先。
# 抽出完了後に手動で6へ引き上げ可、_run_safe.sh の第2引数で調整可能)
COLLECT_MAXPAR="${3:-3}"

OUT_DIR="data/verify/labeled_win_c20_2026-07-26"
LOG="logs/orchestrate_labeled_win_c20_2026-07-26.log"
mkdir -p "${OUT_DIR}" logs

echo "[orchestrate] 開始 $(date)" >> "${LOG}"

# --- 1. 選定が揃うまで監視ループ ---
start_ts=$(date +%s)
selected_count=0
while true; do
  ./venv/bin/python -m scripts.build_labeled_win_quality_gate >> "${LOG}" 2>&1
  ./venv/bin/python -m scripts.select_labeled_win_videos --n "${TARGET_N}" >> "${LOG}" 2>&1
  selected_count=$(wc -l < "${OUT_DIR}/selected_videos.txt" 2>/dev/null || echo 0)
  now=$(date +%s)
  elapsed=$((now - start_ts))
  echo "[orchestrate $(date '+%H:%M:%S')] 選定済み=${selected_count}/${TARGET_N} 経過=${elapsed}s" >> "${LOG}"
  if [ "${selected_count}" -ge "${TARGET_N}" ]; then
    echo "[orchestrate] 目標本数到達、選定確定" >> "${LOG}"
    break
  fi
  if [ "${elapsed}" -ge "${MAX_WAIT_SEC}" ]; then
    echo "[orchestrate][WARN] 上限時間到達、現本数(${selected_count})で妥協して続行" >> "${LOG}"
    break
  fi
  if grep -q "\[all93\] ALL DONE" logs/winners_panel_diff_all93_2026-07-26.log 2>/dev/null; then
    echo "[orchestrate] all93抽出が完了済み、これ以上増えないため現本数で確定" >> "${LOG}"
    break
  fi
  sleep "${POLL_INTERVAL_SEC}"
done

# --- 2. ジョブ生成 ---
bash scripts/_gen_jobs_labeled_win_c20_2026-07-26.sh "${OUT_DIR}/selected_videos.txt" >> "${LOG}" 2>&1

# --- 3. 熱対策済み実行 (完了までブロック) ---
echo "[orchestrate] 収集ジョブ開始 $(date) MAXPAR=${COLLECT_MAXPAR}" >> "${LOG}"
bash scripts/_run_safe.sh scripts/_jobs_labeled_win_c20_2026-07-26.txt "${COLLECT_MAXPAR}" 45 2 >> "${LOG}" 2>&1
echo "[orchestrate] 収集ジョブ完了 $(date)" >> "${LOG}"

# --- 4. ラベル結合 ---
./venv/bin/python -m scripts.label_win_from_winners \
  --study "${OUT_DIR}/study" \
  --winners-dir data/verify/winners_panel_diff_gated_2026-07-26 \
  --out "${OUT_DIR}/labeled_win_c20.csv" >> "${LOG}" 2>&1

echo "[orchestrate] ALL DONE $(date)" >> "${LOG}"
