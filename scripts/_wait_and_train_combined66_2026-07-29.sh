#!/bin/bash
# #43段階3: m30収集(マスター級残り26本)の完走を待って combined66(m30+m20+c20)
# 学習・評価を自動実行する (2026-07-29)。
# 雛形: scripts/_wait_and_finalize_2026-07-28.sh (実際に機能した完走待ちポーリング形式)
#
# 完走検知は以下いずれかを満たしたときとする(user指定の2通り):
#   (a) logs/orchestrate_labeled_win_m30_2026-07-28.log に "[m30] ALL DONE" が出現
#   (b) study/ 配下のCSVが78件そろい、かつ収集プロセス(_collect_1t)が0本
#
# fail-silent 防止: labeled_win_m30.csv の存在・最低行数を検証してから学習起動。
# 検証に失敗した場合は学習を起動せず、ログにFATALを残して終了する。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

M30_LOG="logs/orchestrate_labeled_win_m30_2026-07-28.log"
M30_STUDY_DIR="data/verify/labeled_win_m30_2026-07-28/study"
M30_CSV="data/verify/labeled_win_m30_2026-07-28/labeled_win_m30.csv"
WATCHER_LOG="logs/train_combined66_2026-07-29.log"

# m30ジョブ生成時点(_gen_jobs_labeled_win_m30_2026-07-28.sh)の総ジョブ数(26動画x3窓)
EXPECTED_STUDY_CSV=78
# labeled_win_m30.csv の最低行数の床(0行/ヘッダのみの空データを弾く目的、
# 実績値(m20: 58547行/20本)よりずっと低い安全側の床)
MIN_ROWS=100

mkdir -p logs

echo "[wait] $(date) m30完走待機開始 (PID=$$)" >> "${WATCHER_LOG}"

while true; do
  if grep -q "\[m30\] ALL DONE" "${M30_LOG}" 2>/dev/null; then
    echo "[wait] $(date) シグナル(a) ログ完了マーカー検知" >> "${WATCHER_LOG}"
    break
  fi

  n_csv=$(ls "${M30_STUDY_DIR}"/*.csv 2>/dev/null | wc -l)
  n_proc=$(pgrep -c -f _collect_1t 2>/dev/null)
  n_proc=${n_proc:-0}
  if [ "${n_csv}" -ge "${EXPECTED_STUDY_CSV}" ] && [ "${n_proc}" -eq 0 ]; then
    echo "[wait] $(date) シグナル(b) study CSV ${n_csv}/${EXPECTED_STUDY_CSV} 件 + 収集プロセス終了 検知" >> "${WATCHER_LOG}"
    break
  fi

  sleep 60
done

echo "[wait] $(date) 完走検知 -> labeled_win_m30.csv 検証開始" >> "${WATCHER_LOG}"

# --- fail-silent防止: labeled_win_m30.csv の存在・行数を検証してから学習起動 ---
if [ ! -f "${M30_CSV}" ]; then
  echo "[FATAL] $(date) ${M30_CSV} が存在しない。ラベル結合(label_win_from_winners)が" \
       "失敗した可能性。学習を起動せず停止する。" >> "${WATCHER_LOG}"
  exit 1
fi
n_rows=$(wc -l < "${M30_CSV}")
if [ "${n_rows}" -lt "${MIN_ROWS}" ]; then
  echo "[FATAL] $(date) ${M30_CSV} の行数(${n_rows})が最低基準(${MIN_ROWS})未満。" \
       "空/破損データの疑い。学習を起動せず停止する。" >> "${WATCHER_LOG}"
  exit 1
fi
echo "[wait] $(date) ${M30_CSV} 検証OK (${n_rows} 行、ヘッダ込み)。combined66学習を起動する。" >> "${WATCHER_LOG}"

# --- 学習・評価一式起動 (内部の _build_labeled_win_combined66_2026-07-29.py で
#     スキーマ一致・動画あたり行数の厳密検証を再度行う。二重ガード) ---
nice -n 10 bash scripts/_run_win_eval_combined66_2026-07-29.sh >> "${WATCHER_LOG}" 2>&1
RC=$?
if [ ${RC} -ne 0 ]; then
  echo "[FATAL] $(date) combined66学習・評価パイプラインが失敗(exit=${RC})。" \
       "詳細は data/verify/win_eval_combined66_2026-07-29/*.log を確認。" >> "${WATCHER_LOG}"
  exit 1
fi

echo "[done] $(date) combined66完了。summary.md: data/verify/win_eval_combined66_2026-07-29/summary.md" >> "${WATCHER_LOG}"
