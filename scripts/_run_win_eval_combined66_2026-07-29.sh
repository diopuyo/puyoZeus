#!/bin/bash
# #43段階3: combined66(m30+m20+c20) 学習・評価一式 (2026-07-29)。
# m20評価プロトコル(data/verify/win_eval_m20_2026-07-28/, project_m20_eval_tier_settled_2026-07-28)
# を完全踏襲。既存スクリプトの再利用のみで新規学習ロジックは書かない。
#
# 呼び出し元: scripts/_wait_and_train_combined66_2026-07-29.sh
# (m30収集完走・labeled_win_m30.csv 検証OK後にのみ呼ばれる前提。単体実行時は
#  labeled_win_m30.csv の存在を事前に確認すること)
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

OUT_DIR="data/verify/win_eval_combined66_2026-07-29"
mkdir -p "${OUT_DIR}"

M30_CSV="data/verify/labeled_win_m30_2026-07-28/labeled_win_m30.csv"
M20_CSV="data/verify/labeled_win_m20_2026-07-28/labeled_win_m20.csv"
C20_CSV="data/verify/labeled_win_c20_2026-07-26/labeled_win_c20.csv"
COMBINED66_CSV="${OUT_DIR}/labeled_win_combined66.csv"

M30_STUDY="data/verify/labeled_win_m30_2026-07-28/study"
M20_STUDY="data/verify/labeled_win_m20_2026-07-28/study"
C20_STUDY="data/verify/labeled_win_c20_2026-07-26/study"

echo "[combined66] 開始 $(date)"

# --- 1. combined66 CSV結合 (fail-silent防止: スキーマ/行数検証込み) ---
echo "[combined66] Step1: CSV結合"
nice -n 10 ./venv/bin/python -m scripts._build_labeled_win_combined66_2026-07-29 \
  --sources "${M30_CSV}" "${M20_CSV}" "${C20_CSV}" \
  --out "${COMBINED66_CSV}"
if [ $? -ne 0 ]; then
  echo "[combined66] [FATAL] Step1(CSV結合)が失敗。以降のステップを中止する。" >&2
  exit 1
fi

# --- 2. 絶対手数境界での学習・評価(m20/combined40と同一境界=序盤<=18,中盤18-40,終盤>40) ---
echo "[combined66] Step2: 絶対境界 model_indicator_win"
nice -n 10 ./venv/bin/python -m scripts.model_indicator_win \
  --labeled "${COMBINED66_CSV}" \
  --fixed-q33 18 --fixed-q67 40 \
  --out "${OUT_DIR}/combined66_importance.csv" \
  > "${OUT_DIR}/combined66_run.log" 2>&1
if [ $? -ne 0 ]; then
  echo "[combined66] [FATAL] Step2(model_indicator_win)が失敗。ログ確認: ${OUT_DIR}/combined66_run.log" >&2
  exit 1
fi

# --- 3. 相対位相(セグメント内進行率、主指標)評価 ---
echo "[combined66] Step3: 相対位相評価"
nice -n 10 ./venv/bin/python -m scripts._tmp_relphase_win_auc_generic_2026-07-28 \
  --study-dir "${M30_STUDY}" "${M20_STUDY}" "${C20_STUDY}" \
  --labeled "${COMBINED66_CSV}" \
  --out-dir "${OUT_DIR}/relphase_combined66" \
  > "${OUT_DIR}/relphase_combined66_run.log" 2>&1
if [ $? -ne 0 ]; then
  echo "[combined66] [FATAL] Step3(相対位相評価)が失敗。ログ確認: ${OUT_DIR}/relphase_combined66_run.log" >&2
  exit 1
fi

# --- 4. video別 LOGO(Leave-One-Group-Out)内訳 ---
echo "[combined66] Step4: LOGO内訳"
nice -n 10 ./venv/bin/python -m scripts._tmp_video_phase_auc_breakdown \
  --labeled "${COMBINED66_CSV}" \
  --out-csv "${OUT_DIR}/combined66_video_breakdown_summary.csv" \
  > "${OUT_DIR}/combined66_video_breakdown.log" 2>&1
if [ $? -ne 0 ]; then
  echo "[combined66] [FATAL] Step4(LOGO内訳)が失敗。ログ確認: ${OUT_DIR}/combined66_video_breakdown.log" >&2
  exit 1
fi

# --- 5. 中盤限定 Permutation Importance ---
echo "[combined66] Step5: 中盤限定importance"
nice -n 10 ./venv/bin/python -m scripts._tmp_midphase_importance_2026-07-28 \
  --labeled "${COMBINED66_CSV}" \
  --fixed-q33 18 --fixed-q67 40 \
  --out "${OUT_DIR}/combined66_midphase_importance.csv" \
  > "${OUT_DIR}/combined66_midphase_importance_run.log" 2>&1
if [ $? -ne 0 ]; then
  echo "[combined66] [FATAL] Step5(中盤importance)が失敗。ログ確認: ${OUT_DIR}/combined66_midphase_importance_run.log" >&2
  exit 1
fi

# --- 6. summary.md 自動生成(v10/c20/m20/combined40と横並び比較) ---
echo "[combined66] Step6: summary.md生成"
nice -n 10 ./venv/bin/python -m scripts._gen_summary_combined66_2026-07-29 \
  --combined66-dir "${OUT_DIR}"
if [ $? -ne 0 ]; then
  echo "[combined66] [FATAL] Step6(summary.md生成)が失敗。" >&2
  exit 1
fi

echo "[combined66] 全ステップ完了 $(date)"
echo "[combined66] 出力: ${OUT_DIR}/summary.md"
