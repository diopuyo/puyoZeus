#!/bin/bash
# 全フレーム基準データ収集 (2026-07-30) の実行本体。
#
# 前提: scripts/_gen_jobs_allframes_ref_2026-07-30.sh 実行済で
#   scripts/_jobs_allframes_ref_2026-07-30.txt (330ジョブ) が生成されていること。
#
# 手順:
#   1. ジョブが参照する動画を data/frames (9p) から $HOME/frames (ext4) へ
#      未コピー分のみコピー (9p I/O ボトルネック回避、既存 c20/m20/m30 収集の慣例踏襲)
#   2. scripts/_run_safe.sh で MAXPAR=8, COOLDOWN=5s, THREADS=1 で消化
#   3. 完了マーカーをログに書き込む (fail-silent防止)
#
# 2026-07-30 追記 (MAXPAR 14→8 に変更、実測による訂正):
#   MAXPAR=14 は project_collect_indicators_v2_perf_2026-07-20 の実績最適値だが、
#   これは sample-interval=0.2 (間引きあり、軽量) の指標収集での calibration。
#   本ジョブは --sample-interval 0 (全フレーム) + --enable-chain-tracker +
#   --with-next で1ジョブあたりのCPU負荷が大幅に重く、実機は
#   `grep "cpu cores" /proc/cpuinfo` = 8 (13th Gen i7-13620H、16スレッドは
#   ハイパースレッディング) と判明。MAXPAR=14 (+他プロセス2本で実質16並列) で
#   実測した所要時間は 37-41秒/動画秒 (4動画実測平均) となり、1-2並列で解いた
#   場合の 11-15秒/動画秒 の約3倍に悪化していた (物理コア数を超える
#   オーバーサブスクリプションが原因)。MAXPAR=8 (物理コア数一致) に落として
#   総スループット改善を狙う (未検証のため一部完走を見てから最終判断する)。
#
# 出力先: data/indicators_v2/boards_lean_allframes_ref_2026-07-30/
#   (既存の基準データセット boards_lean_fixed_regen_2026-07-28 等とは別ディレクトリ、
#    混ざらないことを明示的に保証する)
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1

# MAXPAR/JOBS は環境変数で上書き可能 (既定値は従来通り動く、後方互換維持)
MAXPAR="${ALLFRAMES_REF_MAXPAR:-8}"
JOBS="${ALLFRAMES_REF_JOBS:-scripts/_jobs_allframes_ref_2026-07-30.txt}"
JOBS_TOTAL_FILE="scripts/_jobs_allframes_ref_2026-07-30.txt"  # 完了率は常に全330本基準
LOG="logs/allframes_ref_2026-07-30.log"
OUT_NPZ_DIR="data/indicators_v2/boards_lean_allframes_ref_2026-07-30"
mkdir -p "${OUT_NPZ_DIR}" "$HOME/frames" logs

echo "[allframes-ref] 開始 $(date) MAXPAR=${MAXPAR} THREADS=1 JOBS=${JOBS} ジョブ数=$(wc -l < "${JOBS}")" >> "${LOG}"

# --- 1. ジョブが参照する動画のうち未コピー分だけ ext4 へコピー ---
# _jobs...txt の "--video $HOME/frames/video_cN.mp4" から video_cN.mp4 部分を機械的に抽出
grep -o 'video_c[0-9]\+\.mp4' "${JOBS}" | sort -u | while IFS= read -r fname; do
  if [ ! -f "$HOME/frames/${fname}" ]; then
    echo "[copy] ${fname} -> ext4" >> "${LOG}"
    cp "data/frames/${fname}" "$HOME/frames/${fname}"
  fi
done
echo "[allframes-ref] 動画コピー確認完了 $(date)" >> "${LOG}"

# --- 2. ジョブを高並列(MAXPAR, THREADS=1, COOLDOWN=5)で消化 ---
bash scripts/_run_safe.sh "${JOBS}" "${MAXPAR}" 5 1 >> "${LOG}" 2>&1
echo "[allframes-ref] 収集ジョブ完了 $(date)" >> "${LOG}"

# --- 3. 完了確認 (期待 npz 数=全330本と実数を突き合わせ、fail-silent防止) ---
EXPECTED=$(wc -l < "${JOBS_TOTAL_FILE}")
ACTUAL=$(ls "${OUT_NPZ_DIR}"/*.npz 2>/dev/null | wc -l)
echo "[allframes-ref] npz出力確認 ${ACTUAL}/${EXPECTED}" >> "${LOG}"
echo "[allframes-ref] ALL_DONE $(date +%s) actual=${ACTUAL} expected=${EXPECTED}" >> "${LOG}"
