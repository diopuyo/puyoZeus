#!/bin/bash
# #43 段階3: マスター級残り26本 (m20未使用分 video_c56-c81) の指標収集ジョブ生成。
# video_c{n} を $HOME/frames (ext4) へコピーし、base/gap/mid の3窓ジョブを
# _jobs_labeled_win_m30_2026-07-28.txt に書き出す (scripts/_run_safe.sh で消化する形式)。
# 窓定義は m20 (2026-07-28) と完全同一 (踏襲):
#   base = 0-300s / gap = 300-900s (--start-sec 300 --max-sec 600) /
#   mid  = 1200-1560s (--start-sec 1200 --max-sec 360)
#
# 使い方: bash scripts/_gen_jobs_labeled_win_m30_2026-07-28.sh <selected_videos_m30.txt>
#
# 2026-07-29 追記: --sample-interval-frames 15 を一度追加したが、投入前検証
# (data/verify/sample_interval_verify_2026-07-29/) でtsumo (手数) カウンタが
# 実ゲームプレイ区間で 0/38 (1P) 0/40 (2P) と全数消失する致命的な取りこぼしを
# 実測したため revert 済み。原因は src/recognition_pipeline.py の tsumo_count
# がTSUMO_FALL→STABLE着地の一点検出(edge-triggered)であり、15フレーム間引き
# だとサンプル間で着地イベントごと丸ごと見逃され得ること、かつ
# chain-tsumo-undershoot クランプ(同ファイル5144-5156行)が未検出分を0に
# ゼロ化してしまうこと。ユーザー判断待ち(現状は間引き無しのまま維持)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1

VIDEO_LIST="${1:?selected_videos_m30.txt (1行1 video_id) が必要}"
OUT_STUDY_DIR="data/verify/labeled_win_m30_2026-07-28/study"
mkdir -p "${OUT_STUDY_DIR}"
mkdir -p "$HOME/frames" logs

JOBS="scripts/_jobs_labeled_win_m30_2026-07-28.txt"
: > "${JOBS}"

while IFS= read -r vid; do
  [ -z "$vid" ] && continue
  n="${vid#video_c}"
  # 9p I/O ボトルネック回避のため ext4 ($HOME/frames) へコピー (未コピーの場合のみ)
  if [ ! -f "$HOME/frames/video_c${n}.mp4" ]; then
    echo "[copy] video_c${n}.mp4 -> ext4"
    cp "data/frames/video_c${n}.mp4" "$HOME/frames/"
  fi
  for suf_args in "|--max-sec 300" "_gap|--start-sec 300 --max-sec 600" "_mid|--start-sec 1200 --max-sec 360"; do
    suf="${suf_args%%|*}"
    args="${suf_args#*|}"
    echo "PYTHONPATH=. ./venv/bin/python -m scripts._collect_1t --video \$HOME/frames/video_c${n}.mp4 --out ${OUT_STUDY_DIR}/c${n}${suf}.csv ${args} > logs/labeled_win_m30_c${n}${suf}_2026-07-28.log 2>&1"
  done >> "${JOBS}"
done < "${VIDEO_LIST}"

echo "[gen] ジョブ数: $(wc -l < "${JOBS}")"
