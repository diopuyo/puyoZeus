#!/bin/bash
# 打ち合い時間モデル確定のための4動画(c5,c30,c31,c83)指標収集ジョブ生成 (2026-07-29)。
#
# 背景 (userタスク): data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv の
# 着弾検出イベント25件中、study CSV 未収集のため検証に使えなかった11件
# (c30:3件 / c31:1件 / c5:4件 / c83:3件) を回収する。
#
# 窓は m30/m20 と同一の3窓 (base=0-300s / gap=300-900s /
# mid=1200-1560s, --start-sec 1200 --max-sec 360) を踏襲する。ただし
# 対象11件の t_landed を全件確認した結果、
#   c5  game13,1P: t_chain_start=1072.4s, t_landed=1079.6s
#   c30 game0, 2P: t_chain_start=1168.6s, t_landed=1171.8s
# の2件が既存3窓の谷間 (900-1200s) に落ちる (mid窓は1200sからのため
# 1171.8s等をカバーしない)。この2動画のみ gap2 窓 (900-1200s, 300秒) を
# 追加する。c31(該当イベントは426.8sでgap窓内)・c83(全3件がbase/gap窓内)
# には該当イベントが無いため gap2 を追加しない (過剰収集回避)。
#
# フレーム間引きは使わない (--sample-interval-frames 渡さない)。
# 2026-07-29 revert 済み事案: 間引くとtsumo(手数)カウンタが
# 実ゲームプレイ区間で全数消失することが判明したため
# (scripts/_gen_jobs_labeled_win_m30_2026-07-28.sh 冒頭コメント参照)。
#
# 出力先は既存 labeled_win_m30/m20/c20 と衝突しない新規ディレクトリ
# (data/verify/labeled_win_extra4_2026-07-29/study/)。
#
# 使い方: bash scripts/_gen_jobs_labeled_win_extra4_2026-07-29.sh
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1

OUT_STUDY_DIR="data/verify/labeled_win_extra4_2026-07-29/study"
mkdir -p "${OUT_STUDY_DIR}"
mkdir -p "$HOME/frames" logs

JOBS="scripts/_jobs_labeled_win_extra4_2026-07-29.txt"
: > "${JOBS}"

# video_stem -> 900-1200s窓(gap2)が必要か (1=必要、該当着弾イベント有り)
declare -A NEED_GAP2=( [c5]=1 [c30]=1 [c31]=0 [c83]=0 )

for n in 5 30 31 83; do
  vid="c${n}"
  # 9p I/O ボトルネック回避のため ext4 ($HOME/frames) へコピー (未コピーの場合のみ)
  if [ ! -f "$HOME/frames/video_${vid}.mp4" ]; then
    echo "[copy] video_${vid}.mp4 -> ext4"
    cp "data/frames/video_${vid}.mp4" "$HOME/frames/"
  fi

  windows=("|--max-sec 300" "_gap|--start-sec 300 --max-sec 600" "_mid|--start-sec 1200 --max-sec 360")
  if [ "${NEED_GAP2[$vid]}" = "1" ]; then
    windows+=("_gap2|--start-sec 900 --max-sec 300")
  fi

  for suf_args in "${windows[@]}"; do
    suf="${suf_args%%|*}"
    args="${suf_args#*|}"
    echo "PYTHONPATH=. ./venv/bin/python -m scripts._collect_1t --video \$HOME/frames/video_${vid}.mp4 --out ${OUT_STUDY_DIR}/${vid}${suf}.csv ${args} > logs/labeled_win_extra4_${vid}${suf}_2026-07-29.log 2>&1"
  done >> "${JOBS}"
done

echo "[gen] ジョブ数: $(wc -l < "${JOBS}")"
