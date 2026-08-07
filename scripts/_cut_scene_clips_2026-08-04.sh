#!/bin/bash
# 指摘シーンクリップ4本の再切り出し (2026-08-04 連続クランプ反映後、clip_*_final上書き)。
# 各クリップの絶対区間 (match_0X.mp4内の相対秒 = 絶対t_sec - 試合区間開始):
#   scene1: match_01.mp4 相対(41.9,65.9)  [絶対2925.0-2949.0, 24.0秒、旧durationと一致]
#   scene2: match_02.mp4 相対(33.8,63.8)  [絶対2980.0-3010.0=公式5シーン窓、30.0秒、旧durationと一致]
#   scene4: match_04.mp4 相対(21.0,48.0)  [絶対3084.9-3111.9(試合終端)、27.0秒、旧durationと一致]
#   scene5: match_05.mp4 相対(44.7,66.7)  [絶対3150.0-3172.0、22.0秒、旧durationと一致]
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF="$(PYTHONPATH=. ./venv/bin/python -c "from src.video_compositer import VideoCompositor; print(VideoCompositor._resolve_ffmpeg_bin())")"
DIR="data/verify/advantage_videos_olRyxDGacbg_2026-08-03"

cut() {
  local src="$1" start="$2" dur="$3" out="$4"
  echo "[cut] ${src} ${start}s +${dur}s -> ${out}"
  "${FF}" -y -ss "${start}" -t "${dur}" -i "${DIR}/${src}" -c copy "${DIR}/${out}"
}

cut match_01.mp4 41.9 24.0 clip_scene1_kaeshi_kakutei_final.mp4
cut match_02.mp4 33.8 30.0 clip_scene2_35byou_horyu_final.mp4
cut match_04.mp4 21.0 27.0 clip_scene4_oburekaisho_final.mp4
cut match_05.mp4 44.7 22.0 clip_scene5_51_56byou_final.mp4
echo "[all done]"
ls -la "${DIR}"/clip_*_final.mp4
