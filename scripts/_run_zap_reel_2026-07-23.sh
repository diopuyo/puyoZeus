#!/bin/bash
# 認識レビュー用ザッピング動画 (2026-07-23): チャレンジャー/マスター/S級から11本を
# 各~25秒切り出し + --show-recognition + ラベル焼き + h264 化 + 最終concat。
set -u
PROJ="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "$PROJ" || exit 1

RAW_DIR="data/indicators_v2/overlay/zap/raw"
LAB_DIR="data/indicators_v2/overlay/zap/labeled"
LOG_DIR="logs/zap_reel"
mkdir -p "$RAW_DIR" "$LAB_DIR" "$LOG_DIR"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "[ffmpeg] $FF"

# 各行: 番号 動画file ティア 対戦カード start end warmup
JOBS=(
  "01|video_c5|チャレンジャー|Shiyota vs やまだ|1019|1044|15"
  "02|video_c12|チャレンジャー|ゆがるむ vs やまだ|1574|1599|15"
  "03|video_c20|チャレンジャー|SAKI vs ヨダソウマ|1904|1929|15"
  "04|video_c28|チャレンジャー|SAKI vs ともくん|1009|1034|15"
  "05|video_c40|マスター|selva vs とりいぬ|988|1013|15"
  "06|video_c65|マスター|MGR vs ゆが|1828|1853|15"
  "07|video_31|マスター|まじぇす vs レイン|660|685|15"
  "08|video_37|マスター|やまゆう vs 3|2119|2144|15"
  "09|video_c82|S級|レイン vs ゆうき|871|896|15"
  "10|video_c83|S級|ゆが vs レイン|1371|1396|15"
  "11|video_c84|S級|ゆが vs ゆうき|2140|2165|15"
)

echo "[start render] $(date)"
for job in "${JOBS[@]}"; do
  IFS='|' read -r num vid tier card start end warmup <<< "$job"
  (
    raw="$RAW_DIR/${num}_${vid}.mp4"
    lab="$LAB_DIR/${num}_${vid}_labeled.mp4"
    h264="$LAB_DIR/${num}_${vid}_h264.mp4"
    echo "[job $num] $vid render start=$start end=$end $(date)"
    PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
      --video "data/frames/${vid}.mp4" --out "$raw" \
      --start-sec "$start" --end-sec "$end" --warmup-sec "$warmup" \
      --show-recognition > "$LOG_DIR/${num}_${vid}_render.log" 2>&1
    echo "[job $num] $vid render done $(date)"
    PYTHONPATH=. ./venv/bin/python -m scripts._zap_label_burn \
      --src "$raw" --dst "$lab" \
      --video-id "$vid" --tier "$tier" --matchup "$card" \
      > "$LOG_DIR/${num}_${vid}_label.log" 2>&1
    echo "[job $num] $vid label done $(date)"
    "$FF" -y -i "$lab" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
      -movflags +faststart "$h264" > "$LOG_DIR/${num}_${vid}_h264.log" 2>&1
    echo "[job $num] $vid h264 done $(date)"
  ) &
done
wait
echo "[all render done] $(date)"

# concat用リスト (番号順)
CONCAT_LIST="$LAB_DIR/_concat_list.txt"
> "$CONCAT_LIST"
for job in "${JOBS[@]}"; do
  IFS='|' read -r num vid tier card start end warmup <<< "$job"
  echo "file '$(pwd)/$LAB_DIR/${num}_${vid}_h264.mp4'" >> "$CONCAT_LIST"
done

FINAL="data/indicators_v2/overlay/zap/recognition_zap_reel_2026-07-23.mp4"
echo "[concat] $(date)"
"$FF" -y -f concat -safe 0 -i "$CONCAT_LIST" -c copy "$FINAL" \
  > "$LOG_DIR/_concat.log" 2>&1
echo "[concat done] $(date)"
ls -la "$FINAL"
echo "[ALL DONE] $(date)"
