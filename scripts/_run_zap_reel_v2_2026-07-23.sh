#!/bin/bash
# 認識レビュー用ザッピング動画 v2 (2026-07-23): 20-30本規模に拡大。
# c20/c40 は既にレンダー済(v1で完了)のためコピー再利用、残り26本を2波で並列レンダー。
set -u
PROJ="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "$PROJ" || exit 1

RAW_DIR="data/indicators_v2/overlay/zap/raw"
LAB_DIR="data/indicators_v2/overlay/zap/labeled"
LOG_DIR="logs/zap_reel"
mkdir -p "$RAW_DIR" "$LAB_DIR" "$LOG_DIR"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "[ffmpeg] $FF"

run_job() {
  local num="$1" vid="$2" tier="$3" card="$4" start="$5" end="$6" warmup="$7"
  local raw="$RAW_DIR/${num}_${vid}.mp4"
  local lab="$LAB_DIR/${num}_${vid}_labeled.mp4"
  local h264="$LAB_DIR/${num}_${vid}_h264.mp4"
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
}

# 波1: 元11本のうち再送9本(既知の良ウィンドウ) + 新規4本
WAVE1=(
  "01|video_c5|チャレンジャー|Shiyota vs やまだ|1019|1044|15"
  "03|video_c12|チャレンジャー|ゆがるむ vs やまだ|1574|1599|15"
  "08|video_c28|チャレンジャー|SAKI vs ともくん|1009|1034|15"
  "14|video_c65|マスター|MGR vs ゆが|1828|1853|15"
  "18|video_31|マスター|まじぇす vs レイン|660|685|15"
  "21|video_37|マスター|やまゆう vs 3|2119|2144|15"
  "22|video_c82|S級|レイン vs ゆうき|871|896|15"
  "23|video_c83|S級|ゆが vs レイン|1371|1396|15"
  "24|video_c84|S級|ゆが vs ゆうき|2140|2165|15"
  "02|video_c8|チャレンジャー|delta vs かき|987|1002|10"
  "04|video_c15|チャレンジャー|live vs やまだ|1542|1557|10"
  "05|video_c17|チャレンジャー|ぬえ vs ともくん|1720|1735|10"
  "07|video_c23|チャレンジャー|delta vs やまだ|1214|1229|10"
)
# 波2: 新規13本(マスター残り + S級混在(A/B1〜C1/C2)下位ティア)
WAVE2=(
  "09|video_c31|チャレンジャー|live vs かき|2165|2180|10"
  "11|video_c45|マスター|:o vs MGR|1275|1290|10"
  "12|video_c50|マスター|Ponderion vs reoru|1561|1576|10"
  "13|video_c58|マスター|SAKI vs スラさん|691|706|10"
  "15|video_c70|マスター|:o vs ゆが|1690|1705|10"
  "16|video_c78|マスター|reoru vs syakegohan|1409|1424|10"
  "17|video_c95|マスター進出決定|3 vs にゃんきち|2059|2074|10"
  "19|video_33|マスター|hov vs むー|844|859|10"
  "20|video_36|マスター|カキ様 vs とりいぬ|2060|2075|10"
  "25|video_c85|A・B1級|jig vs いつき|975|990|10"
  "26|video_c86|B2・C1級|くろy vs nuirapa|1129|1144|10"
  "27|video_c89|B1・B2級|むるむる vs 翔太|1594|1609|10"
  "28|video_c92|C1・C2級|くろけろ vs Ion|756|771|10"
)

echo "[wave1 start] $(date)"
for job in "${WAVE1[@]}"; do
  IFS='|' read -r num vid tier card start end warmup <<< "$job"
  run_job "$num" "$vid" "$tier" "$card" "$start" "$end" "$warmup" &
done
wait
echo "[wave1 done] $(date)"

echo "[wave2 start] $(date)"
for job in "${WAVE2[@]}"; do
  IFS='|' read -r num vid tier card start end warmup <<< "$job"
  run_job "$num" "$vid" "$tier" "$card" "$start" "$end" "$warmup" &
done
wait
echo "[wave2 done] $(date)"

# 既にv1で完成済の2本(元03/05)を新番号(06,10)へコピー
cp "$LAB_DIR/03_video_c20_h264.mp4" "$LAB_DIR/06_video_c20_h264.mp4"
cp "$LAB_DIR/05_video_c40_h264.mp4" "$LAB_DIR/10_video_c40_h264.mp4"

# concat用リスト (番号順、tierグループ順)
ORDER=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28)
VIDNAME_BY_NUM() {
  case "$1" in
    01) echo video_c5;; 02) echo video_c8;; 03) echo video_c12;; 04) echo video_c15;;
    05) echo video_c17;; 06) echo video_c20;; 07) echo video_c23;; 08) echo video_c28;;
    09) echo video_c31;; 10) echo video_c40;; 11) echo video_c45;; 12) echo video_c50;;
    13) echo video_c58;; 14) echo video_c65;; 15) echo video_c70;; 16) echo video_c78;;
    17) echo video_c95;; 18) echo video_31;; 19) echo video_33;; 20) echo video_36;;
    21) echo video_37;; 22) echo video_c82;; 23) echo video_c83;; 24) echo video_c84;;
    25) echo video_c85;; 26) echo video_c86;; 27) echo video_c89;; 28) echo video_c92;;
  esac
}

CONCAT_LIST="$LAB_DIR/_concat_list_v2.txt"
> "$CONCAT_LIST"
for num in "${ORDER[@]}"; do
  vid=$(VIDNAME_BY_NUM "$num")
  echo "file '$(pwd)/$LAB_DIR/${num}_${vid}_h264.mp4'" >> "$CONCAT_LIST"
done

FINAL="data/indicators_v2/overlay/zap/recognition_zap_reel_v2_2026-07-23.mp4"
echo "[concat] $(date)"
"$FF" -y -f concat -safe 0 -i "$CONCAT_LIST" -c copy "$FINAL" \
  > "$LOG_DIR/_concat_v2.log" 2>&1
echo "[concat done] $(date)"
ls -la "$FINAL"
echo "[ALL DONE V2] $(date)"
