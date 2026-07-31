#!/bin/bash
# 2試合(1試合単位)を新レイアウトでレンダ→h264化。並列2・cv2デフォルト(2プロセスなら競合軽微)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
OUTDIR=data/indicators_v2/overlay
mkdir -p "$OUTDIR"

render() {
  local vid=$1 start=$2 end=$3 warm=$4 name=$5
  local src="$HOME/frames/video_${vid}.mp4"
  [ -f "$src" ] || src="data/frames/video_${vid}.mp4"
  nice -n 5 ./venv/bin/python -m scripts.visualize_advantage_overlay \
    --video "$src" --out "$OUTDIR/${name}.mp4" \
    --start-sec "$start" --end-sec "$end" --warmup-sec "$warm" \
    --exclude-video "video_${vid}" > "logs/render_${name}.log" 2>&1
}

# video_30 game18: 2828-2946s 勝者1P / video_35 game11: 1396-1514s 勝者2P
render 30 2828 2946 16 match_v30_g18_1Pwin &
render 35 1396 1514 16 match_v35_g11_2Pwin &
wait
echo "[render] 2試合レンダ完了"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
for name in match_v30_g18_1Pwin match_v35_g11_2Pwin; do
  "$FF" -y -i "$OUTDIR/${name}.mp4" -c:v libx264 -preset medium -crf 24 \
    -pix_fmt yuv420p -movflags +faststart "$OUTDIR/${name}_h264.mp4" 2>/dev/null
  ls -la "$OUTDIR/${name}_h264.mp4"
done
echo "[render] h264化完了 ALL DONE"
