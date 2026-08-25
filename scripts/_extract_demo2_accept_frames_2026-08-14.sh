#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/demo_fixed_2026-08-13/frames_final_accept
mkdir -p "$OUT"
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
grab() {
  local src="$1" t="$2" name="$3"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$t" -i "$src" -frames:v 1 "$OUT/$name.png"
  echo "[grab] $name t=$t"
}
D2=data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4
# デモ内相対秒 = source - 230 (start_sec)
grab "$D2" 40.0 demo2v2_issue11_before_t40.0
grab "$D2" 41.5 demo2v2_issue11_window_t41.5
grab "$D2" 43.7 demo2v2_issue11_window_t43.7
grab "$D2" 44.9 demo2v2_issue11_window_t44.9
grab "$D2" 99.3 demo2v2_defender_example_t99.3
grab "$D2" 106.0 demo2v2_defender_example_t106.0
grab "$D2" 108.0 demo2v2_defender_example_t108.0
echo DONE_DEMO2_FRAMES
