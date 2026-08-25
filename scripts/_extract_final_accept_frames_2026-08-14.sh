#!/bin/bash
# 検収最終確認フレーム抽出。data/verify/demo_fixed_2026-08-13/frames_final_accept/ へ保存。
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
CONFIRM=data/verify/demo_fixed_2026-08-13/demo_fixed_3match.mp4
# デモ内相対秒 = source - 162 (start_sec)
grab "$CONFIRM" 70.87 final2_issue12_before_t70.87
grab "$CONFIRM" 72.53 final2_issue12_pretransient_t72.53
grab "$CONFIRM" 72.87 final2_issue12_holdstart_t72.87
grab "$CONFIRM" 74.0  final2_issue12_holdmid_t74.0
grab "$CONFIRM" 75.9  final2_issue12_holdend_t75.9
grab "$CONFIRM" 31.0  final2_issue910_before_t31.0
grab "$CONFIRM" 32.13 final2_issue910_pretransient_t32.13
grab "$CONFIRM" 32.53 final2_issue910_holdstart_t32.53
grab "$CONFIRM" 36.0  final2_issue910_holdmid_t36.0
grab "$CONFIRM" 38.9  final2_issue910_holdend_t38.9
echo DONE_CONFIRM_FRAMES
