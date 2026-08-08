#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(PYTHONPATH=. ./venv/bin/python -c "from src.video_compositer import VideoCompositor; print(VideoCompositor._resolve_ffmpeg_bin())")
echo "FFBIN=${FF}"
rm -f /tmp/test_audio.aac
"${FF}" -y -ss 2883.1 -t 5 -i data/frames/video_olRyxDGacbg.mp4 -vn -c:a aac /tmp/test_audio.aac > logs/_probe_audio_extract_2026-08-03.log 2>&1
echo "extract exit=$?"
ls -la /tmp/test_audio.aac
echo "--- extract log tail ---"
tail -30 logs/_probe_audio_extract_2026-08-03.log

echo "=== mux step ==="
SILENT=data/verify/advantage_videos_olRyxDGacbg_2026-08-03/delta_winprob_demo_olRyxDGacbg_g1_silent.mp4
ls -la "$SILENT"
"${FF}" -y -i "$SILENT" -i /tmp/test_audio.aac -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest /tmp/test_muxed.mp4 > logs/_probe_audio_mux_2026-08-03.log 2>&1
echo "mux exit=$?"
ls -la /tmp/test_muxed.mp4
echo "--- mux log tail ---"
tail -30 logs/_probe_audio_mux_2026-08-03.log
