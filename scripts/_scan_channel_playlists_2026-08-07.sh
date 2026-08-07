#!/bin/bash
# 既知動画からチャンネルを特定し、チャンネルの全プレイリストを列挙 (新リーグ探索)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
YT="./venv/bin/python -m yt_dlp --no-update"
echo "=== チャンネル特定 (xX4qC3im26w = plC先頭) ==="
CH=$($YT --print "%(channel_url)s" --no-playlist "https://www.youtube.com/watch?v=xX4qC3im26w" 2>/dev/null | tail -1)
echo "channel: $CH"
echo "=== チャンネルのプレイリスト一覧 ==="
$YT --flat-playlist --print "%(id)s\t%(title)s" "$CH/playlists" 2>/dev/null
