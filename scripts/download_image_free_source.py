"""画像非同梱比較用の原動画を取得し、保持契約とメタデータを残す。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yt_dlp

from scripts.obs_priority_guard import ObsPriorityGuard

VIDEO_ID = 'c0BQoMJwwQU'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', type=Path)
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    args.output.mkdir(parents=True, exist_ok=True)
    options = dict(format='bestvideo[ext=mp4][height<=1080]/best[ext=mp4][height<=1080]',
                   outtmpl=str(args.output / 'zenchi_%(id)s.%(ext)s'),
                   noplaylist=True, continuedl=True, overwrites=False,
                   writeinfojson=True, writethumbnail=False,
                   progress_hooks=[lambda _: guard.checkpoint()],
                   socket_timeout=20, retries=3, fragment_retries=3)
    if args.node is not None:
        options['js_runtimes'] = {'node': {'path': str(args.node)}}
    receipt = dict(video_id=VIDEO_ID, url=f'https://www.youtube.com/watch?v={VIDEO_ID}',
                   retention='保持。元動画を保持する2026-08-31契約と今回の指示による',
                   audio='認識比較用の映像トラックを選択。音声付き成果物ではない',
                   completed=False)
    path = args.output / 'source_contract.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(receipt['url'], download=True)
    receipt.update(completed=True, title=info.get('title'), duration=info.get('duration'),
                   format_id=info.get('format_id'), filename=downloader.prepare_filename(info))
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
