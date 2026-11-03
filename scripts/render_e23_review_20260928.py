"""固定評価合格時だけ、E22と同条件＋新フラグのレビュー動画を作る。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg

from scripts import render_e18_review_20260928 as video


def run() -> None:
    """E22の再現コマンドへ新フラグだけを追加し、音声付き30MB以内にする。"""
    assert json.loads(Path('logs/e23/on/METRICS.json').read_text())['candidate']
    prior = video.prior
    dest = prior.ROOT/'logs/review_zenchi_g41_43_e23'
    prior.DEST = video.DEST = dest
    video.MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e23_mobile.mp4')
    baseline = prior.ROOT/'logs/review_zenchi_g41_43_e22'
    previous = json.loads((baseline/'status.json').read_text())
    cmd = [p.replace(str(baseline), str(dest)) for p in previous['command']]
    cmd.append('--multi-landing-death')
    start, end = (float(cmd[cmd.index(f)+1]) for f in ('--start-sec', '--end-sec'))
    state = dict(command=cmd, state='running', started=time.time())
    prior.save('status.json', state)
    with (dest/'render.log').open('a') as log:
        subprocess.run(cmd, cwd=prior.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i',
            str(dest/'overlay_video_only.mp4'), '-ss', str(start), '-t', str(end-start),
            '-i', cmd[cmd.index('--video')+1], '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(dest/'overlay.mp4')],
            stdout=log, stderr=subprocess.STDOUT, check=True)
        full = prior.verify(dest/'overlay.mp4')
        with (dest/'review_data.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == full['frames'] and any(r['1P_NF_ojama_k1'] for r in rows)
        prior.save('complete.json', dict(full=full, csv_rows=len(rows), mobile=video.mobile(log),
            mobile_encoding=dict(width=prior.WIDTH, crf=prior.CRF,
                                 maxrate_kbps=prior.VIDEO_KBPS, audio_kbps=prior.AUDIO_KBPS)))
    state.update(state='completed', elapsed_seconds=time.time()-state['started'])
    prior.save('status.json', state)


if __name__ == '__main__':
    run()
