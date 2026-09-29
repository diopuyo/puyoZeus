"""E27の検収合格または明示的な候補承認に基づき、音声付きレビューを作る。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time
import imageio_ffmpeg
from scripts import render_e18_review_20260928 as video


def run(approved_candidate: bool = False) -> None:
    """横1280・CRF28・映像上限1Mbps・30MB以下・音声保持を検証する。"""
    assert approved_candidate or json.loads(Path('logs/e27/on/METRICS.json').read_text())['candidate']
    prior = video.prior
    dest = prior.ROOT/'logs/review_zenchi_g41_43_e27'
    prior.DEST = video.DEST = dest
    video.MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e27_mobile.mp4')
    baseline = prior.ROOT/'logs/review_zenchi_g41_43_e22'
    previous = json.loads((baseline/'status.json').read_text())
    cmd = [p.replace(str(baseline), str(dest)) for p in previous['command']]
    from scripts.run_e27 import OPTIONS
    options = OPTIONS
    flags = {key: enabled for key, enabled in options.items()
             if key in ('color_score_safety', 'multi_landing_death', 'death_pending_ledger', 'hidden_row_death')}
    flags['midchain_completion'] = True
    cmd.extend('--'+key.replace('_', '-') for key, enabled in flags.items() if enabled)
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
