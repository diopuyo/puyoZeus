"""指定したE34検証の全条件合格時だけ、音声付きレビューを生成する。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg

from scripts import render_e18_review_20260928 as video


def main(summary: Path = Path('logs/e34b/SUMMARY.json')) -> None:
    """既存E27コマンドにE32と起点ガードを追加し、既存の音声・圧縮検収を使う。"""
    assert json.loads(summary.read_text())['passed']
    prior = video.prior
    baseline = prior.ROOT/'logs/review_zenchi_g41_43_e27'
    dest = prior.ROOT/'logs/review_zenchi_g41_43_e34'
    prior.DEST = video.DEST = dest
    video.MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e34_mobile.mp4')
    cmd = [p.replace(str(baseline), str(dest))
           for p in json.loads((baseline/'status.json').read_text())['command']]
    cmd.extend(['--prefire-snapshot', '--hidden-row-belief', '--prefire-origin-guard'])
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
        prior.save('complete.json', dict(full=prior.verify(dest/'overlay.mp4'), mobile=video.mobile(log),
            mobile_encoding=dict(width=prior.WIDTH, crf=prior.CRF,
                                 maxrate_kbps=prior.VIDEO_KBPS, audio_kbps=prior.AUDIO_KBPS)))
    (dest/'overlay_video_only.mp4').unlink()
    state.update(state='completed', elapsed_seconds=time.time()-state['started'])
    prior.save('status.json', state)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('logs/e34b/SUMMARY.json'))
    main(parser.parse_args().summary)
