"""E35全条件合格時だけ、本番構成の音声付きレビューを出力する。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg

from scripts import render_e18_review_20260928 as video


def main() -> None:
    """解像度・CRF・映像上限・音声・サイズを既存検収器で確認する。"""
    assert json.loads(Path('logs/e35/SUMMARY.json').read_text())['passed']
    prior = video.prior
    base = prior.ROOT/'logs/review_zenchi_g41_43_e27'
    dest = prior.ROOT/'logs/review_zenchi_g41_43_e35'
    prior.DEST = video.DEST = dest
    video.MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e35_mobile.mp4')
    cmd = [v.replace(str(base), str(dest)) for v in json.loads((base/'status.json').read_text())['command']]
    cmd[cmd.index('-m')+1] = 'scripts.visualize_e35_overlay'
    cmd.remove('--worker')
    cmd.extend(['--production-exchange-event', '--post-counter-death-bound'])
    start, end = (float(cmd[cmd.index(k)+1]) for k in ('--start-sec', '--end-sec'))
    state = dict(command=cmd, state='running', started=time.time())
    prior.save('status.json', state)
    with (dest/'render.log').open('a') as log:
        subprocess.run(['nice', '-n', '10', *cmd], cwd=prior.ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(dest/'overlay_video_only.mp4'),
            '-ss', str(start), '-t', str(end-start), '-i', cmd[cmd.index('--video')+1],
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest',
            str(dest/'overlay.mp4')], stdout=log, stderr=subprocess.STDOUT, check=True)
        prior.save('complete.json', dict(full=prior.verify(dest/'overlay.mp4'), mobile=video.mobile(log),
            encoding=dict(width=prior.WIDTH, crf=prior.CRF, maxrate_kbps=prior.VIDEO_KBPS,
                          audio_kbps=prior.AUDIO_KBPS)))
    (dest/'overlay_video_only.mp4').unlink()
    state.update(state='completed', elapsed_seconds=time.time()-state['started'])
    prior.save('status.json', state)


if __name__ == '__main__':
    main()
