"""R1全条件合格時だけ、本番評価と新フラグで音声付きレビューを生成する。"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import time
from typing import TextIO
import imageio_ffmpeg
from scripts import render_e18_review_20260928 as video
from scripts.replay_exchange_event_20260926 import compare

DEST = video.prior.ROOT/'logs/review_zenchi_g41_43_r1'
MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_r1_mobile.mp4')


def command() -> list[str]:
    """E14の同区間・暖機を保ち、モデルと撃ち合い条件は本番CLIへ任せる。"""
    baseline = video.prior.ROOT/'logs/review_zenchi_g41_43_e14'
    cmd = [part.replace(str(baseline), str(DEST))
           for part in json.loads((baseline/'status.json').read_text())['command']]
    index = cmd.index('--exchange-event-model-dir')
    del cmd[index:index+2]
    return cmd+['--production-exchange-event', '--placement-signal-reconcile']


def mux(cmd: list[str], log: TextIO) -> None:
    """元映像の対応区間から音声を取り出し、描画した映像へ合成する。"""
    start, end = (float(cmd[cmd.index(flag)+1]) for flag in ('--start-sec', '--end-sec'))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i',
        str(DEST/'overlay_video_only.mp4'), '-ss', str(start), '-t', str(end-start),
        '-i', cmd[cmd.index('--video')+1], '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(DEST/'overlay.mp4')],
        stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """合否、再生との一致、音声、1280幅・CRF28・1Mbps・30MB上限を検収する。"""
    assert json.loads(Path('logs/r1/SUMMARY.json').read_text())['passed']
    prior = video.prior
    prior.DEST = video.DEST = DEST
    video.MOBILE = MOBILE
    cmd = command()
    state = dict(command=cmd, state='running', started=time.time())
    prior.save('status.json', state)
    with (DEST/'render.log').open('a') as log:
        subprocess.run(cmd, cwd=prior.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        parity = compare(Path('logs/r1/on/review'), DEST)
        mux(cmd, log)
        prior.save('complete.json', dict(full=prior.verify(DEST/'overlay.mp4'),
            mobile=video.mobile(log), replay_equivalence=parity,
            mobile_encoding=dict(width=prior.WIDTH, crf=prior.CRF,
                                 maxrate_kbps=prior.VIDEO_KBPS, audio_kbps=prior.AUDIO_KBPS)))
    state.update(state='completed', elapsed_seconds=time.time()-state['started'])
    prior.save('status.json', state)


if __name__ == '__main__':
    main()
