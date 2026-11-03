"""前回と同じ元映像・切出し・MJPEG方式で検証素材を新規作成する。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE = Path('C:/puyo1_tools/soak/media/video_d04.mp4')
OUT = Path('D:/puyo_analyzer/packaging/build_20261020a_work/smoke_game_600_720.avi')
START, SECONDS = 600, 120


def main() -> None:
    os.environ['OPENCV_FFMPEG_DLL_DIR'] = 'D:/puyo_analyzer/packaging/dev_ffmpeg'
    os.environ['OPENCV_FFMPEG_THREADS'] = '1'
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from smoke_live_obs_20261020 import memory_guard
    import cv2
    memory_guard()
    assert SOURCE.is_file() and not OUT.exists()
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC,START*1000)
    writer = cv2.VideoWriter(str(OUT),cv2.CAP_OPENCV_MJPEG,
        cv2.VideoWriter_fourcc(*'MJPG'),fps,(1920,1080))
    assert writer.isOpened()
    try:
        for index in range(round(fps*SECONDS)):
            ok, frame = cap.read()
            assert ok, index
            writer.write(cv2.resize(frame,(1920,1080)))
            if index % round(fps*10) == 0:
                memory_guard()
                print('seconds',index/fps,flush=True)
    finally:
        writer.release()
        cap.release()
    with OUT.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    receipt = dict(source=str(SOURCE),start_sec=START,duration_sec=SECONDS,fps=fps,
        output=str(OUT),bytes=OUT.stat().st_size,sha256=digest,codec='CAP_OPENCV_MJPEG / MJPG',
        previous='packaging/smoke_live_v49f.py make_videoと同じ区間・解像度・encoder')
    OUT.with_suffix('.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(receipt,flush=True)


if __name__ == '__main__':
    main()
