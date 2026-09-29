"""指定場面の原映像を目視確認用に抜き出す。"""
from pathlib import Path
import cv2

SOURCE = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4')
TIMES = (2752.65, 2754.5, 2755.016667, 2755.35, 2755.416667, 2756.716667,
         2759.95, 2760.816667, 2762.15, 2765., 2768., 2770., 2771.)


def main() -> None:
    """動画時刻とフレームを対応付け、元の音声・映像は変更しない。"""
    dest = Path('logs/e28/frames')
    dest.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(SOURCE))
    for stamp in TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
        ok, frame = capture.read()
        assert ok, stamp
        cv2.imwrite(str(dest/f'{stamp:.3f}.jpg'), frame)
    capture.release()


if __name__ == '__main__':
    main()
