"""指定場面の原映像を目視確認用に抜き出す。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import cv2

SOURCE = Path(_public_dev_text('<repo>/data/frames/video_zenchi_c0BQoMJwwQU.mp4'))
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
