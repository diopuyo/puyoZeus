"""E27レビューの映像・音声・サイズと、実描画の場面時刻を検査する。"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
from scripts.run_e3_exchange_eval_20260926 import save_json, digest

DEST = Path('logs/review_zenchi_g41_43_e27')
OUT = Path('logs/e28')
MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e27_mobile.mp4')
PREVIEW_TIMES = (20., 100., 175.95)


def previews(path: Path, prefix: str) -> None:
    """冒頭・途中・指定場面の画面を目視で確認できるようにする。"""
    capture = cv2.VideoCapture(str(path))
    for stamp in PREVIEW_TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
        ok, frame = capture.read()
        assert ok
        assert cv2.imwrite(str(OUT/'frames'/f'{prefix}_{stamp:.2f}.jpg'), frame)
    capture.release()


def main() -> None:
    """完成記録だけでなく、最終ファイルを全編デコードして確認する。"""
    status = json.loads((DEST/'status.json').read_text())
    assert status['state'] == 'completed'
    complete = json.loads((DEST/'complete.json').read_text())
    assert complete['mobile'][0]['bytes'] <= 30_000_000
    assert complete['mobile'][0]['width'] == 1280
    assert complete['mobile'][0]['frames'] == complete['full']['frames'] == complete['csv_rows']
    assert complete['mobile'][0]['audio'] == complete['full']['audio'] == 'aac'
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(MOBILE),
        '-f', 'null', '-'], capture_output=True, text=True, check=True)
    assert not result.stderr.strip(), result.stderr
    with (DEST/'review_data.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    scene = [r for r in rows if float(r['t_sec']) >= 2755]
    first = next(float(r['t_sec']) for r in scene if float(r['p1_display']) >= .95)
    terminal = next(float(r['t_sec']) for r in scene if r['source'] == 'confirmed_death')
    previews(DEST/'overlay.mp4', 'full')
    previews(MOBILE, 'mobile')
    value = dict(**complete, decode_errors=0, live_first_p2_le_5=first,
        live_confirmed_death=terminal, replay_first_p2_le_5=2771.25,
        sha256={str(p): digest(p) for p in (DEST/'overlay.mp4', MOBILE)},
        settings=complete['mobile_encoding'])
    save_json(OUT/'VIDEO_VERIFICATION.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
