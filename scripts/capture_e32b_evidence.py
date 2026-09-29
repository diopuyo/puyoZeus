"""E32bで使った式の表示時刻を、元映像の静止画として保存する。"""
from __future__ import annotations
import json
from pathlib import Path
import cv2

OUT = Path('logs/e32b/images')
TIMES = {
    'zenchi': (2848.383333333333, 2934.883333333333, 2936.2166666666667,
               2938.9166666666665, 3095.35, 3102.2166666666667,
               3109.1833333333334, 3110.0833333333335, 3314.9166666666665),
    'q_7gc4TgFig': (878.8, 880.7666666666667, 882.8),
    'fcXG83vInDY': (267.93333333333334, 269.96666666666664, 275.0, 817.0),
}


def main() -> None:
    """元解像度のフレームを取り出し、参照時刻と実フレーム番号を記録する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    cv2.setNumThreads(1)
    for source,times in TIMES.items():
        metadata = json.loads(Path(f'logs/e31/records/{source}.jsonl.json').read_text())
        cap = cv2.VideoCapture(metadata['video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps > 0, metadata['video']
        for stamp in times:
            index = round(stamp*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            success, frame = cap.read()
            assert success, (source, stamp)
            path = OUT/f'{source}_{stamp:.3f}.jpg'
            cv2.imwrite(str(path),frame)
            manifest.append(dict(source=source,t=stamp,frame=index,fps=fps,path=str(path)))
        cap.release()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
