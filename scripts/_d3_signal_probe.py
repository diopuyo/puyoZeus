"""保存された式欠測を、原映像の既存OCRで確認する。"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import cv2
from src.score_ocr import ScoreOcr, SCORE_1P_REGION, SCORE_2P_REGION
from scripts.enrich_e26_midchain import frame_at
from scripts._d3_inventory import OUT

VIDEO = Path('/mnt/d/puyo_analyzer/videos/source/mia8KCjr52g_first_0_900_20260925_v1.mp4')
TIMES = (509.0, 509.1, 509.3, 510.0)


def main() -> None:
    """元映像の式欄を観測し、診断用画像だけを書き出す。"""
    os.nice(19)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(VIDEO), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    fps, reader, output = cap.get(cv2.CAP_PROP_FPS), ScoreOcr.load_default(), []
    for stamp in TIMES:
        frame = frame_at(cap, round(stamp*fps))
        cv2.imwrite(str(OUT/f'mia_formula_{stamp:.1f}.jpg'), cv2.resize(frame, (960, 540)))
        matches = []
        for region in (SCORE_1P_REGION, SCORE_2P_REGION):
            y1, y2, x1, x2 = region
            crop = frame[y1-16:y2+16, x1:x2]
            scores = cv2.matchTemplate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), reader._mult_template_gray, cv2.TM_CCOEFF_NORMED)
            _, high, _, location = cv2.minMaxLoc(scores)
            matches.append(dict(ncc=high, location=location))
        output.append(dict(t=stamp, matches=matches, sides=[asdict(reader.read_formula_side(frame, side)) for side in ('1P','2P')]))
    cap.release()
    (OUT/'mia_formula_probe.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
