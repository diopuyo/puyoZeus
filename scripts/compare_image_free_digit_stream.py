"""動画を順に全フレーム読み、OCR差を欠測も含め記録する（下流は別試験）。"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

from scripts.compare_image_free_digits import cv2, np
from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_digit import ImageFreeScoreOcr
from src.score_ocr import ScoreOcr

SIDES = ('1P', '2P')
PROGRESS_SECONDS = 30
FORMULA_FIELDS = ('valid', 'left', 'right', 'product')
BATCH_FRAMES = 24


def compare_side(frame: np.ndarray, side: str, teacher: ScoreOcr,
                 student: ImageFreeScoreOcr, prepared: tuple | None = None) -> tuple[dict, dict]:
    """信頼度の差も保存し、値の一致とは分けて数える。"""
    old = teacher.read_side_detail(frame, side)
    new = student.read_side_detail(frame, side) if prepared is None else prepared[0]
    old_f = asdict(teacher.read_formula_side(frame, side, old[2], old[3]))
    new_f = asdict(student.read_formula_side(frame, side, new[2], new[3])
                   if prepared is None else prepared[1])
    same_formula = all(old_f[k] == new_f[k] for k in FORMULA_FIELDS)
    counts = dict(digits=(sum(a == b for a, b in zip(old[2], new[2])), len(old[2])),
                  scores=(int(old[0] == new[0]), 1), formulas=(int(same_formula), 1))
    detail = dict(side=side, old_score=old[0], new_score=new[0],
                  old_digits=old[2], new_digits=new[2], old_confidences=old[3],
                  new_confidences=new[3], old_formula=old_f, new_formula=new_f)
    return counts, detail


def consecutive_frames(cap: cv2.VideoCapture, student: ImageFreeScoreOcr,
                       guard: ObsPriorityGuard) -> Iterator[tuple[np.ndarray, list]]:
    """seekや間引きをせず、連続フレームをまとめて推論する。"""
    while True:
        images = []
        for _ in range(BATCH_FRAMES):
            guard.checkpoint()
            ok, frame = cap.read()
            if not ok:
                break
            images.append(frame)
        if not images:
            return
        guard.checkpoint()
        prepared = student.read_batch(images)
        yield from zip(images, prepared)


def write_report(output: Path, counts: dict, frames: int, expected: int,
                 differences: int, complete: bool) -> None:
    """途中経過は合格にしない。動画終端と総フレーム数を検査する。"""
    metrics = {key: dict(equal=a, total=b, rate=a / b if b else None)
               for key, (a, b) in counts.items()}
    report = dict(metrics=metrics, frames=frames, expected_frames=expected,
                  difference_sides=differences, complete=complete,
                  gate_passed=complete and frames == expected and frames > 0
                  and all(a == b for a, b in counts.values()),
                  downstream='確定盤面・B3特徴・表示勝率はこのOCR単体試験では未測定',
                  method='フレーム0から順にread、間引きなし')
    temporary = output / 'comparison.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(output / 'comparison.json')


def run(video: Path, templates: Path, model: Path, output: Path) -> None:
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    teacher, student = ScoreOcr.load_default(templates), ImageFreeScoreOcr(model)
    cap = cv2.VideoCapture(str(video))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if expected <= 0 or fps <= 0:
        raise ValueError(f'動画を開けません: {video}')
    output.mkdir(parents=True, exist_ok=True)
    counts = {name: [0, 0] for name in ('digits', 'scores', 'formulas')}
    frames = differences = 0
    last = time.monotonic()
    with gzip.open(output / 'differences.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for frame, prepared in consecutive_frames(cap, student, guard):
            guard.checkpoint()
            for side_index, side in enumerate(SIDES):
                values, detail = compare_side(frame, side, teacher, student, prepared[side_index])
                for key, (a, b) in values.items():
                    counts[key][0] += a
                    counts[key][1] += b
                if any(a != b for a, b in values.values()):
                    differences += 1
                    detail.update(frame=frames, t_sec=frames / fps)
                    stream.write(json.dumps(detail, ensure_ascii=False) + '\n')
            frames += 1
            if time.monotonic() - last >= PROGRESS_SECONDS:
                stream.flush()
                write_report(output, counts, frames, expected, differences, False)
                print(f'{frames}/{expected} 差={differences}', flush=True)
                last = time.monotonic()
    cap.release()
    write_report(output, counts, frames, expected, differences, True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'templates', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    run(args.video, args.templates, args.model, args.output)


if __name__ == '__main__':
    main()
