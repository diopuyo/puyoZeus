"""scan_telop_prefilter_b21 の npz を集計し、粗スコアの床ごとに「決定 (is_visible / bbox) の差」を全数で数える。

列: frame, exact_t0, exact_t1, x_t0, y_t0, x_t1, y_t1, half_t0, half_t1, quarter_t0, quarter_t1。
決定の再構成は TelopDetector.detect と同じ (テンプレートは名前順、`>` で最大更新、可視は best>=閾値、bbox は best の位置)。
使い方: python analyze_telop_scan.py <npz...> [--scale quarter|half] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLD = 0.55
FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
EXACT = slice(1, 3)
POSITION = {0: (3, 4), 1: (5, 6)}
COARSE = {'half': slice(7, 9), 'quarter': slice(9, 11)}
TEMPLATES = 2


def decisions(data: np.ndarray, coarse: np.ndarray, floor: float | None) -> list[tuple]:
    """各フレームの (is_visible, x, y, template index)。floor=None は予備判定なし (基準)。"""
    out = []
    for row, coarse_row in zip(data, coarse):
        best, best_index = -1.0, -1
        for index in range(TEMPLATES):
            if floor is not None and coarse_row[index] < floor:
                continue
            if row[1+index] > best:
                best, best_index = float(row[1+index]), index
        visible = best >= THRESHOLD
        pos = tuple(row[list(POSITION[best_index])]) if visible else None
        out.append((visible, pos, best_index if visible else -1))
    return out


def summarize(data: np.ndarray, scale: str) -> dict:
    coarse = data[:, COARSE[scale]]
    exact = data[:, EXACT]
    reference = decisions(data, coarse, None)
    positives = exact >= THRESHOLD-1e-4  # 本番の peak が全解像度へ進む条件と同じ
    result = dict(frames=int(len(data)), visible_frames=int(sum(v for v, _, _ in reference)),
                  positive_template_frames=int(positives.sum()),
                  min_coarse_on_positive=[float(coarse[positives[:, i], i].min()) if positives[:, i].any() else None
                                          for i in range(TEMPLATES)],
                  max_coarse_on_negative=[float(coarse[~positives[:, i], i].max()) for i in range(TEMPLATES)],
                  floors={})
    for floor in FLOORS:
        fast = decisions(data, coarse, floor)
        diffs = sum(a != b for a, b in zip(reference, fast))
        goes_exact = float((coarse >= floor).mean())
        result['floors'][str(floor)] = dict(decision_differences=int(diffs), exact_path_fraction=goes_exact,
                                            skipped_template_frames=int((coarse < floor).sum()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--scale', choices=sorted(COARSE), default='quarter')
    parser.add_argument('--json', type=Path)
    options = parser.parse_args()
    per_file = {}
    chunks = []
    for path in options.files:
        data = np.load(path)['data']
        chunks.append(data)
        per_file[path.name] = summarize(data, options.scale)
    total = summarize(np.concatenate(chunks), options.scale)
    output = dict(scale=options.scale, total=total, per_file=per_file)
    text = json.dumps(output, indent=1)
    if options.json:
        options.json.write_text(text, encoding='utf-8')
    print(json.dumps(total, indent=1))


if __name__ == '__main__':
    main()
