"""B14: 全フレーム完全一致と段別時間を検査する。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OUTPUT_FIELDS = ('frame', 't_sec', 'boards', 'raw', 'states', 'stable', 'active', 'scores', 'ready')
SOURCE_FILES = ('src/background_fingerprint.py', 'src/image_reader.py',
                'src/match_end_detector.py', 'src/telop_detector.py', 'src/prepared_template.py',
                'src/ui_mask.py', 'src/recognition_pipeline.py', 'src/production_config.py')
P50_LIMIT_MS, P95_LIMIT_MS = 28.0, 33.0
REPRESENTATIVE_START, REPRESENTATIVE_END = 2880.55, 3000.55
REPRESENTATIVE_FRAMES = 3600


def source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def compare_arrays(reference: dict, candidate: dict) -> dict:
    """欠落・順序差・一セル差も許容せず、許容誤差なしで比較する。"""
    count = len(reference['frame'])
    if count == 0 or len(candidate['frame']) != count:
        raise ValueError('比較母数が空または不一致です')
    differences = {}
    for key in OUTPUT_FIELDS:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape:
            raise ValueError(f'{key}: shape不一致です')
        mismatch = (left != right).reshape(count, -1).any(axis=1)
        differences[key] = int(mismatch.sum())
    return dict(frames=count, board_cells=count*2*13*6,
                differing_frames=differences, equal=not any(differences.values()))


def read_arrays(path: Path, start: float, end: float) -> dict:
    with np.load(path/'recognition.npz') as data:
        mask = (data['t_sec'] >= start) & (data['t_sec'] < end)
        return {key: data[key][mask] for key in OUTPUT_FIELDS}


def stage_summary(path: Path, start: float, end: float) -> dict:
    from scripts.live_b14_meter import STAGES
    with np.load(path/'stages.npz') as data:
        mask = (data['t_sec'] >= start) & (data['t_sec'] < end)
        result = {key: np.percentile(data[key][mask], [50, 95]).tolist()
                  for key in (*STAGES, 'recognition_ms')}
        result['frames'] = int(mask.sum())
    return result


def validate(reference: Path, candidate: Path, start: float, end: float) -> dict:
    manifest = json.loads((candidate/'manifest.json').read_text())
    if manifest['source_hashes'] != source_hashes() or manifest['baseline'] or manifest['original_templates']:
        raise ValueError('検証対象と現在の認識コードが一致しません')
    bounds = json.loads((candidate/'bounds.json').read_text())
    actual_end = min(end, bounds['end']/bounds['fps'])
    left, right = (read_arrays(path, start, actual_end) for path in (reference, candidate))
    frames = np.arange(bounds['start'], bounds['end'], bounds['stride'])
    expected = frames[(frames/bounds['fps'] >= start) & (frames/bounds['fps'] < actual_end)]
    if not np.array_equal(right['frame'], expected):
        raise ValueError('対象の全フレームが揃っていません')
    result = compare_arrays(left, right)
    stages = stage_summary(candidate, REPRESENTATIVE_START, REPRESENTATIVE_END)
    if stages['frames'] != REPRESENTATIVE_FRAMES:
        raise ValueError('代表2分の3600フレームが必要です')
    p50, p95 = stages['recognition_ms']
    result.update(stages=stages, target_met=p50 <= P50_LIMIT_MS and p95 <= P95_LIMIT_MS,
                  source_hashes=manifest['source_hashes'], reference=str(reference), candidate=str(candidate),
                  comparison_start=start, comparison_end=actual_end,
                  measurement_start=REPRESENTATIVE_START, measurement_end=REPRESENTATIVE_END)
    (candidate/'validation.json').write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--start', type=float, default=2580.566)
    parser.add_argument('--end', type=float, default=3000.566)
    args = parser.parse_args()
    result = validate(args.reference, args.candidate, args.start, args.end)
    print(json.dumps(result, indent=2))
    if not result['equal']:
        raise SystemExit('出力差があるため採用不可')


if __name__ == '__main__':
    main()
