"""保存済み通し差分に対し、待機区間からの再初期化を反証する短区間台。"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import sys
import time

FRAME_SIZE = (1920, 1080)


def reference_rows(directory: Path, start: int, end: int) -> dict[int, dict]:
    """差分票は疎な旧出力。反証には使えるが全項目一致の証明には使わない。"""
    manifest = json.loads((directory / 'checkpoint.json').read_text())
    result = {}
    for name in manifest['chunks']:
        with gzip.open(directory / name, 'rt') as stream:
            for row in json.load(stream)['rows']:
                if start <= row['frame'] < end:
                    result[row['frame']] = row['old']
    return result


def probe(case: dict, config: dict, previous: Path, output: Path) -> dict:
    import cv2
    from scripts.compare_image_free_all import mode
    from scripts.image_free_runtime import Runtime
    from scripts.obs_priority_guard import ObsPriorityGuard
    video = next(v for v in config['videos'] if v['id'] == case['video'])
    start, end = case['start'], case['end']
    reference = reference_rows(previous / video['id'], start, end)
    guard, mismatches, checked = ObsPriorityGuard(), [], 0
    directory = output / f"{video['id']}_{start}"
    directory.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    began = time.monotonic()
    with mode(False):
        runtime = Runtime(Path(video['path']), directory, config)
        for index in range(start, end):
            guard.checkpoint()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'復号失敗: {index}')
            actual = runtime.step(cv2.resize(frame, FRAME_SIZE), index, fps)
            expected = reference.get(index, {})
            checked += len(expected)
            fields = [k for k, value in expected.items() if actual[k] != value]
            if fields:
                details = {k: dict(old=expected[k], split=actual[k]) for k in fields
                           if not isinstance(expected[k], dict)}
                for key in fields:
                    if isinstance(expected[key], dict) and isinstance(actual[key], dict):
                        details[key] = {k: dict(old=v, split=actual[key].get(k))
                                        for k, v in expected[key].items()
                                        if v != actual[key].get(k)}
                mismatches.append(dict(frame=index, fields=fields, details=details))
    cap.release()
    return dict(**case, checked_fields=checked, reference_frames=len(reference),
                mismatches=mismatches, complete_reference=False, passed=False,
                seconds=time.monotonic()-began, frames=end-start,
                rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    # 旧通し実行の凍結ソースを使い、並行中の候補修正から隔離する。
    sys.path.insert(0, str(args.previous / 'source'))
    import scripts
    scripts.__path__ = [str(args.previous / 'source/scripts'), *scripts.__path__]
    from scripts.compare_image_free_all import read_config, save_json
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    config = read_config(args.previous / 'config.json')
    os.chdir(config['asset_root'])
    os.environ['PUYO_B3_JUDGMENT'] = '1'
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in json.loads(args.cases.read_text()):
        results.append(probe(case, config, args.previous, args.output))
        save_json(args.output / 'split_proof.json', dict(passed=False, cases=results,
            reason='疎な差分票だけでは全項目一致を証明できない。差があれば分割禁止。'))


if __name__ == '__main__':
    main()
