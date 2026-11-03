"""Windows/Linux推論の診断原票を全列比較する。許容誤差での合格は行わない。"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def compare(root: Path) -> dict:
    linux = [json.loads(s) for s in (root/'numeric_linux_1.jsonl').read_text().splitlines()]
    windows = [json.loads(s) for s in (root/'numeric_windows_1.jsonl').read_text().splitlines()]
    result = dict(linux_calls=len(linux), windows_calls=len(windows), input_differences=0,
                  raw_differences=0, probability_differences=0, samples=[])
    for index, (a, b) in enumerate(zip(linux, windows)):
        result['input_differences'] += (a['shape'], a['input_sha']) != (b['shape'], b['input_sha'])
        result['raw_differences'] += a['raw'] != b['raw']
        result['probability_differences'] += a['probability'] != b['probability']
        if a['probability'] != b['probability']:
            result['samples'].append(dict(index=index, linux=a, windows=b,
                math_sigmoid=[[1/(1+math.exp(-x)) for x in row] for row in a['raw']]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = compare(args.root)
    with (args.root/'numeric_comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
