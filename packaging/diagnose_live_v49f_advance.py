"""保存された同一の枝計算結果から、OS別の補正計算を比較する。"""
from __future__ import annotations
import argparse
import json
import math
from types import SimpleNamespace
import sys
from pathlib import Path

DIAGNOSTIC_ROW = 103


def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.app))
    import numpy as np
    from scipy.special import expit
    from src.prefire_choice_model import predict_pair, features
    from src.prefire_post_v46 import projected_advance
    if args.scalar:
        from live_v49f_numeric_experiment import configure
        configure(SimpleNamespace(evaluators=[]))
    with args.input.open() as stream:
        value = next(json.loads(line) for index, line in enumerate(stream) if index == DIAGNOSTIC_ROW)
    model = json.loads((args.app/'docs/verification/prefire_v44/model.json').read_text())['models']['exchange_mixture']
    records, stamp = value['pre_records'], value['input_sec']
    result = dict(pair=predict_pair(records, model, stamp),
                  advance=projected_advance(records, model, stamp), expected=value['pre_delta_full'])
    vector, coef = features(*records), np.asarray(model['choice']['coef'])
    logit = float(vector@coef+model['choice']['intercept'])
    result['choice'] = dict(vector=vector.tolist(), coef=coef.tolist(), logit=logit,
        scipy_sigmoid=float(expit(logit)), math_sigmoid=1/(1+math.exp(-logit)))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(result['advance'], result['expected'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalar', action='store_true')
    for key in ('app', 'input', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    run(parser.parse_args())
