"""OS間の最初の数値差を、特徴量・木の出力・確率変換へ分解する。"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def instrument(path: Path) -> object:
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    output = path.open('x', encoding='utf-8', buffering=1)
    original = HGB.predict_proba

    def predict(model: object, values: object) -> object:
        probabilities = original(model, values)
        raw = model._raw_predict(values)
        output.write(json.dumps(dict(shape=list(values.shape),
            input_sha=hashlib.sha256(values.tobytes()).hexdigest(),
            raw=raw.tolist(), probability=probabilities.tolist()))+'\n')
        return probabilities

    HGB.predict_proba = predict
    return output


def run(args: argparse.Namespace) -> None:
    sys.path[:0] = [str(args.app), str(args.native)]
    from src.phase_j.live_v49f_engine import load_engine
    from src.phase_j.live_v49f_origins import OriginStream
    log = instrument(args.out)
    engine = load_engine(args.assets/'model.joblib', args.assets/'timing_table.json',
        args.app/'docs/verification/exch_review_v310/policy_locked.json',
        args.app/'docs/verification/prefire_v44/model.json')
    origins = OriginStream(engine.base.policy, engine.base.native)
    updates = 0
    try:
        with gzip.open(args.packets, 'rt', encoding='utf-8') as stream:
            for line in stream:
                packet = json.loads(line)
                forecasts = origins.observe(packet['game'], packet['t'], packet['origins'])
                _, value = engine.observe(packet['t'], packet['game'], packet['raw'], forecasts)
                updates += value is not None
                if updates >= 20:
                    break
    finally:
        engine.close()
        log.close()
    print('diagnostic updates', updates, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('app', 'native', 'assets', 'packets', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    run(parser.parse_args())
