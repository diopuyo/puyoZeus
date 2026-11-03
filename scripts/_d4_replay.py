"""本番CLIの保存入力再生を外から観測するD4専用診断。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from functools import wraps
import gzip
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from scripts import replay_exchange_event_20260926 as replay_module
from src.exchange_event_evaluator import FileExchangeModels
from src.exchange_event_record import encode

OUT = Path('logs/d4')
Q = 'q_7gc4TgFig'


def dump(stream: Any, value: Any) -> None:
    """NumPyと盤面の保存形式を本番記録と揃える。"""
    stream.write(json.dumps(encode(value), separators=(',', ':'))+'\n')


class Trace:
    """評価後の状態だけを複製し、評価器には書き戻さない。"""
    def __init__(self, dest: Path) -> None:
        self.stream = gzip.open(dest/'trace.jsonl.gz', 'wt')
        self.calls: list = []
        self.number = 0

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        tracker, prefire = overlay.tracker, overlay._prefire
        current = tracker.current
        chains = [] if current is None else [asdict(c) for c in current.chains]
        beliefs = [] if prefire is None else [[c.probs for c in b.cells] for b in prefire.beliefs]
        dump(self.stream, dict(t=inputs[3], game=inputs[4], p1=tracker.probability,
            source=tracker.source, exchange=current.exchange_id if current else None,
            chains=chains, counts=getattr(tracker, 'count_sides', None), hidden=beliefs,
            layer=getattr(tracker, 'layer_eval', None), model_calls=self.calls))
        self.calls = []
        self.number += 1
        if self.number % 3000 == 0:
            print(dict(frames=self.number, t=inputs[3]), flush=True)

    def install(self) -> None:
        """モデルが実際に受け取った特徴量と出力を保存する。"""
        original = FileExchangeModels.predict_source_probability
        @wraps(original)
        def predict(model: Any, name: str, features: np.ndarray) -> float:
            probability = original(model, name, features)
            self.calls.append(dict(name=name, features=features.tolist(), p=probability))
            return probability
        FileExchangeModels.predict_source_probability = predict


def main() -> None:
    """独立プロセス内だけ計装し、本番OFF/ONとの等価性を検収する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--mode', choices=('off', 'on'), required=True)
    args = parser.parse_args()
    dest = OUT/'replay'/args.mode/args.source
    dest.mkdir(parents=True, exist_ok=True)
    trace = Trace(dest)
    trace.install()
    original = replay_module.replay
    def observed(*values: Any, **kwargs: Any) -> dict:
        return original(*values, **dict(kwargs, observer=trace))
    replay_module.replay = observed
    root = Path('logs/e31' if args.mode == 'off' else 'logs/r1')
    sys.argv = ['d4', str(root/'records'/f'{args.source}.jsonl.gz'), '--out', str(dest),
                '--production-exchange-event']
    replay_module.main()
    trace.stream.close()
    suffix = args.source if args.source in ('review', 'zenchi') else f'renders/{args.source}/on'
    check = replay_module.compare(Path('logs/r1')/args.mode/suffix, dest)
    (dest/'EQUIVALENCE.json').write_text(json.dumps(check))


if __name__ == '__main__':
    main()
