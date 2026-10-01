"""固定L・学習済み係数でPhase 6を因果再生する。学習はセット1だけ。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_prefire_replay_20260930 import RECORDS, options, model_directory
from scripts.prefire_v5b_deferred import DeferredComputations
from src.prefire_best_play_v6 import BestPlayV6Layer, Strength

OUT = Path('logs/prefire_prediction/v6')
EVALSET = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set')


def record_path(source: str) -> Path:
    """セットを混ぜず既存記録を読む。"""
    if source.startswith('c') and source[1:].isdigit():
        return EVALSET/'collect/records'/f'{source}.jsonl.gz'
    if source.startswith('s') and source[1:].isdigit():
        return EVALSET/'set2/collect/records'/f'{source}.jsonl.gz'
    if source == 'check':
        return EVALSET/'set2/collect/records/check.jsonl.gz'
    return RECORDS/f'{source}.jsonl.gz'


def run(source: str, mode: str, record: Path | None = None) -> dict:
    """基準照合以外は固定L必須。a(z)未学習の収集をセット2へ適用しない。"""
    from scripts import d5_runtime
    from scripts.replay_exchange_event_20260926 import replay
    if mode == 'train' and source not in ('zenchi', *(f'c{i}' for i in range(1, 7))):
        raise ValueError('係数の学習用再生はセット1だけ')
    dest = OUT/mode/source
    if (dest/'DONE.json').exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True, exist_ok=True)
    d5_runtime.OUT = dest/'runtime'
    d5_runtime.install()
    opts = dict(options('off'), switch_smoothing=True)
    adapter = DeferredComputations(BestPlayV6Layer)
    if mode != 'baseline':
        latency = json.loads((OUT/'latency.json').read_text())['latency_sec']
        strength = Strength((0.,)*5, identity=True) if mode == 'train' else Strength.load(OUT/'strength.json')
        opts.update(prefire_best_play_v6=True, prefire_best_play_latency=latency, prefire_v6_strength=strength)
        adapter.install()
    try:
        result = replay(record or record_path(source), dest, model_directory(opts), True, None, **opts)
    finally:
        if mode != 'baseline':
            adapter.restore()
    result.update(input=str(record or record_path(source)), deferred=adapter.counts(),
                  options={k:v for k,v in opts.items() if k != 'prefire_v6_strength'})
    (dest/'DONE.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main() -> None:
    """単体・短窓を先に実行してから全区間へ広げる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('baseline', 'train', 'replay', 'cut'))
    parser.add_argument('source')
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.mode, args.record)), flush=True)


if __name__ == '__main__':
    main()
