"""静止・予告ゼロの毎秒標本で近未来火力を監査用に計測する。"""
from __future__ import annotations
import gzip
import json
import time
from typing import Any
from scripts._audit_static_1003_extract import OUT, SOURCES, guard
import numpy as np
import pandas as pd
from src.board import Board
from src import indicators_v2 as iv
from src.scoring import compute_effective_rate


def pair(value: Any) -> tuple | None:
    """保存NEXTの型を復元する。"""
    return tuple(value['tuple']) if value is not None else None


def measure(side: dict, elapsed: float) -> dict:
    """既存指標の生値を保持し、Kを実際の最短手数とは混同しない。"""
    board = Board.from_list(side['confirmed_board']['board']['array'])
    fire = iv.near_future_fire_power(board, pair(side['next_pair']), pair(side['dnext_pair']),
                                    elapsed, fast_expand=True)
    result = dict(chain=iv.current_max_chain(board).raw,
                  ignite=iv.min_puyos_to_ignite(board).raw,
                  immediate=iv.immediate_fire_power(board, elapsed).raw,
                  known_next=fire.used_real_next)
    result.update({f'k{k}': fire.values[k].raw for k in range(1, 6)})
    return result


def run(source: str) -> None:
    """全時間の推測補間をせず、標本の時刻と実測母数を保存する。"""
    dest = OUT / f'{source}_fire.csv'
    prior = pd.read_csv(dest).set_index('t') if dest.exists() else None
    frame = pd.read_pickle(OUT / f'{source}_frames.pkl').set_index('t')
    with gzip.open(OUT / f'{source}_samples.json.gz', 'rt') as stream:
        samples = json.load(stream)
    cache, results, started = {}, [], time.monotonic()
    for sample in samples:
        row = frame.loc[sample['t']]
        if row.source != 'G_fe' or row.forecast != 0 or row.net != 0:
            continue
        # 保存されたモデル呼出の経過秒から同じ会計起点へ合わせる。
        elapsed = row.model_elapsed
        if prior is not None and sample['t'] in prior.index:
            old = prior.loc[sample['t']].to_dict()
            if compute_effective_rate(old['elapsed']) == compute_effective_rate(elapsed):
                results.append(dict(old, t=sample['t'], elapsed=elapsed))
                continue
        measured = dict(t=sample['t'], source_record=source, elapsed=elapsed)
        for i, side in enumerate(sample['sides'], 1):
            key = json.dumps([side['confirmed_board'], side['next_pair'], side['dnext_pair'],
                              compute_effective_rate(elapsed)], sort_keys=True)
            if key not in cache:
                cache[key] = measure(side, elapsed)
            measured.update({f'p{i}_{k}': v for k, v in cache[key].items()})
        results.append(measured)
        if len(results) % 50 == 0:
            guard()
            print(source, len(results), round(time.monotonic() - started, 1), flush=True)
    pd.DataFrame(results).to_csv(dest, index=False)
    print('FIRE_DONE', source, len(results), flush=True)


def main() -> None:
    """監査火力計測は一プロセスで逐次実行する。"""
    guard()
    for source in SOURCES:
        run(source)


if __name__ == '__main__':
    main()
