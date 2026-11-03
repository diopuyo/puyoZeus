"""静止表示の全STABLEフレームへ、盤面・NEXT別に火力を対応させる。"""
from __future__ import annotations
import gzip
import hashlib
import json
import time
import argparse
from scripts._audit_static_1003_extract import OUT, BASE, SOURCES, guard
from scripts._audit_static_1003_fire import measure
from src.scoring import compute_effective_rate
import pandas as pd


def identity(sides: list[dict], elapsed: float) -> str:
    """火力計算に関係する盤面・NEXT・おじゃま換算率だけをキャッシュキーにする。"""
    value = [[s['confirmed_board'], s['next_pair'], s['dnext_pair']] for s in sides]
    raw = json.dumps([value, compute_effective_rate(elapsed)], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def cached_samples(source: str, frame: pd.DataFrame) -> dict:
    """毎秒標本の既存実測を同一入力に限って再利用する。"""
    prior = pd.read_csv(OUT / f'{source}_fire.csv').set_index('t')
    with gzip.open(OUT / f'{source}_samples.json.gz', 'rt') as stream:
        samples = json.load(stream)
    cache = {}
    for s in samples:
        if s['t'] not in prior.index:
            continue
        row = prior.loc[s['t']]
        elapsed = frame.loc[s['t'], 'model_elapsed']
        if compute_effective_rate(elapsed) == compute_effective_rate(row.elapsed):
            cache[identity(s['sides'], elapsed)] = {k: v for k, v in row.items() if k.startswith('p')}
    return cache


def run(source: str) -> None:
    """対象フレームを間引かず処理し、重複盤面だけ計算を再利用する。"""
    dest = OUT / f'{source}_fire_dense.csv'
    if dest.exists():
        return
    frame = pd.read_pickle(OUT / f'{source}_frames.pkl').set_index('t')
    cache = cached_samples(source, frame)
    rows, current, computed, started = [], None, 0, time.monotonic()
    with gzip.open(BASE / f'collect/records/{source}.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['kind'] == 'update':
                current = r['args']['tuple']
            if r['kind'] != 'display':
                continue
            f = frame.loc[r['t_sec']]
            if f.source != 'G_fe' or f.forecast != 0 or f.net != 0:
                continue
            sides = [current[0]['namespace'][s]['namespace'] for s in ('p1', 'p2')]
            if not all(s['state']['state'] == 'STABLE' and s['confirmed_board'] for s in sides):
                continue
            key = identity(sides, f.model_elapsed)
            if key not in cache:
                cache[key] = {f'p{i}_{k}': v for i, side in enumerate(sides, 1)
                              for k, v in measure(side, f.model_elapsed).items()}
                computed += 1
                if computed % 100 == 0:
                    guard()
                    print(source, computed, round(time.monotonic() - started, 1), flush=True)
            rows.append(dict(t=r['t_sec'], source_record=source, key=key, **cache[key]))
    pd.DataFrame(rows).to_csv(dest, index=False)
    print('DENSE_DONE', source, len(rows), len(cache), computed, flush=True)


def main() -> None:
    """常にnice10で全分割を順次監査する。"""
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=SOURCES)
    args = parser.parse_args()
    for source in ([args.source] if args.source else SOURCES):
        run(source)


if __name__ == '__main__':
    main()
