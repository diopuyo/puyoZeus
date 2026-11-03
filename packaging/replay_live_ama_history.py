"""保存済み認識原票とB3世代原票を結合してamaの履歴欠番を再現する。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PUYO_LIVE_SCORE_CADENCE'] = '1'
os.environ['PUYO_OBS_EXACT_SEED'] = '1'


def legacy_tape(source: Path) -> type:
    """封印済み配布物の旧クラスを読取り専用で読み込む。"""
    spec = importlib.util.spec_from_file_location('ama_tape_before', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    return old.CausalTape


def parity_check(source: Path) -> dict:
    """成功した全bundleを修正前の同じ関数でも計算し、全フィールドを比較する。"""
    from src.ama_evaluation_v7_tape import CausalTape
    previous = legacy_tape(source)
    fixed, counts = CausalTape.bundle, dict(equal=0, recovered_missing=0)
    def bundle(self: object, colors: list) -> dict:
        try:
            result = fixed(self, colors)
        except ValueError as error:
            if str(error) == '連鎖前ツモ番号の観測履歴不足':
                try:
                    previous.bundle(self, colors)
                except KeyError:
                    counts['recovered_missing'] += 1
                else:
                    raise AssertionError('旧版でKeyErrorになる範囲以外の変更')
            raise
        assert result == previous.bundle(self, colors)
        counts['equal'] += 1
        return result
    CausalTape.bundle = bundle
    return counts


def packets(source: Path) -> list[dict]:
    timing = {round(r['t_sec'], 6): r for r in
              map(json.loads, (source/'b3_state_timing.jsonl').read_text().splitlines())}
    rows = []
    with gzip.open(source/'inputs.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'update':
                continue
            args = row['args']['tuple']
            meta = timing[round(args[3], 6)]
            rows.append(dict(t=args[3], game=args[4], id=meta['frame'], generation=meta['epoch'],
                raw=[args[0]['namespace'][s]['namespace'] for s in ('p1', 'p2')]))
    return rows


def snapshot(inputs: object) -> dict:
    tape = inputs.tape
    return dict(last=tape.last, pinned=tape.pinned, states=tape.states,
        sides=[asdict(s) for s in tape.tape.sides], aligned=tape.tape.aligned,
        counter_history=tape.counter_history, inheritance=tape.inheritance,
        colors=sorted(inputs.colors))


def replay(args: argparse.Namespace) -> None:
    import puyo_core
    from src.phase_j.live_ama_inputs import AmaInputs
    from src.ama_tsumo_seed_v5 import load_table
    args.out.mkdir(exist_ok=False)
    if args.legacy_source:
        from src.phase_j import live_ama_inputs
        live_ama_inputs.CausalTape = legacy_tape(args.legacy_source)
    parity = parity_check(args.parity_source) if args.parity_source else None
    table = load_table(ROOT/'models/ama_plain/seed_table_cache.npz')
    identity, inputs, failure = None, None, None
    rows, count, missing = packets(args.source), 0, 0
    with (args.out/'results.jsonl').open('w', encoding='utf-8') as stream:
        for index, packet in enumerate(rows):
            current = packet['generation'], packet['game']
            if current != identity:
                identity, inputs = current, AmaInputs(puyo_core, table)
            try:
                item = inputs.observe(packet)
                count += item is not None
                missing += item is not None and 'missing' in item
                stream.write(json.dumps(dict(packet=packet, item=item), ensure_ascii=False)+'\n')
            except Exception as error:
                failure = dict(index=index, packet=packet, state=snapshot(inputs),
                    error=f'{type(error).__name__}: {error}', stack=traceback.format_exc())
                break
    receipt = dict(source=str(args.source), packets=len(rows), requests=count, parity=parity,
                   missing=missing, failure=failure)
    (args.out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(packets=len(rows), requests=count, missing=missing,
                         failure=None if failure is None else failure['error'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--parity-source', type=Path)
    parser.add_argument('--legacy-source', type=Path)
    replay(parser.parse_args())
