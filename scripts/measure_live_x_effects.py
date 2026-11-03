"""撃った連鎖の火力 X の測定台 (段0) のうち、誤発火 M4 と副作用 M5 を数える。

M4 誤発火: 公開票に載った X の発火 (側, trigger) のうち、同じ側の実連鎖 (CHAIN 通知の
連鎖・断片の開始) が MATCH_RADIUS_SEC 以内に無いもの。取り下げ秒 = その X を載せた最初の
公開から、同じ側で載らなくなった最初の公開まで。窓は録画内の試合 (21c は45試合)。
段3 の暫定 X 欄 (x_provisional) が公開票にあれば同じ規則で別に数える。
M5 副作用: 2走行の B3・ama・v4.9f の更新原票を、計時の列を除いた全列で突き合わせ、公開行数を比べる。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.measure_live_x_timing import (SAVED_21C, jsonl_rows, load_publications,
                                           index_results, pickle_stream)

DIAG = Path(_public_dev_text('<work>/codex_jobs/exch_review_v3/OBS21C_EXCHANGE_LAG_DIAG_20261021'))
MATCH_RADIUS_SEC = 1.5     # summarize_replay.TRIGGER_RADIUS と同じ近傍
SIDES = (0, 1)
# compare_wall_gate.py の semantic と同じ除外 + 実時間 run の壁時計列。
# ama の更新票の計時 (2026-10-23 追加): result.timings (setup/conversion/read_*_sec 一式) と result.cpu_sec
# は走行ごとに変わる壁時計・CPU 時間。`_sec` を接尾辞で一括除外はしない: input_sec (入力の記録秒)・
# elapsed_sec・duration_sec・t_sec は走行間で同じ値になるべき意味の列で、除くと比較が緩む。
TIMING_KEYS = frozenset(('ready_sec', 'over_budget', 'ms', 'monotonic', 'timings', 'cpu_sec'))
TIMING_SUFFIXES = ('_ms', '_at')
UPDATE_FILES = dict(b3='b3_updates.jsonl', v49f='v49f_updates.jsonl', ama='ama_updates.jsonl')
SAMPLE_LIMIT = 20


def truth_starts(chains: Iterable[dict]) -> dict[int, list[float]]:
    """実連鎖の開始と断片の開始を側ごとに並べる (公開 trigger は断片開始のこともある)。"""
    starts: dict[int, list[float]] = {side: [] for side in SIDES}
    for chain in chains:
        starts[chain['side']].append(chain['start'])
        starts[chain['side']].extend(part['start'] for part in chain.get('parts', ()))
    return {side: sorted(set(values)) for side, values in starts.items()}


def published_fires(display: list[dict], indexed: dict) -> list[dict]:
    """公開票ごとに、載っている X の (側, trigger) を取り出す。"""
    rows = []
    for publication in display:
        pair = indexed.get(publication['vf'].get('completed_at'))
        fires: dict[int, float] = {}
        if pair is not None and pair[1] is not None:
            row, base = pair
            for side in SIDES:
                trigger = base['chain_ids'][side]
                if trigger is not None and row['value']['predicted_scores'][side]:
                    fires[side] = trigger
        provisional = {item['side']: item['trigger'] for item in publication.get('x_provisional') or ()}
        rows.append(dict(t=publication['t'], fires=fires, provisional=provisional))
    return rows


def misfire_spans(rows: list[dict], key: str, truth: dict[int, list[float]],
                  window: tuple[float, float]) -> list[dict]:
    """実連鎖の無い発火を、最初に載った公開から載らなくなる公開までの区間として返す。"""
    spans, open_spans = [], {}
    for row in rows:
        for side in SIDES:
            trigger = row[key].get(side)
            current = open_spans.get(side)
            if current is not None and current['trigger'] != trigger:
                current['withdrawn_sec'] = row['t']-current['first_sec']
                spans.append(open_spans.pop(side))
            if trigger is None or side in open_spans or not window[0] <= trigger < window[1]:
                continue
            if any(abs(trigger-t) <= MATCH_RADIUS_SEC for t in truth[side]):
                continue
            open_spans[side] = dict(side=side, trigger=trigger, first_sec=row['t'], withdrawn_sec=None)
    return spans+list(open_spans.values())


def count_misfires(display: list[dict], indexed: dict, chains: list[dict],
                   window: tuple[float, float]) -> dict:
    """M4。確定 X と暫定 X を分けて、件数と取り下げ秒を返す。"""
    from scripts.measure_live_x_timing import percentiles
    rows = published_fires(display, indexed)
    truth = truth_starts(chains)
    result = dict(window=list(window), match_radius_sec=MATCH_RADIUS_SEC,
                  truth_starts={str(s): len(v) for s, v in truth.items()})
    for key in ('fires', 'provisional'):
        spans = misfire_spans(rows, key, truth, window)
        published = {(side, row[key][side]) for row in rows for side in SIDES if side in row[key]}
        withdrawn = [s['withdrawn_sec'] for s in spans if s['withdrawn_sec'] is not None]
        result[key] = dict(published_fires=len(published), misfires=len(spans),
                           never_withdrawn=sum(s['withdrawn_sec'] is None for s in spans),
                           withdrawn_sec=percentiles(withdrawn), spans=spans)
    return result


def recorded_window(matches: list[dict], recording_start: float) -> tuple[float, float]:
    """録画内の試合だけを窓にする (21c は 47 試合中 45)。"""
    inside = [m for m in matches if m['start'] >= recording_start]
    return min(m['start'] for m in inside), max(m['end'] for m in inside)


def semantic(value: Any) -> Any:
    """計時の値だけを除き、盤・得点・確率・例外は残す。"""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items()
                if k not in TIMING_KEYS and not k.endswith(TIMING_SUFFIXES)}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def update_key(kind: str, row: dict) -> tuple:
    if kind == 'ama':
        return (row.get('generation'), row.get('frame'), (row.get('item') or {}).get('index'))
    if kind == 'b3':
        return (row.get('epoch'), row.get('frame'), row.get('accepted'))
    return (row.get('game'), row.get('id', row.get('frame')))


def compare_updates(kind: str, left: Path, right: Path) -> dict:
    """同じ鍵の行を計時以外の全列で比べる。片側だけの行も数える (実時間 run はコマが違う)。"""
    rows = [{update_key(kind, r): semantic(r) for r in jsonl_rows(path)} for path in (left, right)]
    common = rows[0].keys() & rows[1].keys()
    different = sorted((k for k in common if rows[0][k] != rows[1][k]), key=str)
    return dict(left=len(rows[0]), right=len(rows[1]), common=len(common),
                different=len(different), samples=[str(k) for k in different[:SAMPLE_LIMIT]])


def compare_runs(left: Path, right: Path) -> dict:
    """M5。公開行数と B3・ama・v4.9f 更新原票の一致。母数を必ず併記する。"""
    counts = [sum(1 for _ in pickle_stream(path/'spool/publications.pickle')) for path in (left, right)]
    result = dict(publications=dict(left=counts[0], right=counts[1],
                                    increase_fraction=(counts[1]-counts[0])/counts[0] if counts[0] else None))
    for kind, name in UPDATE_FILES.items():
        if (left/name).exists() and (right/name).exists():
            result[kind] = compare_updates(kind, left/name, right/name)
        else:
            result[kind] = dict(missing=[str(p) for p in (left/name, right/name) if not p.exists()])
    return result


def read_diag(name: str) -> Any:
    with gzip.open(DIAG/f'{name}.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    misfire = sub.add_parser('misfire', help='M4 誤発火')
    misfire.add_argument('--run', type=Path, default=SAVED_21C)
    misfire.add_argument('--placement', choices=('wall', 'input'), default='wall')
    misfire.add_argument('--recording-start', type=float, default=715.394343291)
    compare = sub.add_parser('compare', help='M5 副作用')
    compare.add_argument('--left', type=Path, required=True)
    compare.add_argument('--right', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'compare':
        print(json.dumps(compare_runs(args.left, args.right), ensure_ascii=False, indent=2))
        return
    display = load_publications(args.run, args.placement)
    indexed = index_results(jsonl_rows(args.run/'v49f_updates.jsonl'))
    window = recorded_window(read_diag('matches'), args.recording_start)
    print(json.dumps(count_misfires(display, indexed, read_diag('chains'), window),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
