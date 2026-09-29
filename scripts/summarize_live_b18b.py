"""E31画像窓・E32採否とライブ通知原票を同じ発火キーで照合する。"""
from __future__ import annotations

from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

from src.exchange_event_record import read_records
from scripts.verify_live_b18b import OUT, save
from scripts.verify_live_b18a import compare_rows
from scripts.run_live_pipeline_20260928 import compare_arrays

STAMP_DIGITS = 6
DISPLAY_COLUMNS = ('t_sec', 'game_idx', 'display_adv', 'display_p1', 'source')

def fire_key(row: dict) -> tuple:
    return row['game'], row['side'], round(row['trigger_sec'], STAMP_DIGITS)


def captured_windows(record: Path) -> dict:
    values = {}
    for row in read_records(record):
        if row['kind'] != 'update':
            continue
        result, game = row['args'][0], row['args'][4]
        for idx, side in enumerate((result.p1, result.p2)):
            if side.chain_event is not None:
                key = (game, f'{idx+1}P', round(side.chain_event.trigger_sec, STAMP_DIGITS))
                values.setdefault(key, getattr(side, 'prefire_snapshot', None))
    return values


def reference_windows(source: str) -> dict:
    with gzip.open(f'logs/e31/records/{source}.jsonl.windows.json.gz', 'rt') as stream:
        rows = json.load(stream)
    values = {}
    for row in rows:
        values[(row['game'], f"{row['side']+1}P", round(row['trigger_sec'], STAMP_DIGITS))] = {
            k: v for k, v in row.items() if k not in ('raw_frames', 'game', 'side', 'trigger_sec')}
    return values


def compare_windows(source: str, dest: Path) -> dict:
    old, new = reference_windows(source), captured_windows(dest/'inputs.jsonl.gz')
    same, boards, reasons = 0, 0, 0
    mismatches = []
    for key, expected in old.items():
        actual = new.get(key)
        same += actual == expected
        boards += actual is not None and actual.get('board') == expected.get('board')
        reasons += actual is not None and actual.get('reason') == expected.get('reason')
        if actual != expected:
            mismatches.append(dict(key=key, expected=expected, actual=actual))
    return dict(total=len(old), exact=same, boards=boards, reasons=reasons,
                missing=sum(k not in new for k in old), mismatches=mismatches)


def compare_acceptance(source: str, dest: Path) -> dict:
    old = json.loads(Path(f'logs/e32/on/{source}/snapshot_final_audit.json').read_text())['rows']
    new = json.loads((dest/'prefire_audit.json').read_text())['rows']
    actual = {fire_key(r): r for r in new}
    count, boards, accepted, hidden = 0, 0, 0, 0
    mismatches = []
    for expected in old:
        key = fire_key(expected)
        row = actual.get(key)
        decision = row is not None and all(row[k] == expected[k] for k in ('accepted', 'reason'))
        count += decision
        if expected['accepted']:
            accepted += 1
            boards += bool(row and row['accepted'] and row['snapshot']['board'] == expected['snapshot']['board'])
            hidden += bool(row and row.get('hidden_distributions') == expected.get('hidden_distributions'))
        if not decision or (expected['accepted'] and not (row and row['accepted'] and
                row['snapshot']['board'] == expected['snapshot']['board'])):
            mismatches.append(dict(key=key, expected=expected, actual=row))
    return dict(total=len(old), decisions=count, accepted_total=accepted, accepted_boards=boards,
        new_total=len(new), new_accepted=sum(r['accepted'] for r in new),
        missing=sum(fire_key(r) not in actual for r in old),
        extra=len(set(actual)-{fire_key(r) for r in old}),
        hidden_distributions=hidden, old_reasons=dict(Counter(str(r['reason']) for r in old)),
        new_reasons=dict(Counter(str(r['reason']) for r in new)), mismatches=mismatches)


def complete_display_trace(path: Path) -> list[dict]:
    """旧B18aの勝率原票にも、同じ本番EMAを因果順に適用して表示列を補う。"""
    from types import SimpleNamespace
    from src.phase_j.live_notification_eval import _ExchangeDisplayEMA, _exchange_display
    rows = json.loads(path.read_text())
    smoothing = _ExchangeDisplayEMA()
    for row in rows:
        if 'display' not in row:
            overlay = SimpleNamespace(tracker=SimpleNamespace(probability=row['p1'], source=row['source']))
            row['display'] = list(_exchange_display(overlay, 0., .5, smoothing, row['t_sec']))
    return rows


def compare_original(source: str, dest: Path) -> dict:
    """新通知の再生一致に加え、認識し直す前のE31原票との差も隠さず数える。"""
    reference = OUT/'saved'/source/'offline/probabilities.json'
    if source == 'review' and not reference.exists():
        reference = Path('logs/live_b18a/offline/probabilities.json')
    old, new = complete_display_trace(reference), complete_display_trace(dest/'probabilities.json')
    probability = lambda rows: [{k: v for k, v in r.items() if k != 'display'} for r in rows]
    return dict(probability=compare_rows(probability(old), probability(new)),
                probability_and_ema=compare_rows(old, new),
                published_display=compare_original_display(source, dest))


def compare_legacy_schema(source: str) -> dict | None:
    """before_boardだけを旧原票へ揃えた、原因切り分け専用再生を照合する。"""
    path = OUT/'schema'/source/'probabilities.json'
    if not path.exists():
        return None
    reference = OUT/'saved'/source/'offline/probabilities.json'
    old, new = complete_display_trace(reference), complete_display_trace(path)
    probability = lambda rows: [{k: v for k, v in r.items() if k != 'display'} for r in rows]
    return dict(probability=compare_rows(probability(old), probability(new)),
                probability_and_ema=compare_rows(old, new))


def compare_original_display(source: str, dest: Path) -> dict:
    """公開した時刻をE31の全表示行へ突き合わせ、欠測時の表示も比較する。"""
    with np.load(OUT/'saved'/source/'offline/display.npz') as old, np.load(dest/'display.npz') as new:
        positions = {stamp: idx for idx, stamp in enumerate(old['t_sec'])}
        expected, actual = [], []
        for idx, stamp in enumerate(new['t_sec']):
            before = positions.get(stamp)
            expected.append(dict(missing=True, t_sec=float(stamp)) if before is None else {
                key: old[key][before].item() for key in DISPLAY_COLUMNS})
            actual.append({key: new[key][idx].item() for key in DISPLAY_COLUMNS})
    return compare_rows(expected, actual)


def summarize(source: str, directory: Path | None = None) -> dict:
    dest = directory or OUT/'video'/source
    display = compare_arrays(dest/'replay/offline/display.npz', dest/'display.npz')
    result = dict(source=source, windows=compare_windows(source, dest),
        acceptance=compare_acceptance(source, dest), video_display=display,
        video_trace=json.loads((dest/'comparison.json').read_text()),
        original_record=compare_original(source, dest),
        legacy_schema=compare_legacy_schema(source),
        replay=json.loads((dest/'replay/comparison.json').read_text()),
        cost=json.loads((dest/'cost.json').read_text()))
    save(dest/'report.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('review', 'zenchi'))
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.directory and not args.source:
        parser.error('--directoryには--sourceが必要')
    reports = [summarize(source, args.directory)
               for source in ((args.source,) if args.source else ('review', 'zenchi'))]
    save(OUT/'report.json', reports)
    for row in reports:
        print(json.dumps({key: {k: v for k, v in value.items() if k != 'mismatches'}
            if isinstance(value, dict) else value for key, value in row.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
