"""D2の最初の分岐を、巨大配列と区別した小さい原票へまとめる。"""
from __future__ import annotations

import json
import gzip
import hashlib
from typing import Any

from scripts._d2_compare import lines
from scripts._d2_inventory import OUT, save

STAMP = 1.8


def repeat_hashes() -> dict:
    """gzipの作成時刻を除き、収集原票本文全体の一致を確認する。"""
    output = {}
    for left, right in (('q_new', 'q_new2'), ('q_sig', 'q_sig2')):
        hashes = []
        for name in (left, right):
            with gzip.open(OUT/name/'inputs.jsonl.gz', 'rb') as stream:
                hashes.append(hashlib.file_digest(stream, 'sha256').hexdigest())
        output[f'{left}:{right}'] = dict(sha256=hashes, equal=hashes[0] == hashes[1])
    return output


def source_first_frames() -> dict:
    """各動画の最初の確定盤面差で観測と内部状態を比較する。"""
    audit = json.loads((OUT/'original_board_audit.json').read_text())
    output = {}
    sources = dict(q='q_7gc4TgFig', fc='fcXG83vInDY', mia='mia8KCjr52g',
                   zenchi='zenchi', review='review')
    for name, source in sources.items():
        names = (f'{name}_sig', f'{name}_new') if name != 'q' else ('q_sig2', 'q_new2')
        if not all((OUT/n/'complete.json').exists() for n in names):
            continue
        stamp = audit[source]['e31_to_e34b']['first']
        traces = [lines(OUT/n/'trace.jsonl.gz') for n in names]
        offsets = [next(i for i, r in enumerate(t) if r['t_sec'] == stamp) for t in traces]
        rows = [t[i] for t, i in zip(traces, offsets)]
        output[source] = dict(t_sec=stamp, rows=rows,
            previous=[t[i-1] for t, i in zip(traces, offsets)],
            equal={k: all(a[k] == b[k] for a, b in zip(rows[0]['sides'], rows[1]['sides']))
                   for k in ('cnn_board', 'raw_cnn_board', 'hsv_board', 'state', 'confirmed_board')})
    return output


def reset_events() -> dict:
    """再同期resetの発生フレームと実装行を保存する。"""
    output = {}
    for name in ('q_reset', 'q_reset_old'):
        if not (OUT/name/'complete.json').exists():
            continue
        output[name] = [dict(t_sec=r['t_sec'], frame_idx=r['frame_idx'], resets=r['resets'])
                        for r in lines(OUT/name/'trace.jsonl.gz') if r.get('resets')]
    return output


def count(board: Any) -> int | None:
    """UNKNOWNを含む非空セル数を返す。"""
    if board is None:
        return None
    return sum(v != 0 for row in board['board']['array'] for v in row)


def main() -> None:
    """最初の分岐と反復のビット一致を簡潔に保存する。"""
    first = {}
    for name in ('q_new2', 'q_sig2', 'q_hsv', 'q_start'):
        root = OUT/name
        if not (root/'complete.json').exists():
            continue
        row = next(r for r in lines(root/'trace.jsonl.gz') if r['t_sec'] == STAMP)
        first[name] = dict(t_sec=row['t_sec'], frame=row['frame_idx'],
            internal=row.get('internal'), sides=[dict(state=s['state'], drift=s.get('drift'),
                counts={k: count(s[k]) for k in
                        ('cnn_board', 'raw_cnn_board', 'hsv_board', 'confirmed_board')})
                for s in row['sides']])
    comparison = json.loads((OUT/'comparisons.json').read_text())
    brief = {name: dict(rows=entry['e31']['rows'],
             old=entry['e31']['per_field'], new=entry['e34b']['per_field'])
             for name, entry in comparison['runs'].items()}
    pairs = {name: {k: v for k, v in value.items() if k != 'first'}
             for name, value in comparison['pairs'].items()}
    value = dict(first_frame=first, runs=brief, pairs=pairs, repeat_hashes=repeat_hashes())
    save('summary.json', value)
    save('first_frames.json', source_first_frames())
    save('reset_events.json', reset_events())
    print(json.dumps(dict(runs=len(brief), repeat_hashes=value['repeat_hashes']), indent=2))


if __name__ == '__main__':
    main()
