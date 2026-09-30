"""基準の profile から、遅い prove_multilanding 呼出し上位と遅い通知上位を表にする (入力の盤面は出さない)。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOP_N = 20
PROVE_FIELDS = ('source', 'sec', 'fill', 'incoming', 'hands', 'queue_len', 'nodes', 'reason', 'dead',
                'rounds', 'responses_max', 'responses_sum')
NOTE_FIELDS = ('source', 't_sec', 'sec', 'post_counter', 'weighted', 'prove', 'future_send')


def rows_of(root: Path) -> dict[str, list[dict]]:
    return {p.stem.removeprefix('profile_'): [json.loads(l) for l in p.read_text().splitlines()]
            for p in sorted(root.glob('profile_*.jsonl'))}


def top_prove(data: dict[str, list[dict]]) -> list[dict]:
    calls = [dict(r, source=s) for s, rows in data.items() for r in rows if r['fn'] == 'prove_multilanding']
    return sorted(calls, key=lambda r: -r['sec'])[:TOP_N]


def notifications(data: dict[str, list[dict]]) -> list[dict]:
    """projection_evaluate 1 行 = 通知 1 件。直前までの子呼出しを合算して内訳を付ける。"""
    out = []
    for source, rows in data.items():
        children: list[dict] = []
        for r in rows:
            if r['fn'] != 'projection_evaluate':
                children.append(r)
                continue
            total = lambda fn: sum(c['sec'] for c in children if c['fn'] == fn)
            out.append(dict(source=source, t_sec=r['t_sec'], sec=r['sec'], post_counter=total('post_counter_evaluate'),
                            weighted=total('weighted_landing'), prove=total('prove_multilanding'),
                            future_send=total('future_send_miss')))
            children = []
    return out


def write(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    lines = ['\t'.join(fields)]
    for r in rows:
        lines.append('\t'.join(f'{r[f]:.3f}' if isinstance(r[f], float) else str(r[f]) for f in fields))
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    root = Path(sys.argv[1])
    data = rows_of(root)
    write(root / 'TOP20_prove.tsv', top_prove(data), PROVE_FIELDS)
    write(root / 'TOP20_notifications.tsv', sorted(notifications(data), key=lambda r: -r['sec'])[:TOP_N], NOTE_FIELDS)


if __name__ == '__main__':
    main()
