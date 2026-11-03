"""行単位の認識追跡をセル単位の書込み履歴へ変換する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._diag_set2_falsedeath_extract import OUT

MAX_CONTIGUOUS_GAP = 0.1
WRITES = ((13, 'p2', 4417.366667), (15, 'p2', 4597.333333),
          (43, 'p2', 6171.933333), (45, 'p1', 6308.6), (58, 'p2', 7002.516667))


def state_evidence() -> list[dict]:
    """外側のsm.update書込みを、遷移とSTABLE内復旧へ区別する状態原票。"""
    rows = []
    for game, side, stamp in WRITES:
        inputs = json.loads((OUT / f'game{game:02d}_inputs.json').read_text())
        values = []
        for value in inputs:
            args = value['args']['tuple']
            if stamp - MAX_CONTIGUOUS_GAP <= args[3] <= stamp + MAX_CONTIGUOUS_GAP:
                current = args[0]['namespace'][side]['namespace']
                values.append(dict(t_sec=args[3], state=current['state']['state'],
                                   board=current['confirmed_board']))
        rows.append(dict(game=game, side=side, write_sec=stamp, values=values))
    return rows


def cells(path: Path) -> list[dict]:
    """追跡窓の間の空白を、窓先頭の書込みと誤認しない。"""
    latest, rows = {}, []
    for line in path.read_text().splitlines():
        value = json.loads(line)
        side, stamp = value['side'], value['t_sec']
        previous = latest.get(side)
        before = value['before']
        if before is None and previous and stamp - previous[0] <= MAX_CONTIGUOUS_GAP:
            before = previous[1]
        if before is not None:
            for row, (left, right) in enumerate(zip(before, value['after'])):
                for col, (old, new) in enumerate(zip(left, right)):
                    if old != new:
                        rows.append(dict(t_sec=stamp, side=side, state=value['state'],
                            line=value['previous_line'], row=row, col=col, before=old, after=new))
        latest[side] = (stamp, value['after'])
    return rows


def main() -> None:
    """書込みセルと出力一致検査を一つの索引へまとめる。"""
    result = {}
    for part in ('s2', 's5', 's6'):
        path = OUT / 'recognition' / f'{part}_writes.jsonl'
        if not path.exists():
            continue
        rows = cells(path)
        (OUT / f'{part}_write_cells.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
        identity = path.with_name(f'{part}_identity.json')
        if identity.exists():
            raw = json.loads(identity.read_text())
            result[part] = dict(compared=raw['compared'], mismatch_count=len(raw['mismatches']),
                target_rows=len(raw['target_rows']), target_mismatches=sum(not r['identical'] for r in raw['target_rows']))
    (OUT / 'RECOGNITION_IDENTITY.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    (OUT / 'WRITE_STATE_EVIDENCE.json').write_text(json.dumps(state_evidence(), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
