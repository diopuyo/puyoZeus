"""c5原票の変化点を端末とJSONへ簡潔に表示する。"""
from __future__ import annotations

import json
from scripts._diag_c5_inspect import BASE, OUT, guard


def main() -> None:
    """式・得点・予告・盤面枚数の変化を確認する。"""
    guard()
    rows = json.loads((OUT / 'input_changes.json').read_text())
    previous = None
    for row in rows:
        value = (row['formula'], row['displayed'], row['snapshot'],
                 [s['state'] for s in row['sides']])
        if value != previous:
            print(round(row['t'], 3), value)
            previous = value
    print('BOARDS')
    for row in json.loads((OUT / 'stable_boards.json').read_text()):
        cells = sum(row['board'], [])
        print(round(row['t'], 3), row['side'], 'colored', sum(1<=c<=5 for c in cells), 'ojama', cells.count(9))
    path = OUT / 'first_proof_inputs.json'
    if path.exists():
        data = json.loads(path.read_text())
        print('PROOF', json.dumps({k:v for k,v in data.items() if k not in ('context','latest','safety')}, ensure_ascii=False))
        print('SAFETY', json.dumps(data['safety'], ensure_ascii=False))
        print('CONTEXT', json.dumps(data['context'], ensure_ascii=False))


if __name__ == '__main__':
    main()
