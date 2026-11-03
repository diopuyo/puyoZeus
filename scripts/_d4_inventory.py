"""D4の保存資産と型を読み取り専用で確認する。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.r1_measure_helpers import lines

OUT = Path('logs/d4')
Q = 'q_7gc4TgFig'


def main() -> None:
    """巨大原票は先頭の型だけを表示する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ('measurements',):
        rows = json.loads(Path(f'logs/d3/{name}.json').read_text())
        result[name] = [r for r in rows if r.get('early')][:1]
    for name in ('count_trace.json', 'prefire_audit.json'):
        rows = json.loads((Path('logs/r1/on/renders')/Q/'on'/name).read_text())
        result[name] = rows[:1] if isinstance(rows, list) else {k: v[:1] if isinstance(v, list) else v for k,v in rows.items()}
    for name in ('records', 'followup'):
        suffix = '.jsonl.gz' if name == 'records' else '.observations.jsonl.gz'
        result[name] = next(lines(Path('logs/r1')/name/(Q+suffix)))
    data = np.load(f'logs/r1/on/renders/{Q}/on/display.npz')
    result['display'] = {k: dict(shape=data[k].shape, sample=data[k].tolist()[:2] if data[k].ndim else data[k].tolist()) for k in data.files}
    (OUT/'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
