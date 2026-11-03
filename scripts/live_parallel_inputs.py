"""指定済み記録の母数・末尾・画像の有無を確認する。再認識とは呼ばない。"""
from __future__ import annotations

from collections import Counter
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [Path(f'C:/puyo1_tools/soak/run_{run}/output/inputs.jsonl.gz')
           for run in ('g', 'h', 'i', 'j', 'k')]
SOURCES.append(Path('C:/PuyoLive_20261012e/PuyoLive/output/inputs.jsonl.gz'))


def inspect_input(path: Path) -> dict[str, Any]:
    """欠損・未完了を隠さず、完全に読めたJSON行だけ数える。"""
    if not path.exists():
        return dict(path=str(path), exists=False)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    kinds: Counter = Counter()
    fields = set()
    error = None
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                kinds[row.get('kind')] += 1
                fields.update(row)
    except (EOFError, OSError, ValueError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    return dict(path=str(path), exists=True, sha256=digest.hexdigest(),
        kinds=kinds, fields=sorted(fields), error=error,
        complete=bool(kinds['complete']), recognition_images=False,
        evidence='src/exchange_event_record.pyの保存仕様は認識画像・logitsを含まない')


def main() -> None:
    sys.path.insert(0, str(ROOT/'scripts'))
    from live_parallel_probe import guard
    guard()
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    rows = []
    for path in SOURCES:
        guard()
        rows.append(inspect_input(path))
        print(path.parent.parent.name, rows[-1].get('kinds'), flush=True)
    (ROOT/'logs/live_parallel/input_inventory.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
