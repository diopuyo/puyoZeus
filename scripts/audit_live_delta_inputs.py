"""既存再投入記録の母数・完了状態・画素入力欠落を読取専用で確認する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(_public_dev_text('<work>/delta/input_inventory.json'))


def inspect(path: Path) -> dict:
    counts: Counter = Counter()
    result = dict(path=str(path), exists=path.exists(), pixel_frames=0)
    if not path.exists():
        return result
    with path.open('rb') as stream:
        result['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                counts[row.get('kind', 'unknown')] += 1
                if row.get('kind') == 'header':
                    result['header'] = row
                if row.get('kind') == 'update' and 'first_update_keys' not in result:
                    result['first_update_keys'] = list(row)
                    result['first_result_keys'] = list(row['args']['tuple'][0]['namespace'])
    except (EOFError, json.JSONDecodeError) as error:
        result['incomplete_reason'] = type(error).__name__
    result['records'] = dict(counts)
    result['recognition_replay_possible'] = False
    result['reason'] = 'ExchangeEventRecorderは認識後の盤面を保存する仕様。画素入力は未記録。'
    return result


def main() -> None:
    from scripts.bench_live_delta import check_resources
    check_resources()
    paths = [Path(f'C:/puyo1_tools/soak/run_{name}/output/inputs.jsonl.gz') for name in 'ghijk']
    paths.append(Path('C:/PuyoLive_20261012e/PuyoLive/output/inputs.jsonl.gz'))
    results = []
    for path in paths:
        check_resources()
        results.append(inspect(path))
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    main()
