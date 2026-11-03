"""実配送列から入力可能性とツモ履歴を監査する。未来・補間観測は使わない。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase_j.live_ama_inputs import AmaInputs
from src.ama_tsumo_seed_v5 import load_table
import puyo_core


def main() -> None:
    source, destination = map(Path, sys.argv[1:])
    table = load_table(Path(__file__).resolve().parents[1]/'models/ama_plain/seed_table_cache.npz')
    identity, owner, groups = None, None, []
    for line in source.read_text(encoding='utf-8').splitlines():
        packet = json.loads(line)
        current = packet['generation'], packet['game']
        if identity != current:
            owner, identity = AmaInputs(puyo_core, table), current
            group = dict(identity=current, rows=0, requests=0, valid=0, reasons=Counter())
            groups.append(group)
        item = owner.observe(packet)
        group['rows'] += 1
        group['requests'] += item is not None
        group['valid'] += item is not None and not item.get('missing')
        group['reasons'][item.get('missing', 'valid') if item else owner.missing_reason or 'unchanged'] += 1
        group['tape'] = owner.tape.tape.summary()
    destination.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(groups, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
