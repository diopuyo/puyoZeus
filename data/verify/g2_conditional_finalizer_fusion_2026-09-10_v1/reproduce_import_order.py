"""原legalの遅延importと原backendの一致を、更新なしで実際に再現する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any
import fusion as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'prefix_cpu_v1'
DIAG = ROOT.parent / 'g2_inventory_producer_diagnosis_2026-09-09_v1/probe.py'


def main() -> int:
    mode = sys.argv[1]
    assert mode in ('legal_first', 'world_first')
    with F.modules() as (stage1, world), ExitStack() as stack:
        lib = world.G.L.libraries()
        if mode == 'world_first':
            stack.enter_context(world.G.libraries())
        rows = [json.loads(line) for line in (SOURCE / 'directional_history.jsonl').read_text().splitlines()]
        value = next(row['prepared'] for row in rows if row['prepared'] is not None)
        legal = stage1.F.load('_fusion_original_lazy_legal', DIAG).placement_matches
        matches = legal(lib.P.grid(value['before_grid']), lib.P.grid(value['grid']), value['pair'])
        with world.G.libraries() as ctx:
            actual = ctx.backend
        admission = json.loads((SOURCE / 'ADMISSION.json').read_text())['admission']
        expected = next(row['proof']['backend'] for row in admission
            if row.get('stage') == 'conditional_origin_registered')
        result = dict(mode=mode, matches=bool(matches), equal=actual == expected,
            actual=actual, expected=expected, original_updates=0)
        with (ROOT / ('import_' + mode + '_v1.json')).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps(dict(mode=mode, legal_match=bool(matches), backend_equal=actual == expected,
            extra_actual_files=sorted(set(actual['files'])-set(expected['files'])))), flush=True)
        assert bool(matches) and (actual == expected) == (mode == 'world_first')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
