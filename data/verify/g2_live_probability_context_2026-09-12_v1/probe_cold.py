"""原cold srcを保持した依存解決のみ。factory/update/候補評価は実行しない。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sources() -> dict:
    return {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.items()
            if (n == 'src' or n.startswith('src.')) and getattr(m, '__file__', None)}


def main() -> None:
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location('_live_probability_cold_entry',
                                                REPO / 'scripts/diagnose_video38_confirmed_collapse_v1.py')
    cold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cold)
    cold.load_collector()
    before = sources()
    import loader as L
    parts = L.dependencies()
    resolved = parts.modules()
    value = L.build(object, end_frame=36298)
    assert before == sources(), 'cold_src_changed'
    assert 'anchor_v2' not in sys.modules and 'context' not in sys.modules, 'reserved_name_occupied'
    proof = dict(src_count=len(before), frozen_src_unchanged=True, class_composed=issubclass(value, object),
                 mode_class=resolved.mode.Mode.__module__, factory_created=False, original_updates=0,
                 actual_video=False, quality_gate_clear=False)
    with (ROOT / 'COLD_DEPENDENCIES_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
