"""原生成器上の差替と遅延解決を確認する。元実更新は走らせない。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import core_fixture_v2 as C

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'))
import probe_imports as I


def main() -> None:
    runtime = I.R.V6.V5.V4.OLD
    hand = runtime.OLD.H
    refs: dict = {}
    result = C.replace(runtime.configure(hand.P.OLD.configure, hand.P.configure), refs)
    current = result.OLD.BASE.H.Context
    mro = current.__mro__
    assert refs['deferred'].parts is None and refs['core'] is mro[4], 'not_lazy_or_wrong_core'
    assert mro[5] is refs['original'].__mro__[-2] and len(mro) == 7, 'wrong_original_parent'
    assert all(c.__module__ not in ('proof_cpu_hidden', 'proof_cpu_target', 'proof_cpu_connected') for c in mro)
    value = dict(mro=[dict(module=c.__module__, name=c.__qualname__) for c in mro],
                 dependencies_resolved=False, original_parent_retained=True, main_anchor_not_added_yet=True,
                 original_updates=0, factory_created=False, quality_gate_clear=False)
    with (ROOT / 'CORE_FIXTURE_PREFLIGHT_v2.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
