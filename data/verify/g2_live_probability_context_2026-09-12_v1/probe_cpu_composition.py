"""既存最終CPU生成器の実Context継承元を読む。factoryやupdateは作らない。"""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'))
import probe_imports as I


def main() -> None:
    runtime = I.R.V6.V5.V4.OLD
    hand = runtime.OLD.H
    configured = runtime.configure(hand.P.OLD.configure, hand.P.configure)
    hidden = configured.OLD.BASE.H
    rows = []
    for cls in hidden.Context.__mro__:
        method = vars(cls).get('perform')
        rows.append(dict(name=cls.__qualname__, module=cls.__module__,
                         perform_source=None if method is None else inspect.getsourcefile(method),
                         perform_line=None if method is None else method.__code__.co_firstlineno))
    packet = dict(context_mro_before_main_anchor=rows,
                  finalizer_entry=str(Path(configured.__file__).resolve()),
                  actual_updates=0, factory_created=False, quality_gate_clear=False)
    with (ROOT / 'CPU_COMPOSITION_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(packet, stream, indent=2)
    print(json.dumps(packet))


if __name__ == '__main__':
    main()
