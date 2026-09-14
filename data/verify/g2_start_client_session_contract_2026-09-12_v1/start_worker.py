"""専用子の開始資格だけ拡張し、原wire/モデル/保存/終了を使う。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT.parent / 'g2_joint_ledger_engine_2026-09-12_v1'
PINS = {"start_join.py":"aa66abc3581f3e4d45b9c69fc5bcb457f3849edef4b129c759f8a80facdbb647","start_qualification.py":"51b4e4c8ce978a49121681fcc13f3ab6cdb55afa9a2f9bdf7b029be8d548b2d1"}


def configure() -> None:
    for name, expected in PINS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'start_worker_source'
    sys.path[:0] = [str(LEDGER), str(ROOT), str(ROOT.parents[2])]
    import engine as engine
    engine.guard()
    import start_join as joined
    import start_qualification as qualification
    assert joined.J is engine.J, 'start_worker_original_join'
    pins = dict(engine.PINS)
    for module in (joined, qualification):
        path = Path(module.__file__).resolve()
        pins[str(path.relative_to(ROOT.parent))] = PINS[path.name]
    engine.PINS, engine.J = pins, joined
    engine.guard()


if __name__ == '__main__':
    configure()
    runpy.run_path(str(LEDGER / 'worker_v2.py'), run_name='__main__')
