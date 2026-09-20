"""同call確率所有権・別採録の接続を正常105更新へ追加し非干渉を検査する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import proof_cpu_basis as OLD
import inflight_loader_v2 as L
import basis_binding_loader as B

TRACKING_END_FRAME = 36298  # 対象固定run上限。基準取得14frame期限は変更しない。


class Context(OLD.Context):
    def perform(self, advisory: Any, frame: int, clock: float) -> None:
        super().perform(advisory, frame, clock)
        observer = self.state['settled_basis_observer']
        B.connection().install(self.stack, self.recovery, self.state, observer, TRACKING_END_FRAME)


def main() -> int:
    basis = L.ROOT.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1'
    paths = [L.ROOT / n for n in ('proof_cpu_binding.py', 'basis_binding_loader.py',
        'inflight_loader_v2.py', 'quarantine_v2.py')]
    paths += [basis / n for n in ('gate_v3.py', 'actual_connection_v2.py', 'basis_registry_connection.py')]
    root = L.ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'
    paths += [root / n for n in ('belief.py', 'frozen_physics.py', 'conditioning.py', 'registry.py', 'serialization.py')]
    sha = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = sha()
    OLD.R.C, OLD.basis_connection, OLD.R.Context = L.CONNECTION, L.basis_connection, Context
    code = OLD.R.main()
    unchanged = before == sha()
    output = OLD.R.P.R.Q.ROOT / sys.argv[1]
    with (output / 'BASIS_BINDING_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(source=before, unchanged=unchanged, original_exit=code,
                       quality_gate_clear=False), stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
