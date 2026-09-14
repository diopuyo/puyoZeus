"""原105更新に隠し基準と同call過去票を接続する人工入力限定検査。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
QROOT = ROOT.parent / 'g2_reset_inflight_quarantine_2026-09-11_v1'
VROOT = ROOT.parent / 'g2_transition_endpoint_votes_2026-09-11_v1'
sys.path.insert(0, str(QROOT))
import proof_cpu_target as OLD  # 原driverがtorchをfactory隔離前に読み込む。
import hidden_loader as H

ORIGINAL_VERIFY = OLD.verify


def votes() -> Any:
    load = H.OLD.L.load
    endpoint = load('_g2_past_votes_base', VROOT / 'endpoint_votes.py')
    prior = load('_g2_past_votes_v2', VROOT / 'prior_votes_v2.py', {'endpoint_votes': endpoint})
    return load('_g2_past_votes_qualified', VROOT / 'qualified_connection.py', {'prior_votes_v2': prior})


class Context(OLD.Context):
    def perform(self, advisory: Any, frame: int, clock: float) -> None:
        super().perform(advisory, frame, clock)
        deadline = sys.modules[type(self.recovery).__module__].I.DEADLINE
        votes().install(self.stack, self.recovery, self.state, frame, deadline)


def verify(recovery: Any, lease: Any, factory: Any, state: Any, trace: Any, prior: Any) -> Any:
    value = state['qualified_prior_votes']
    assert value.failure is None and value.rows, 'qualified_votes_nonempty_required'
    assert value.allowed.error is None and any(row['eligible'] for row in value.allowed.rows)
    result = ORIGINAL_VERIFY(recovery, lease, factory, state, trace, prior)
    result.update(qualified_votes=len(value.rows), hidden_basis_connected=True)
    return result


def main() -> int:
    paths = [path for root in (ROOT, VROOT) for path in root.glob('*.py')]
    digest = lambda: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    before = digest()
    OLD.L, OLD.Context, OLD.verify = H, Context, verify
    code = OLD.main()
    unchanged = before == digest()
    output = OLD.DRIVER.P.R.Q.ROOT / sys.argv[1]
    with (output / 'HIDDEN_PRIOR_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(source=before, unchanged=unchanged, original_exit=code,
                       quality_gate_clear=False), stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
