"""元片側検査とv68原票で0回契約を確認。保存後の値を実動作証明に補完しない。"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

VERIFY = Path(__file__).resolve().parent.parent
BASE = VERIFY / 'g2_empty_tail_reset_integration_2026-09-11_v1'
SOURCE = VERIFY / 'g2_hidden_basis_initialization_2026-09-11_v1/side_actual_connection.py'
sys.path.insert(0, str(BASE / 'qa_retired_v1'))
from test_retired import carrier
import retired_completion as R
import probability_owner as P


def original_verify() -> Any:
    tree = ast.parse(SOURCE.read_bytes())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')
    namespace = dict(Any=Any)
    exec(compile(ast.Module([node], []), str(SOURCE), 'exec'), namespace)
    return namespace['verify']


@pytest.fixture(scope='module')
def saved() -> Any:
    return json.loads((BASE / 'prefix_cpu_v68_owner_finish/SIDE_ACTUAL_CONNECTION.json').read_bytes())


def test_saved_side_reset_requires_no_whole_reset(saved: Any) -> None:
    assert saved['stage'] == 'side_reset_connected' and saved['full_reset_calls'] == 0
    original_verify()(saved['before'], saved['after'], N(outer_calls=0, native_calls=0, depth=0))
    waiting, = [row for row in saved['lease_events'] if row['kind'] == 'side_reset_waiting']
    assert waiting['outer_calls'] == waiting['native_calls'] == 0


def test_saved_qualified_archive_matches_new_side_contract(saved: Any) -> None:
    status = json.loads((BASE / 'prefix_cpu_v68_owner_finish/PROBABILISTIC_TRACKING_STATUS.json').read_bytes())
    waiting, = [row for row in saved['lease_events'] if row['kind'] == 'side_reset_waiting']
    expected = status['activation']['prior_pending']['empty_retirement']['archive_sha256']
    lease = N(events=saved['lease_events'], new_scope=tuple(waiting['scope']),
              outer_calls=0, native_calls=0, depth=0, empty_evidence=N(receipt_sha=expected))
    P.side_reset_authority(R.E, lease)


@pytest.mark.parametrize('field', ('outer_calls', 'native_calls', 'depth'))
def test_original_side_guard_rejects_whole_reset_counter(saved: Any, field: str) -> None:
    counts = dict(outer_calls=0, native_calls=0, depth=0)
    counts[field] = 1
    with pytest.raises(AssertionError, match='^actual_full_reset_called$'):
        original_verify()(saved['before'], saved['after'], N(**counts))


def test_old_integer_owner_rejects_side_counter_at_exact_assertion() -> None:
    lease, factory, old, state, key = carrier()
    lease.outer_calls = lease.native_calls = 0
    with pytest.raises(AssertionError) as failure:
        R.retired_owner(lease, factory, old, state, state, key)
    last = failure.value.__traceback__
    while last.tb_next is not None:
        last = last.tb_next
    tree = ast.parse(Path(R.__file__).read_bytes())
    node, = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
             and ast.unparse(node.test) == 'lease.outer_calls == lease.native_calls == 1']
    assert last.tb_frame.f_code is R.retired_owner.__code__ and last.tb_lineno == node.lineno
