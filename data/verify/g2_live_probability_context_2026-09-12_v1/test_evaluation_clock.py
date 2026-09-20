"""原joint入力の時刻契約を再現する。実J鮮度の統合合格には流用しない。"""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'))
from test_joint_evaluation import states
import joint_evaluation as J


def test_static_basis_can_be_evaluated_later() -> None:
    values, observed = states()
    assert all(value.frame == 10 for value in values)
    J.inputs(values, 12, ('STABLE', 'STABLE'), observed)
    assert all(value.frame != 12 for value in values)


@pytest.mark.parametrize('frame', (9, 31))
def test_future_basis_and_expired_evaluation_rejected(frame: int) -> None:
    values, observed = states()
    with pytest.raises(ValueError, match='joint_clock'):
        J.inputs(values, frame, ('STABLE', 'STABLE'), observed)


def test_changed_visible_board_rejected() -> None:
    values, observed = states()
    changed = J.B.Board.from_dict({'grid': J.B.grid(observed[0])})
    changed.set(12, 5, 1)
    with pytest.raises(ValueError, match='joint_current_visible_mismatch'):
        J.inputs(values, 12, ('STABLE', 'STABLE'), (changed, observed[1]))
