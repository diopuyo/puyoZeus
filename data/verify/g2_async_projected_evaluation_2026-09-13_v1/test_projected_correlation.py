"""投影後のcell相関と、現況Beliefへの誤投入拒否を検査する。"""
import pytest
import belief as B
import joint_evaluation as J
from test_joint_evaluation import states
import projected_view as P


def test_projection_preserves_future_visible_and_hidden_joint_support() -> None:
    values, observed = states()
    view = P.project(values[0], values[0].scope, observed[0], origin_frame=10,
        cutoff_frame=12, origin_token='artificial-source-origin')
    actual = {(o.grid[4][0], o.grid[0][1]): o.weight for o in view.outcomes}
    assert actual == pytest.approx({(4, 5): 0.3, (5, 4): 0.7})
    assert (4, 4) not in actual and (5, 5) not in actual
    assert view.hidden_joint_support_preserved


def test_projected_dto_cannot_impersonate_current_belief() -> None:
    values, observed = states()
    view = P.project(values[0], values[0].scope, observed[0], origin_frame=10,
        cutoff_frame=12, origin_token='artificial-source-origin')
    with pytest.raises(ValueError, match='belief_type'):
        B.validate(view)
    with pytest.raises(ValueError, match='belief_type'):
        J.inputs((view, values[1]), 12, ('STABLE', 'STABLE'), observed)
