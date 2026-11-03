"""着弾履歴は受け側盤面のおじゃま純増がある区間だけを数える。"""
import numpy as np
import pytest

from src.b3_features import land_group
from src.b3_state import LAND_CLOSE_SEC, SideState

BEFORE_COUNT = 2
VALID_INCREASE = 6
START_SEC = 10.0
INTERVAL_SEC = 3.0


def _frame(count: int | None, state: str) -> dict:
    board = None
    if count is not None:
        board = np.zeros((13, 6), dtype=np.int8)
        board.ravel()[:count] = 9
    return {"state": state, "confirmed_board": board, "score": 0}


@pytest.mark.parametrize("net", [-2, 0, VALID_INCREASE])
@pytest.mark.parametrize("prior_landing", [False, True])
def test_landing_requires_positive_board_increase(net: int, prior_landing: bool) -> None:
    st = SideState("1P")
    if prior_landing:
        st.update_frame(0.0, _frame(0, "OJAMA_FALL"), True)
        st.update_frame(LAND_CLOSE_SEC, _frame(VALID_INCREASE, "STABLE"), True)
    previous = st.landing
    st.update_frame(START_SEC, _frame(BEFORE_COUNT, "OJAMA_FALL"), True)
    st.update_frame(START_SEC + LAND_CLOSE_SEC, _frame(BEFORE_COUNT + net, "STABLE"), True)
    assert st.fall_start is None
    assert st.land_count == int(prior_landing) + int(net > 0)
    if net > 0:
        assert st.landing.board_net == net
        assert st.landing.finalize_t == START_SEC + LAND_CLOSE_SEC
    else:
        assert st.landing is previous
        row: dict = {}
        land_group(row, st, "p1", START_SEC + LAND_CLOSE_SEC)
        if previous is None:
            assert np.isnan(row["land_board_net_p1"])
        else:
            assert row["land_board_net_p1"] == previous.board_net
    # 無効区間の後も次の正しい着弾を取りこぼさない。
    st.update_frame(START_SEC + INTERVAL_SEC, _frame(0, "OJAMA_FALL"), True)
    st.update_frame(START_SEC + INTERVAL_SEC + LAND_CLOSE_SEC, _frame(VALID_INCREASE, "STABLE"), True)
    assert st.land_count == int(prior_landing) + int(net > 0) + 1
    assert st.landing.board_net == VALID_INCREASE


def test_unknown_board_does_not_establish_landing() -> None:
    st = SideState("1P")
    st.update_frame(START_SEC, _frame(None, "OJAMA_FALL"), True)
    st.update_frame(START_SEC + LAND_CLOSE_SEC, _frame(VALID_INCREASE, "STABLE"), True)
    assert st.land_count == 0 and st.landing is None and st.fall_start is None
