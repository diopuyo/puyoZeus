"""予告の主入力を得点台帳へ切り替える3つの独立フラグを検証する。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import production_config as config
from src.b3_features import mode_group
from src.b3_score_ledger import LedgerEvent
from src.b3_state import GameState, ICON_MIN_SAMPLES, IconCorrector, SideState


@pytest.mark.parametrize("enabled,pending,icons,expected", [
    (False, [0, 0], [6, 0], 1), (False, [30, 0], [0, 0], 0),
    (True, [0, 0], [6, 0], 0), (True, [30, 0], [0, 0], 1),
    (True, [np.nan, np.nan], [30, 0], 0), (True, [0, np.nan], [0, 0], 0),
])
def test_mode_uses_selected_source(monkeypatch: pytest.MonkeyPatch, enabled: bool,
                                  pending: list, icons: list, expected: int) -> None:
    monkeypatch.setattr(config, "B3_LEDGER_MODE_DEFAULT", enabled)
    gs = SimpleNamespace(sides=[SideState("1P"), SideState("2P")],
                         sl_snap=dict(zip(("pending_p1", "pending_p2"), pending)))
    row = {}
    mode_group(row, gs, 1.0, icons)
    assert row == {"mode_bits": 8 * expected, "mode_exchange": expected}


def test_mode_keeps_observed_fall_even_with_empty_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "B3_LEDGER_MODE_DEFAULT", True)
    gs = SimpleNamespace(sides=[SideState("1P"), SideState("2P")], sl_snap=None)
    gs.sides[0].state = "OJAMA_FALL"
    row = {}
    mode_group(row, gs, 1.0, [0, 0])
    assert row["mode_bits"] == 2


@pytest.mark.parametrize("enabled,expected", [(False, 0), (True, 6)])
def test_line_alias_is_independent_and_default_distribution_kept(
        monkeypatch: pytest.MonkeyPatch, enabled: bool, expected: int) -> None:
    monkeypatch.setattr(config, "B3_ICON_LINE_ALIAS_DEFAULT", enabled)
    icon = IconCorrector()
    for i in range(ICON_MIN_SAMPLES):
        icon.push(float(i), {"icons": [{"icon_type": "line", "count_value": 6}], "total_count": 6}, True)
    assert icon.raw["large"] == expected
    assert icon.corrected()["large"] == expected
    assert icon.raw_total == 6


@pytest.mark.parametrize("incomplete,waiting,expected", [(False, False, 1), (True, False, 0), (False, True, 0)])
def test_landing_uses_ledger_event_and_keeps_true_board_increase(
        incomplete: bool, waiting: bool, expected: int) -> None:
    gs = GameState(SimpleNamespace(rows_of=lambda side: ()))
    gs.in_match = True
    gs.sl_snap = {"history_incomplete": incomplete, "history_waiting": waiting, "events": (
        LedgerEvent("observed_landing", 0, 1.0, 1.1, (("n", 30), ("board_increase", 4))),)}
    gs._observe_ledger_landings()
    assert gs.sides[0].land_count == expected
    if expected:
        assert gs.sides[0].landing.icon_drop == 30
        assert gs.sides[0].landing.board_net == 4
        assert gs.sides[0].landing.finalize_t == 1.1


def test_zero_consumption_and_timing_prediction_are_not_landings() -> None:
    gs = GameState(SimpleNamespace(rows_of=lambda side: ()))
    gs.in_match = True
    gs.sl_snap = {"events": (
        LedgerEvent("observed_landing", 0, 1.0, 1.0, (("n", 0), ("board_increase", 2))),
        LedgerEvent("landing_predicted", 1, 1.0, 1.0, (("n", 30),)),)}
    gs._observe_ledger_landings()
    assert [s.land_count for s in gs.sides] == [0, 0]


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_landing_is_disabled_only_by_its_own_flag(
        monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(config, "B3_LEDGER_LANDING_DEFAULT", enabled)
    st = SideState("1P")
    st.board = np.zeros((13, 6), dtype=np.int8)
    st.state = "OJAMA_FALL"
    st._update_landing(0.0, True)
    st.board[-1, :] = 9
    st.state = "STABLE"
    st._update_landing(2.0, True)
    assert st.land_count == (0 if enabled else 1)


def test_all_icon_switches_are_default_off() -> None:
    assert not config.B3_LEDGER_MODE_DEFAULT
    assert not config.B3_LEDGER_LANDING_DEFAULT
    assert not config.B3_ICON_LINE_ALIAS_DEFAULT
