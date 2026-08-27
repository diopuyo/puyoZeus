"""Gate 4 条件5検証器のfail-silent回帰。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "_verify_gate4_condition5_2026-08-26.py")
_SPEC = importlib.util.spec_from_file_location("gate4_condition5_verifier", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


def _closed_data(generated: float = 10.0) -> dict[str, np.ndarray]:
    return {
        "closed_episode_count": np.asarray([0, 1]),
        "last_closed_status": np.asarray(["", "CLOSED"]),
        "last_close_reason": np.asarray(["", "normal_close"]),
        "last_closed_has_settlement": np.asarray([False, True]),
        "last_closed_generated": np.asarray([0.0, generated]),
        "last_closed_canceled": np.asarray([0.0, 3.0]),
        "last_closed_landed": np.asarray([0.0, 7.0]),
        "last_closed_unreconciled": np.asarray([0.0, 0.0]),
        "last_closed_oversettled": np.asarray([0.0, 0.0]),
    }


def test_normal_closed_conservation_passes() -> None:
    data = _closed_data()
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    stats = verifier._closed_stats(data, indexes)
    assert stats.closed_normal == 1
    assert stats.conservation_bad == 0


def test_wrong_generated_total_is_detected() -> None:
    data = _closed_data(generated=11.0)
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    stats = verifier._closed_stats(data, indexes)
    assert stats.conservation_bad == 1


def test_normal_close_cannot_hide_unreconciled_inside_conservation() -> None:
    data = _closed_data()
    data["last_closed_landed"][1] = 6.0
    data["last_closed_unreconciled"][1] = 1.0
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    stats = verifier._closed_stats(data, indexes)
    assert stats.conservation_bad == 0
    assert stats.normal_unreconciled_bad == 1


def test_skipped_close_count_fails_instead_of_hiding_episode() -> None:
    data = {"closed_episode_count": np.asarray([0, 2])}
    with pytest.raises(ValueError, match="1frameで複数close"):
        verifier._closed_row_indexes(data, Path("synthetic.npz"))


def test_warmup_close_count_is_baseline_not_output_window_event() -> None:
    data = {"closed_episode_count": np.asarray([2, 2, 3])}
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    assert indexes.tolist() == [2]


def test_close_uses_terminal_backfilled_summary_row() -> None:
    data = {
        "closed_episode_count": np.asarray([0, 1, 1]),
        "last_closed_status": np.asarray(["", "CLOSED_FORCED", "CLOSED_FORCED"]),
        "last_close_reason": np.asarray(["", "max_sec", "max_sec"]),
        "last_closed_has_settlement": np.asarray([False, True, True]),
        "last_closed_generated": np.asarray([0.0, 10.0, 10.0]),
        "last_closed_canceled": np.asarray([0.0, 3.0, 10.0]),
        "last_closed_landed": np.asarray([0.0, 0.0, 0.0]),
        "last_closed_unreconciled": np.asarray([0.0, 7.0, 0.0]),
        "last_closed_oversettled": np.asarray([0.0, 0.0, 0.0]),
    }
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    assert indexes.tolist() == [2]
    assert verifier._closed_stats(data, indexes).conservation_bad == 0


def test_forced_close_conservation_is_also_checked() -> None:
    data = _closed_data(generated=11.0)
    data["last_closed_status"][1] = "CLOSED_FORCED"
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    assert verifier._closed_stats(data, indexes).conservation_bad == 1


def test_side_wipe_is_not_counted_as_normal_close() -> None:
    data = _closed_data()
    data["last_close_reason"][1] = "side_wipe"
    indexes = verifier._closed_row_indexes(data, Path("synthetic.npz"))
    stats = verifier._closed_stats(data, indexes)
    assert stats.closed_normal == 0
    assert stats.closed_other == 1


def test_post_close_backfill_global_mismatch_is_detected() -> None:
    data = {
        "post_close_settlement_backfilled_count": np.asarray([0, 0, 1]),
        "post_close_finalize_backfilled_count": np.asarray([0, 1, 1]),
        "unreconciled": np.asarray([20.0, 20.0, 15.0]),
        "ledger_residual_all": np.asarray([20.0, 20.0, 15.0]),
        "post_close_outstanding_delta_total": np.asarray([20.0, 20.0, 15.0]),
        "closed_unreconciled_total": np.asarray([20.0, 30.0, 25.0]),
    }
    assert verifier._post_close_backfill_sync_bad(data) == 1


def test_post_close_backfill_matching_deltas_pass() -> None:
    data = {
        "post_close_settlement_backfilled_count": np.asarray([0, 0, 1]),
        "post_close_finalize_backfilled_count": np.asarray([0, 1, 1]),
        "unreconciled": np.asarray([20.0, 21.0, 20.0]),
        "ledger_residual_all": np.asarray([20.0, 30.0, 25.0]),
        "post_close_outstanding_delta_total": np.asarray([20.0, 30.0, 25.0]),
        "closed_unreconciled_total": np.asarray([20.0, 30.0, 25.0]),
    }
    assert verifier._post_close_backfill_sync_bad(data) == 0


def test_old_summary_backfill_uses_all_closed_total_not_last_only() -> None:
    data = {
        "post_close_settlement_backfilled_count": np.asarray([0, 1]),
        "post_close_finalize_backfilled_count": np.asarray([0, 0]),
        "unreconciled": np.asarray([30.0, 25.0]),
        "ledger_residual_all": np.asarray([30.0, 25.0]),
        "post_close_outstanding_delta_total": np.asarray([30.0, 25.0]),
        "closed_unreconciled_total": np.asarray([30.0, 25.0]),
        "last_closed_unreconciled": np.asarray([20.0, 20.0]),
    }
    assert verifier._post_close_backfill_sync_bad(data) == 0
