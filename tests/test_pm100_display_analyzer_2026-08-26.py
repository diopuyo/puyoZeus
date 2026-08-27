"""Gate 4の密な実表示集計器の回帰テスト。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "_analyze_pm100_display_pair_2026-08-26.py"
SPEC = importlib.util.spec_from_file_location("pm100_display_analyzer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def _display(adv: list[float], games: list[int] | None = None) -> dict[str, np.ndarray]:
    n = len(adv)
    return {
        "t_sec": np.arange(n, dtype=float) / 30.0,
        "display_adv": np.asarray(adv, dtype=float),
        "adv_raw_last": np.full(n, 10.0),
        "game_idx": np.zeros(n, dtype=int) if games is None else np.asarray(games),
        "state1": np.full(n, "STABLE"), "state2": np.full(n, "STABLE"),
        "score1": np.zeros(n, dtype=int), "score2": np.zeros(n, dtype=int),
    }


def test_metrics_count_dense_stick_reverse_and_flip() -> None:
    result = analyzer._metrics([("seg01", _display(
        [100.0, 100.0, 0.0, -100.0, -100.0]))])
    assert result["gap_bad"] == 0
    assert result["flip_games"] == {("seg01", 0)}
    assert result["flip_events"] == 1
    assert float(result["stick"]) == pytest.approx(3.0 / 30.0)
    assert float(result["wrong"]) == pytest.approx(1.0 / 30.0)


def test_swing_does_not_cross_game_boundary() -> None:
    d = _display([-100.0, 100.0], games=[0, 1])
    assert analyzer._count_swings(
        d["t_sec"], d["display_adv"], d["game_idx"]) == 0


def test_confirmed_truth_is_assigned_to_previous_game() -> None:
    d = {
        "game_idx": np.asarray([2, 3, 3]),
        "is_dead1_confirmed": np.asarray([False, False, False]),
        "is_dead2_confirmed": np.asarray([False, True, True]),
    }
    assert analyzer._truth_from_confirmed("seg01", d) == {("seg01", 2): "1P"}


def test_dead_tail_truth_uses_only_last_two_seconds() -> None:
    d = {
        "t_sec": np.asarray([0.0, 1.0, 4.0]), "game_idx": np.asarray([0, 0, 0]),
        "is_dead1": np.asarray([True, False, False]),
        "is_dead2": np.asarray([False, False, True]),
    }
    assert analyzer._truth_from_dead_tail("seg01", d) == {("seg01", 0): "1P"}


def test_load_truths_rejects_conflicting_sources(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "seg01_timeline.npz",
        t_sec=np.asarray([0.0, 1.0]), game_idx=np.asarray([0, 1]),
        is_dead1=np.asarray([True, False]), is_dead2=np.asarray([False, False]),
        is_dead1_confirmed=np.asarray([False, False]),
        is_dead2_confirmed=np.asarray([False, True]),
    )
    with pytest.raises(ValueError, match="勝者根拠が矛盾"):
        analyzer._load_truths(tmp_path)


def test_load_truths_ignores_object_columns(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "seg01_timeline.npz",
        t_sec=np.asarray([0.0]), game_idx=np.asarray([0]),
        is_dead1=np.asarray([True]), is_dead2=np.asarray([False]),
        object_noise=np.asarray([["name"]], dtype=object),
    )
    assert analyzer._load_truths(tmp_path) == {("seg01", 0): "2P"}


def test_panel_truth_skips_unknown_and_is_authoritative(tmp_path: Path) -> None:
    path = tmp_path / "truth.tsv"
    path.write_text(
        "segment\tgame_idx\twinner\nseg01\t2\t1P\nseg01\t3\tUNKNOWN\n",
        encoding="utf-8")
    assert analyzer._load_panel_truth(path) == {("seg01", 2): "1P"}
