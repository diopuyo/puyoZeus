"""ChainFineTuner のテスト."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_RED
from src.chain_detector import (
    DEFAULT_CALIBRATION_PATH,
    ERASURE_MIN_DROP,
    SNAPSHOT_LOOKBACK,
    VideoChainTracker,
)
from src.self_supervised.chain_fine_tuner import (
    DEFAULT_ERASURE_GRID,
    DEFAULT_LOOKBACK_GRID,
    MIN_SAMPLES_FOR_FIT,
    ChainFineTuner,
    _grid_to_board,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_CHAIN,
    COMPONENT_NEXT,
    PseudoLabelSample,
)


# ============================
# helpers
# ============================


def _empty_grid() -> list[list[int]]:
    return [[COLOR_EMPTY for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]


def _chainable_grid() -> list[list[int]]:
    """1 連鎖発火する盤面 (赤 4 連結)."""
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    return grid


def _make_chain_sample(
    *,
    grid: list[list[int]] | None = None,
    duration: int = 100,
    score_jump: int = 1000,
    chain_count: int = 1,
    total_erased: int = 4,
    chain_count_match: bool = True,
    confidence: float = 0.95,
    component: str = COMPONENT_CHAIN,
) -> PseudoLabelSample:
    return PseudoLabelSample(
        component=component,
        timestamp=0.0,
        input_data={
            "before_board_grid": grid if grid is not None else _chainable_grid(),
            "duration_frames": duration,
            "score_jump": score_jump,
            "side": "1P",
        },
        label={
            "chain_count": chain_count,
            "total_erased": total_erased,
            "expected_score_delta": 1000,
        },
        confidence=confidence,
        metadata={
            "side": "1P",
            "chain_event_count": chain_count,
            "simulator_count": chain_count,
            "score_match": True,
            "duration_match": True,
            "chain_count_match": chain_count_match,
            "source": "chain_consistency",
        },
    )


# ============================
# 単体動作
# ============================


def test_fine_tune_with_zero_samples(tmp_path: Path) -> None:
    """サンプル 0 件 → skipped、ファイル未生成."""
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    metrics = ft.fine_tune([])
    assert metrics["n_samples"] == 0
    assert metrics["skipped_reason"] == "not_enough_samples"
    assert not cal_path.exists()


def test_fine_tune_below_min_samples(tmp_path: Path) -> None:
    """MIN_SAMPLES_FOR_FIT 未満は学習せず、ファイル未生成."""
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    samples = [
        _make_chain_sample() for _ in range(MIN_SAMPLES_FOR_FIT - 1)
    ]
    metrics = ft.fine_tune(samples)
    assert metrics["skipped_reason"] == "not_enough_samples"
    assert not cal_path.exists()


def test_fine_tune_filters_other_components(tmp_path: Path) -> None:
    """component != chain は無視される."""
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    chain_samples = [
        _make_chain_sample() for _ in range(MIN_SAMPLES_FOR_FIT)
    ]
    noise = [
        PseudoLabelSample(
            component=COMPONENT_NEXT, timestamp=0.0,
            input_data={"foo": "bar"}, label={"top_color": 1},
            confidence=1.0,
        )
        for _ in range(50)
    ]
    metrics = ft.fine_tune(chain_samples + noise)
    assert metrics["n_samples"] == MIN_SAMPLES_FOR_FIT


def test_fine_tune_writes_calibration(tmp_path: Path) -> None:
    """fine_tune 後 JSON ファイルが書かれ、必要キーを含む."""
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    samples = [
        _make_chain_sample() for _ in range(MIN_SAMPLES_FOR_FIT)
    ]
    ft.fine_tune(samples)
    assert cal_path.exists()
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert "erasure_min_drop" in data
    assert "snapshot_lookback" in data
    assert "n_samples" in data
    assert "accuracy_before" in data
    assert "accuracy_after" in data


def test_fine_tune_finds_lower_threshold_for_small_chains(
    tmp_path: Path,
) -> None:
    """4 連結 (total_erased=4) のサンプルが大半なら、erasure>=4 が選ばれる.

    grid の最小値 (=2) は誤検出を増やすので最適にはならず、
    最終 best_params は ERASURE_MIN_DROP 以下に収束する.
    """
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    samples = [
        _make_chain_sample(total_erased=4, duration=100)
        for _ in range(40)
    ]
    metrics = ft.fine_tune(samples)
    assert metrics["best_params"]["erasure_min_drop"] in DEFAULT_ERASURE_GRID
    # accuracy_after は 1.0 に近いはず (全サンプル chainable)
    assert metrics["accuracy_after"] >= metrics["accuracy_before"]


def test_fine_tune_grid_results_complete(tmp_path: Path) -> None:
    """grid_results は |erasure| × |lookback| 通り."""
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    samples = [_make_chain_sample() for _ in range(MIN_SAMPLES_FOR_FIT)]
    metrics = ft.fine_tune(samples)
    expected = len(DEFAULT_ERASURE_GRID) * len(DEFAULT_LOOKBACK_GRID)
    assert len(metrics["grid_results"]) == expected


def test_rollback_no_backup_deletes(tmp_path: Path) -> None:
    """初回 fine_tune 後 rollback → ファイルは削除される."""
    cal_path = tmp_path / "chain_calib.json"
    ft = ChainFineTuner(calibration_path=cal_path)
    samples = [_make_chain_sample() for _ in range(MIN_SAMPLES_FOR_FIT)]
    ft.fine_tune(samples)
    assert cal_path.exists()
    ft.rollback()
    assert not cal_path.exists()


def test_rollback_restores_backup(tmp_path: Path) -> None:
    """既存 calibration があれば fine_tune→rollback で旧値が戻る."""
    cal_path = tmp_path / "chain_calib.json"
    cal_path.write_text(
        json.dumps({
            "erasure_min_drop": 99,
            "snapshot_lookback": 99,
            "n_samples": 0,
            "accuracy_before": 0.0,
            "accuracy_after": 0.0,
        }),
        encoding="utf-8",
    )
    ft = ChainFineTuner(calibration_path=cal_path)
    samples = [_make_chain_sample() for _ in range(MIN_SAMPLES_FOR_FIT)]
    ft.fine_tune(samples)
    # 上書きされる
    data_after = json.loads(cal_path.read_text(encoding="utf-8"))
    assert data_after["erasure_min_drop"] != 99 or \
        data_after["snapshot_lookback"] != 99 or \
        data_after["n_samples"] >= MIN_SAMPLES_FOR_FIT
    ft.rollback()
    data_restored = json.loads(cal_path.read_text(encoding="utf-8"))
    assert data_restored["erasure_min_drop"] == 99
    assert data_restored["snapshot_lookback"] == 99


def test_grid_to_board_handles_unknown(tmp_path: Path) -> None:
    """UNKNOWN (=10) セルは EMPTY 化されて Board が作られる."""
    grid = _empty_grid()
    grid[0][0] = 10  # UNKNOWN
    grid[12][0] = COLOR_RED
    board = _grid_to_board(grid)
    assert board is not None
    assert board.get(0, 0) == COLOR_EMPTY
    assert board.get(12, 0) == COLOR_RED


def test_grid_to_board_rejects_wrong_size() -> None:
    """サイズ不一致は None を返す."""
    short_grid = [[0] * BOARD_COLS for _ in range(BOARD_ROWS - 1)]
    assert _grid_to_board(short_grid) is None


def test_apply_calibration_loads_threshold(tmp_path: Path) -> None:
    """VideoChainTracker(apply_calibration=True) で JSON 値が読み込まれる."""
    cal_path = tmp_path / "chain_tracker_calibration.json"
    cal_path.write_text(
        json.dumps({
            "erasure_min_drop": 7,
            "snapshot_lookback": 3,
            "n_samples": 100,
            "accuracy_before": 0.8,
            "accuracy_after": 0.9,
        }),
        encoding="utf-8",
    )
    tracker = VideoChainTracker(
        apply_calibration=True, calibration_path=cal_path,
    )
    assert tracker._erasure_min_drop == 7
    assert tracker._lookback == 3


def test_apply_calibration_silent_fallback() -> None:
    """ファイル不在でも crash せず、引数 default が使われる."""
    tracker = VideoChainTracker(
        apply_calibration=True,
        calibration_path=Path("does/not/exist.json"),
    )
    assert tracker._erasure_min_drop == ERASURE_MIN_DROP
    assert tracker._lookback == SNAPSHOT_LOOKBACK


def test_apply_calibration_default_off() -> None:
    """apply_calibration=False (default) では既存挙動 (backwards compat)."""
    tracker = VideoChainTracker()
    assert tracker._erasure_min_drop == ERASURE_MIN_DROP
    assert tracker._lookback == SNAPSHOT_LOOKBACK


def test_apply_calibration_corrupt_file_fallback(tmp_path: Path) -> None:
    """JSON 破損ファイルは silent fallback で default が使われる."""
    cal_path = tmp_path / "broken.json"
    cal_path.write_text("{not valid json", encoding="utf-8")
    tracker = VideoChainTracker(
        apply_calibration=True, calibration_path=cal_path,
    )
    assert tracker._erasure_min_drop == ERASURE_MIN_DROP
    assert tracker._lookback == SNAPSHOT_LOOKBACK
