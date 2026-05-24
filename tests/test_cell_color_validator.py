"""CellColorValidator のテスト."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)
from src.board_state_machine import BoardState
from src.self_supervised.cell_color_validator import (
    HISTORY_WINDOW,
    LOOKBACK_FRAMES,
    SETTLE_CONFIDENCE,
    SETTLE_FRAMES_REQUIRED,
    CellColorValidator,
)
from src.self_supervised.pseudo_label import COMPONENT_CELL


# ============================
# モック
# ============================


@dataclass
class _MockSide:
    state: BoardState = BoardState.STABLE
    confirmed_board: Any = None


@dataclass
class _MockResult:
    is_match_active: bool = True
    p1: _MockSide = None
    p2: _MockSide = None


def _empty_board() -> Board:
    return Board()


def _board_with(cells: dict[tuple[int, int], int]) -> Board:
    """指定 (row, col) のセルだけを設定した盤面."""
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def _frame_1080p(value: int = 32) -> np.ndarray:
    """1080p ダミーフレーム (一様灰色)."""
    return np.full((1080, 1920, 3), value, dtype=np.uint8)


def _make_result(
    p1_state: BoardState = BoardState.STABLE,
    p1_board: Board | None = None,
    p2_state: BoardState = BoardState.STABLE,
    p2_board: Board | None = None,
    is_match_active: bool = True,
) -> _MockResult:
    return _MockResult(
        is_match_active=is_match_active,
        p1=_MockSide(
            state=p1_state,
            confirmed_board=p1_board if p1_board is not None else _empty_board(),
        ),
        p2=_MockSide(
            state=p2_state,
            confirmed_board=p2_board if p2_board is not None else _empty_board(),
        ),
    )


# ============================
# 初期化バリデーション
# ============================


def test_validator_init_invalid_settle():
    """settle_frames_required < 2 で ValueError."""
    with pytest.raises(ValueError):
        CellColorValidator(settle_frames_required=1)


def test_validator_init_invalid_lookback():
    """lookback_frames < 1 で ValueError."""
    with pytest.raises(ValueError):
        CellColorValidator(lookback_frames=0)


def test_validator_init_window_too_small():
    """history_window が settle + lookback より小さいと ValueError."""
    with pytest.raises(ValueError):
        CellColorValidator(
            settle_frames_required=5, lookback_frames=10, history_window=10,
        )


# ============================
# 基本動作
# ============================


def test_no_emit_when_inactive():
    """is_match_active=False で何も emit しない."""
    v = CellColorValidator()
    res = _make_result(is_match_active=False)
    for i in range(SETTLE_FRAMES_REQUIRED + 2):
        v.update(i, i * 0.2, res, _frame_1080p())
    assert v.collect() == []


def test_no_emit_when_not_stable():
    """state が STABLE 以外なら履歴を更新せず emit しない."""
    v = CellColorValidator()
    board = _board_with({(11, 0): COLOR_RED})
    res = _make_result(p1_state=BoardState.TSUMO_FALL, p1_board=board)
    for i in range(SETTLE_FRAMES_REQUIRED + LOOKBACK_FRAMES):
        v.update(i, i * 0.2, res, _frame_1080p())
    assert v.collect() == []


def test_no_emit_when_no_misread():
    """settle するが過去 frame で誤認なし → emit ゼロ."""
    v = CellColorValidator()
    # 同じ盤面 (RED at (11,0)) を SETTLE_N+lookback frame 連続で観測
    board = _board_with({(11, 0): COLOR_RED})
    res = _make_result(p1_board=board)
    n_total = SETTLE_FRAMES_REQUIRED + LOOKBACK_FRAMES
    for i in range(n_total):
        v.update(i, i * 0.2, res, _frame_1080p())
    samples = v.collect()
    # 全 frame で同色 → transient 誤認なし → emit ゼロ
    assert len(samples) == 0


# ============================
# settle 検出 + 誤認抽出
# ============================


def test_emit_transient_misread():
    """settle 直前で誤認 frame があれば emit される."""
    v = CellColorValidator()
    # frame 0-2: GREEN (誤認 transient)
    # frame 3-7: RED (= settle 連続 5 frame)
    # → frame 0-2 が誤認 frame として擬似ラベル化されるはず
    misread_board = _board_with({(11, 0): COLOR_GREEN})
    settled_board = _board_with({(11, 0): COLOR_RED})

    res_misread = _make_result(p1_board=misread_board)
    res_settled = _make_result(p1_board=settled_board)

    for i in range(3):
        v.update(i, i * 0.2, res_misread, _frame_1080p())
    for i in range(3, 3 + SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res_settled, _frame_1080p())

    samples = v.collect()
    # cell (11, 0) の擬似ラベルが発生 (3 frame 分)
    cell_samples = [
        s for s in samples
        if s.input_data.get("row") == 11
        and s.input_data.get("col") == 0
        and s.input_data.get("side") == "1P"
    ]
    assert len(cell_samples) == 3
    for s in cell_samples:
        assert s.component == COMPONENT_CELL
        assert s.label == COLOR_RED  # settle 色
        assert s.confidence == SETTLE_CONFIDENCE
        assert s.metadata["predicted_color"] == COLOR_GREEN
        assert s.metadata["settled_color"] == COLOR_RED
        assert s.metadata["source"] == "settle_pattern"


def test_emit_dedup_same_frame():
    """同 cell・同 frame_idx で重複 emit しない."""
    v = CellColorValidator()
    # 1 frame 誤認 + 5 frame 確定 → emit 1 件
    res_mis = _make_result(p1_board=_board_with({(11, 0): COLOR_GREEN}))
    res_set = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    v.update(0, 0.0, res_mis, _frame_1080p())
    for i in range(1, 1 + SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res_set, _frame_1080p())
    # 追加で 1 frame 同じ settle を流す → settle 検出が再発火するが既出なので emit されない
    v.update(1 + SETTLE_FRAMES_REQUIRED, 1.4, res_set, _frame_1080p())
    samples = v.collect()
    cell0 = [
        s for s in samples
        if s.input_data.get("row") == 11
        and s.input_data.get("col") == 0
        and s.metadata.get("frame_idx") == 0
    ]
    # 1 frame 分のみ
    assert len(cell0) == 1


def test_emit_skip_unknown_color():
    """COLOR_UNKNOWN セルは履歴に乗せず emit しない."""
    v = CellColorValidator()
    # row=0 (隠し段) に UNKNOWN を入れた盤面
    unknown_board = _board_with({(0, 0): COLOR_UNKNOWN, (11, 0): COLOR_RED})
    settled_board = _board_with({(0, 0): COLOR_UNKNOWN, (11, 0): COLOR_RED})
    res = _make_result(p1_board=settled_board)
    for i in range(SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res, _frame_1080p())
    samples = v.collect()
    # row=0 (隠し段) からは emit されない
    hidden = [
        s for s in samples
        if s.input_data.get("row") == 0
    ]
    assert len(hidden) == 0


def test_settle_required_consecutive():
    """settle frame 数より少ない連続観測では確定しない."""
    v = CellColorValidator()
    # frame 0-2: GREEN, frame 3-(SETTLE_N-1): RED (連続 RED が SETTLE_N-1 個)
    res_mis = _make_result(p1_board=_board_with({(11, 0): COLOR_GREEN}))
    res_set = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    for i in range(3):
        v.update(i, i * 0.2, res_mis, _frame_1080p())
    n_red = SETTLE_FRAMES_REQUIRED - 1  # 確定足りない
    for i in range(3, 3 + n_red):
        v.update(i, i * 0.2, res_set, _frame_1080p())
    samples = v.collect()
    # まだ settle 未確定 → emit ゼロ
    assert len(samples) == 0


# ============================
# 1P / 2P 独立性
# ============================


def test_1p_2p_independent():
    """1P / 2P は独立に処理される."""
    v = CellColorValidator()
    # 1P: 誤認→確定 / 2P: 全 frame 同色 (誤認なし)
    p1_mis = _board_with({(11, 0): COLOR_GREEN})
    p1_set = _board_with({(11, 0): COLOR_RED})
    p2_const = _board_with({(11, 0): COLOR_BLUE})

    # frame 0-2: 1P=mis, 2P=const
    for i in range(3):
        res = _MockResult(
            is_match_active=True,
            p1=_MockSide(state=BoardState.STABLE, confirmed_board=p1_mis),
            p2=_MockSide(state=BoardState.STABLE, confirmed_board=p2_const),
        )
        v.update(i, i * 0.2, res, _frame_1080p())
    # frame 3-7: 1P=set, 2P=const
    for i in range(3, 3 + SETTLE_FRAMES_REQUIRED):
        res = _MockResult(
            is_match_active=True,
            p1=_MockSide(state=BoardState.STABLE, confirmed_board=p1_set),
            p2=_MockSide(state=BoardState.STABLE, confirmed_board=p2_const),
        )
        v.update(i, i * 0.2, res, _frame_1080p())
    samples = v.collect()
    p1_emit = [s for s in samples if s.input_data["side"] == "1P"]
    p2_emit = [s for s in samples if s.input_data["side"] == "2P"]
    # 1P 側のみ emit (3 frame)
    assert len(p1_emit) == 3
    assert len(p2_emit) == 0


# ============================
# reset
# ============================


def test_reset_clears_state():
    """reset で履歴と emit 履歴が消える."""
    v = CellColorValidator()
    res_mis = _make_result(p1_board=_board_with({(11, 0): COLOR_GREEN}))
    res_set = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    for i in range(3):
        v.update(i, i * 0.2, res_mis, _frame_1080p())
    for i in range(3, 3 + SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res_set, _frame_1080p())
    samples_first = v.collect()
    assert len(samples_first) >= 1
    v.reset()
    # reset 後、同 flow を再実行 → 再度 emit 可能
    for i in range(100, 103):
        v.update(i, i * 0.2, res_mis, _frame_1080p())
    for i in range(103, 103 + SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res_set, _frame_1080p())
    samples_second = v.collect()
    assert len(samples_second) >= 1


# ============================
# 履歴 window
# ============================


def test_history_drops_old_entries():
    """history_window を超えた古い observation は drop される."""
    # window=10 で settle=5 / lookback=4 (= 9 < 10) で初期化可能
    v = CellColorValidator(
        settle_frames_required=5, lookback_frames=4, history_window=10,
    )
    # 大量の frame を流して window より古い entry が消えていることを確認
    res = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    for i in range(50):
        v.update(i, i * 0.2, res, _frame_1080p())
    # 内部 history を直接確認
    key = ("1P", 11, 0)
    assert key in v._history
    assert len(v._history[key]) <= 10


# ============================
# 1080p auto-resize
# ============================


def test_auto_resize_720p_to_1080p():
    """720p frame でも 1080p に resize されて処理される (16:9 のみ)."""
    v = CellColorValidator()
    # 720p ダミーフレーム
    frame_720p = np.full((720, 1280, 3), 32, dtype=np.uint8)
    res = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    for i in range(SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res, frame_720p)
    # crash しなければ OK (履歴更新は走る)
    assert ("1P", 11, 0) in v._history


def test_skip_non_169_frames():
    """16:9 でない frame は skip (履歴更新なし)."""
    v = CellColorValidator()
    # 4:3 frame
    frame_43 = np.full((600, 800, 3), 32, dtype=np.uint8)
    res = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    for i in range(SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res, frame_43)
    # 履歴は空のまま
    assert ("1P", 11, 0) not in v._history


# ============================
# 複数 cell 同時誤認
# ============================


def test_emit_multiple_cells():
    """複数 cell が同時に誤認 → 全て emit される."""
    v = CellColorValidator()
    # cell (11, 0): GREEN→RED
    # cell (11, 1): BLUE→PURPLE (= ユーザー指摘の 13s 例)
    mis_board = _board_with({
        (11, 0): COLOR_GREEN, (11, 1): COLOR_BLUE,
    })
    set_board = _board_with({
        (11, 0): COLOR_RED, (11, 1): COLOR_PURPLE,
    })
    for i in range(3):
        v.update(i, i * 0.2, _make_result(p1_board=mis_board), _frame_1080p())
    for i in range(3, 3 + SETTLE_FRAMES_REQUIRED):
        v.update(
            i, i * 0.2, _make_result(p1_board=set_board), _frame_1080p(),
        )
    samples = v.collect()
    cells_emitted = {
        (s.input_data["row"], s.input_data["col"], s.label)
        for s in samples
        if s.input_data.get("side") == "1P"
    }
    # (11,0)→RED と (11,1)→PURPLE がそれぞれ emit
    assert (11, 0, COLOR_RED) in cells_emitted
    assert (11, 1, COLOR_PURPLE) in cells_emitted


# ============================
# patch サイズ check
# ============================


def test_emitted_patch_is_ndarray():
    """emit された patch が ndarray で空でない."""
    v = CellColorValidator()
    res_mis = _make_result(p1_board=_board_with({(11, 0): COLOR_GREEN}))
    res_set = _make_result(p1_board=_board_with({(11, 0): COLOR_RED}))
    for i in range(3):
        v.update(i, i * 0.2, res_mis, _frame_1080p())
    for i in range(3, 3 + SETTLE_FRAMES_REQUIRED):
        v.update(i, i * 0.2, res_set, _frame_1080p())
    samples = v.collect()
    assert len(samples) >= 1
    patch = samples[0].input_data["patch"]
    assert isinstance(patch, np.ndarray)
    assert patch.ndim == 3
    assert patch.shape[2] == 3  # BGR
    assert patch.size > 0
