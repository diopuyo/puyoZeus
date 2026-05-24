"""B2 ColorOscillationFilter のテスト。"""
from __future__ import annotations

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    Board,
)
from src.color_oscillation_filter import ColorOscillationFilter


def _board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def test_history_below_window_returns_observation() -> None:
    """履歴が window 未満なら観測そのまま。"""
    filt = ColorOscillationFilter(window_size=5)
    obs = _board({(12, 2): COLOR_RED})
    out = filt.update(obs)
    assert out.get(12, 2) == COLOR_RED


def test_oscillation_detected_after_window() -> None:
    """同セルで 2 色観測 → UNKNOWN。"""
    filt = ColorOscillationFilter(window_size=5)
    obs_red = _board({(12, 2): COLOR_RED})
    obs_blue = _board({(12, 2): COLOR_BLUE})
    # 5 フレーム: RED, BLUE, RED, BLUE, RED
    filt.update(obs_red)
    filt.update(obs_blue)
    filt.update(obs_red)
    filt.update(obs_blue)
    out = filt.update(obs_red)
    assert out.get(12, 2) == COLOR_UNKNOWN


def test_no_oscillation_passes_through() -> None:
    """同セル同色のみ → そのまま。"""
    filt = ColorOscillationFilter(window_size=5)
    obs = _board({(12, 2): COLOR_RED})
    for _ in range(5):
        out = filt.update(obs)
    assert out.get(12, 2) == COLOR_RED


def test_empty_does_not_count_as_color() -> None:
    """色 → EMPTY → 色 は振動扱いしない (連鎖消去 + 落下)。"""
    filt = ColorOscillationFilter(window_size=5)
    obs_red = _board({(12, 2): COLOR_RED})
    obs_empty = _board({})
    filt.update(obs_red)
    filt.update(obs_empty)
    filt.update(obs_red)
    filt.update(obs_empty)
    out = filt.update(obs_red)
    # 1 種の色 (RED) のみ観測 → 振動なし
    assert out.get(12, 2) == COLOR_RED


def test_ojama_does_not_trigger() -> None:
    """OJAMA は振動対象外 (NORMAL_COLORS に含まない)。"""
    filt = ColorOscillationFilter(window_size=5)
    obs_red = _board({(12, 2): COLOR_RED})
    obs_ojama = _board({(12, 2): COLOR_OJAMA})
    filt.update(obs_red)
    filt.update(obs_ojama)
    filt.update(obs_red)
    filt.update(obs_ojama)
    out = filt.update(obs_red)
    # OJAMA はカウントされない、RED のみ → 振動なし
    assert out.get(12, 2) == COLOR_RED


def test_three_color_oscillation_detected() -> None:
    """3 色出現も検出 (min_distinct=2 で十分)。"""
    filt = ColorOscillationFilter(window_size=5)
    boards = [
        _board({(12, 2): COLOR_RED}),
        _board({(12, 2): COLOR_BLUE}),
        _board({(12, 2): COLOR_YELLOW}),
        _board({(12, 2): COLOR_RED}),
        _board({(12, 2): COLOR_BLUE}),
    ]
    for b in boards:
        out = filt.update(b)
    assert out.get(12, 2) == COLOR_UNKNOWN


def test_reset_clears_history() -> None:
    filt = ColorOscillationFilter(window_size=3)
    filt.update(_board({(12, 2): COLOR_RED}))
    filt.update(_board({(12, 2): COLOR_BLUE}))
    filt.update(_board({(12, 2): COLOR_RED}))
    # 振動検出された状態
    out = filt.update(_board({(12, 2): COLOR_RED}))
    # reset 後は履歴消える
    filt.reset()
    out2 = filt.update(_board({(12, 2): COLOR_RED}))
    assert out2.get(12, 2) == COLOR_RED


def test_min_distinct_3_strict() -> None:
    """min_distinct=3 で 2 色のみは振動扱いしない。"""
    filt = ColorOscillationFilter(window_size=5, min_distinct_colors=3)
    boards = [
        _board({(12, 2): COLOR_RED}),
        _board({(12, 2): COLOR_BLUE}),
        _board({(12, 2): COLOR_RED}),
        _board({(12, 2): COLOR_BLUE}),
        _board({(12, 2): COLOR_RED}),
    ]
    for b in boards:
        out = filt.update(b)
    # 2 色のみ、3 種未満なので振動扱いしない
    assert out.get(12, 2) == COLOR_RED
