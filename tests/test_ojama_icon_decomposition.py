"""src/scoring.py の ojama アイコン分解とターン落下分割のテスト。"""
from __future__ import annotations

import pytest

from src.scoring import (
    OJAMA_ICON_VALUES,
    OJAMA_MAX_DROP_PER_TURN,
    OJAMA_MAX_ICONS_DISPLAY,
    icons_to_ojama_count,
    ojama_count_to_icons,
    split_ojama_drop_per_turn,
)


def test_constants() -> None:
    assert OJAMA_MAX_ICONS_DISPLAY == 6
    assert OJAMA_MAX_DROP_PER_TURN == 30
    assert dict(OJAMA_ICON_VALUES)["crown"] == 720
    assert dict(OJAMA_ICON_VALUES)["small"] == 1


def test_zero_pending_no_icons() -> None:
    assert ojama_count_to_icons(0) == []
    assert ojama_count_to_icons(-5) == []


def test_one_small_only() -> None:
    assert ojama_count_to_icons(1) == [("small", 1)]
    assert ojama_count_to_icons(5) == [("small", 5)]


def test_six_smalls_filling_display() -> None:
    """6 個ちょうど = small 6 個で表示枠 6 を埋める。"""
    assert ojama_count_to_icons(6) == [("large", 1)]
    # large 1 個で表示。large=6 なのでこちらが優先


def test_large_priority_over_small() -> None:
    """7 個 = large 1 個 + small 1 個。"""
    icons = ojama_count_to_icons(7)
    assert ("large", 1) in icons
    assert ("small", 1) in icons


def test_rock_priority() -> None:
    """30 個 = rock 1 個。"""
    assert ojama_count_to_icons(30) == [("rock", 1)]


def test_combined_decomposition() -> None:
    """100 個 = rock 3 + large 1 + small 4 = 8 アイコン → small 切り捨てで 6 アイコン。"""
    icons = ojama_count_to_icons(100)
    used = sum(c for _, c in icons)
    assert used <= OJAMA_MAX_ICONS_DISPLAY
    # rock 3 と large 1 は確実に入る (3+1 = 4, あと 2 枠)
    icon_dict = dict(icons)
    assert icon_dict.get("rock") == 3
    assert icon_dict.get("large") == 1
    # small は max 2 個 (枠残り 2)、表示落ち = 4 - 2 = 2 個


def test_overflow_truncates_smalls() -> None:
    """large×6 は 6 アイコン埋まり small は表示落ち。"""
    icons = ojama_count_to_icons(36 + 5)  # large 6 + small 5、large 6 = 36, +5 = 41
    used = sum(c for _, c in icons)
    assert used <= 6


def test_huge_count_uses_top_icons() -> None:
    """1 万個 = crown を中心に 6 アイコン。"""
    icons = ojama_count_to_icons(10000)
    icon_dict = dict(icons)
    used = sum(c for _, c in icons)
    assert used <= 6
    # crown が含まれる
    assert "crown" in icon_dict


def test_icons_to_count_inverse_for_displayable() -> None:
    """6 アイコン以下に収まる個数なら、分解 → 集約で一致。"""
    for count in [1, 5, 6, 30, 60, 90, 720, 720 + 360 + 180 + 30 + 6 + 1]:
        icons = ojama_count_to_icons(count)
        if sum(c for _, c in icons) <= 6:
            assert icons_to_ojama_count(icons) == count


def test_drop_per_turn_split_basic() -> None:
    """30 個未満は全部今ターン落下。"""
    assert split_ojama_drop_per_turn(20) == (20, 0)
    assert split_ojama_drop_per_turn(30) == (30, 0)


def test_drop_per_turn_split_overflow() -> None:
    """30 個超過は次ターン繰越。"""
    assert split_ojama_drop_per_turn(50) == (30, 20)
    assert split_ojama_drop_per_turn(100) == (30, 70)


def test_drop_per_turn_negative_zero() -> None:
    assert split_ojama_drop_per_turn(0) == (0, 0)
    assert split_ojama_drop_per_turn(-5) == (0, 0)
