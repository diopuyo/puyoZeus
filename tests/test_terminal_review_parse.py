"""terminal_review の編集テンプレパーサテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# scripts ディレクトリも import 可能にする
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
)

from terminal_review import _parse_template  # noqa: E402


def test_parse_basic_both_sides() -> None:
    text = """
# comment
=== 1P ===
r01: ......
r12: RBGYPO

=== 2P ===
r12: ......
r01: ......
"""
    cells = _parse_template(text)
    # 1P: 12 + 6 = 18, 2P: 12 本の行数ベースではなく、記述行のみ
    # このテストでは 1P に 2 行 (r01, r12) = 12 セル, 2P に 2 行 (r01, r12) = 12 セル
    assert len(cells) == 24
    by_key = {(c.side, c.row, c.col): c for c in cells}
    assert by_key[("1P", 12, 0)].code == COLOR_RED
    assert by_key[("1P", 12, 1)].code == COLOR_BLUE
    assert by_key[("1P", 12, 2)].code == COLOR_GREEN
    assert by_key[("1P", 12, 3)].code == COLOR_YELLOW
    assert by_key[("1P", 12, 4)].code == COLOR_PURPLE
    assert by_key[("1P", 12, 5)].code == COLOR_OJAMA
    assert by_key[("1P", 1, 0)].code == COLOR_EMPTY


def test_parse_spaces_and_underscore() -> None:
    text = """
=== 1P ===
r12:  R . B _ Y O
"""
    cells = _parse_template(text)
    assert len(cells) == 6
    by_col = {c.col: c for c in cells}
    assert by_col[0].code == COLOR_RED
    assert by_col[1].code == COLOR_EMPTY  # .
    assert by_col[2].code == COLOR_BLUE
    assert by_col[3].code == COLOR_EMPTY  # _
    assert by_col[4].code == COLOR_YELLOW
    assert by_col[5].code == COLOR_OJAMA


def test_parse_skip_marker() -> None:
    text = """
=== 1P ===
r12: ?.....
"""
    cells = _parse_template(text)
    assert cells[0].skip is True
    assert cells[0].code is None
    assert cells[1].skip is False
    assert cells[1].code == COLOR_EMPTY


def test_parse_rejects_wrong_length() -> None:
    text = """
=== 1P ===
r12: .....
"""
    with pytest.raises(ValueError, match="6 文字必要"):
        _parse_template(text)


def test_parse_rejects_unknown_char() -> None:
    text = """
=== 1P ===
r12: ....X.
"""
    with pytest.raises(ValueError, match="不明な文字"):
        _parse_template(text)


def test_parse_case_insensitive() -> None:
    text = """
=== 1P ===
r12: rbgypo
"""
    cells = _parse_template(text)
    codes = [c.code for c in cells]
    assert codes == [
        COLOR_RED, COLOR_BLUE, COLOR_GREEN,
        COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
    ]


def test_parse_ignores_lines_before_side_header() -> None:
    text = """
r12: RRRRRR
=== 1P ===
r12: ......
"""
    cells = _parse_template(text)
    # side ヘッダ前の r12 行は無視される
    assert len(cells) == 6
    assert all(c.side == "1P" for c in cells)
