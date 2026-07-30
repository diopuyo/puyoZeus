"""UiMaskMatcher の glob_pattern (案A) + UI_MASK_TARGET_CELLS 定数のテスト (2026-07-30)。

背景: cv2.matchTemplate が認識時間の大半を占め、そのうち UI マスク判定が
支配的だった (memory project_recognition_profile_matchtemplate_2026-07-30)。
telop_challenger* / match_end_* の4テンプレートは実測でゼロ発火のため、
load_default の既定 glob を "x_mark*.png" に絞り込んだ (案A)。
"""
from __future__ import annotations

from pathlib import Path

from src.ui_mask import (
    DEFAULT_TEMPLATE_DIR,
    UI_MASK_TARGET_CELLS,
    UiMaskMatcher,
    X_MARK_GLOB_PATTERN,
)


class TestGlobPatternDefault:
    """案A: 既定 glob が x_mark*.png のみを読み込むことを確認する。"""

    def test_default_glob_pattern_constant(self) -> None:
        assert X_MARK_GLOB_PATTERN == "x_mark*.png"

    def test_load_default_reads_only_x_mark_templates(self) -> None:
        """models/ui_templates 直下は10 png (x_mark系6 + telop/match_end系4)。
        既定 glob では x_mark 系 6 枚のみロードされることを確認する。
        """
        matcher = UiMaskMatcher.load_default()
        names = set(matcher._templates.keys())
        assert names, "テンプレートが1枚もロードされていない (ディレクトリ不在?)"
        for name in names:
            assert name.startswith("x_mark"), f"想定外テンプレートが混入: {name}"
        # 実ディレクトリの x_mark*.png 数と一致することを確認 (回帰検知)
        expected = {
            p.stem for p in DEFAULT_TEMPLATE_DIR.glob(X_MARK_GLOB_PATTERN)
        }
        assert names == expected

    def test_load_default_glob_pattern_override_reads_all_pngs(self) -> None:
        """glob_pattern="*.png" を明示指定すれば旧来通り全 png を読み込む
        (backwards compat: 引数を渡せば以前の全読み挙動を再現できる)。
        """
        matcher_all = UiMaskMatcher.load_default(glob_pattern="*.png")
        matcher_x_mark_only = UiMaskMatcher.load_default()
        assert len(matcher_all._templates) >= len(matcher_x_mark_only._templates)
        non_x_mark = {
            name for name in matcher_all._templates
            if not name.startswith("x_mark")
        }
        assert non_x_mark, "全読み込みなのに telop/match_end 系が1枚も無い"

    def test_load_default_missing_dir_still_empty_matcher(self) -> None:
        """存在しないディレクトリは glob_pattern に関わらず空マッチャー。"""
        matcher = UiMaskMatcher.load_default(
            template_dir=Path("models/__no_such_dir__"),
        )
        assert matcher._templates == {}
        assert matcher.is_ui(__import__("numpy").zeros((4, 4, 3), dtype="uint8")) is False


class TestUiMaskTargetCells:
    """UI_MASK_TARGET_CELLS 定数 (案B で使う raw row/col 座標集合) の確認。"""

    def test_target_cells_is_single_cell_raw_row1_col2(self) -> None:
        """実測 (診断スクリプト 112発火全件) と一致する (raw row=1, col=2)。"""
        assert UI_MASK_TARGET_CELLS == frozenset({(1, 2)})

    def test_target_cells_is_frozenset_of_int_tuples(self) -> None:
        assert isinstance(UI_MASK_TARGET_CELLS, frozenset)
        for cell in UI_MASK_TARGET_CELLS:
            assert isinstance(cell, tuple)
            assert len(cell) == 2
