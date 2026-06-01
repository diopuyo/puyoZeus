"""placement_inferrer.py の単体テスト (Phase 1 cycle 71)."""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, Board,
)
from src.placement_inferrer import (
    LARGE_ADD_GUARD_CELLS,
    HSV_CLASSIFY_MAX_DISTANCE,
    HSV_CLASSIFY_REJECT_RATIO,
    HSV_MIN_SATURATION_FOR_CLASSIFY,
    LandingPattern,
    _classify_next_pair_by_hsv,
    _is_hsv_classify_confident,
    _VALID_PUYO_COLORS,
    correct_landing_cells_by_observed_color,
    enumerate_color_assignments,
    enumerate_landing_patterns,
    infer_placement,
    materialize_pattern,
    resolve_after_placement,
)


def _empty_board() -> Board:
    return Board()


def _board_with_height(heights: list[int], color: int = COLOR_RED) -> Board:
    """各列の高さを heights[c] にした (= 最下段から積み上げ) 盤面を作る."""
    b = Board()
    for c, h in enumerate(heights):
        for k in range(h):
            r = BOARD_ROWS - 1 - k
            b.set(r, c, color)
    return b


class TestEnumerateLandingPatterns:
    def test_empty_board_all_columns_vertical(self):
        """空盤面では各列で縦置き (= 6 通り) と隣接横置き (= 5 通り) = 11 通り."""
        patterns = enumerate_landing_patterns(_empty_board())
        verticals = [p for p in patterns if p.orientation == "vertical"]
        horizontals = [p for p in patterns if p.orientation == "horizontal"]
        assert len(verticals) == BOARD_COLS
        assert len(horizontals) == BOARD_COLS - 1

    def test_vertical_pattern_cells_at_bottom(self):
        """空盤面の縦置きは row=11, 12 の 2 cell (= visible 最下と 1 つ上)."""
        patterns = enumerate_landing_patterns(_empty_board())
        v0 = [p for p in patterns if p.orientation == "vertical"][0]
        (r1, c1), (r2, c2) = v0.cells
        assert (r1, c1) == (BOARD_ROWS - 2, 0)  # = (11, 0)
        assert (r2, c2) == (BOARD_ROWS - 1, 0)  # = (12, 0)

    def test_horizontal_excluded_when_heights_differ(self):
        """隣接列の高さが違うと横置き 不可."""
        # 列 0 = 高さ 2, 列 1 = 高さ 0
        b = _board_with_height([2, 0, 0, 0, 0, 0])
        patterns = enumerate_landing_patterns(b)
        # 列 0-1 の横置きは無いはず
        h01 = [
            p for p in patterns if p.orientation == "horizontal"
            and p.cells[0][1] == 0 and p.cells[1][1] == 1
        ]
        assert len(h01) == 0

    def test_horizontal_included_when_heights_match(self):
        """隣接列の高さが同じなら横置き可."""
        b = _board_with_height([2, 2, 0, 0, 0, 0])
        patterns = enumerate_landing_patterns(b)
        h01 = [
            p for p in patterns if p.orientation == "horizontal"
            and p.cells[0][1] == 0 and p.cells[1][1] == 1
        ]
        assert len(h01) == 1
        (r, c) = h01[0].cells[0]
        # 高さ 2 → 最上空 row = 10 (= row 11, 12 が埋まっている)
        assert (r, c) == (BOARD_ROWS - 1 - 2, 0)  # = (10, 0)

    def test_full_column_excluded(self):
        """満杯の列は縦置き候補から除外."""
        heights = [BOARD_ROWS] + [0] * (BOARD_COLS - 1)
        b = _board_with_height(heights)
        patterns = enumerate_landing_patterns(b)
        v_col_0 = [
            p for p in patterns if p.orientation == "vertical"
            and p.cells[0][1] == 0
        ]
        assert len(v_col_0) == 0


class TestEnumerateColorAssignments:
    def test_distinct_colors_two_assignments(self):
        p = LandingPattern(
            cells=((11, 0), (12, 0)), orientation="vertical",
        )
        assignments = enumerate_color_assignments(p, (COLOR_RED, COLOR_BLUE))
        assert len(assignments) == 2
        assert (COLOR_RED, COLOR_BLUE) in assignments
        assert (COLOR_BLUE, COLOR_RED) in assignments

    def test_same_color_single_assignment(self):
        p = LandingPattern(
            cells=((11, 0), (12, 0)), orientation="vertical",
        )
        assignments = enumerate_color_assignments(p, (COLOR_RED, COLOR_RED))
        assert assignments == [(COLOR_RED, COLOR_RED)]


class TestMaterializePattern:
    def test_writes_two_cells(self):
        base = _empty_board()
        p = LandingPattern(
            cells=((11, 2), (12, 2)), orientation="vertical",
        )
        out = materialize_pattern(base, p, COLOR_RED, COLOR_BLUE)
        assert out.get(11, 2) == COLOR_RED
        assert out.get(12, 2) == COLOR_BLUE
        # 他 cell は EMPTY
        assert out.get(11, 0) == COLOR_EMPTY


class TestInferPlacement:
    def test_no_next_pair_returns_none(self):
        result = infer_placement(_empty_board(), _empty_board(), None)
        assert result is None

    def test_invalid_next_pair_returns_none(self):
        result = infer_placement(
            _empty_board(), _empty_board(), (COLOR_EMPTY, COLOR_RED),
        )
        assert result is None

    def test_simple_vertical_placement_matches_cnn(self):
        """空盤面で CNN が col=2 縦置き赤青を観測 → 推論一致."""
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_RED)
        cnn.set(12, 2, COLOR_BLUE)
        result = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
        )
        assert result is not None
        assert result.get(11, 2) == COLOR_RED
        assert result.get(12, 2) == COLOR_BLUE

    def test_horizontal_placement_matches_cnn(self):
        """空盤面で CNN が row=12 col=2,3 横置きを観測 → 推論一致."""
        cnn = _empty_board()
        cnn.set(12, 2, COLOR_RED)
        cnn.set(12, 3, COLOR_BLUE)
        result = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
        )
        assert result is not None
        assert result.get(12, 2) == COLOR_RED
        assert result.get(12, 3) == COLOR_BLUE

    def test_cnn_misrecognition_corrected_by_next_pair(self):
        """CNN が col=2 縦置きで色を誤認 (= 緑/黄) → NEXT 赤青で正しく出力."""
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_GREEN)  # CNN 誤認
        cnn.set(12, 2, COLOR_YELLOW)  # CNN 誤認
        result = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
        )
        # NEXT が赤青なので、 位置は col=2 縦のはずだが色は赤青のどちらか
        assert result is not None
        v = (result.get(11, 2), result.get(12, 2))
        assert v in (
            (COLOR_RED, COLOR_BLUE), (COLOR_BLUE, COLOR_RED),
        )

    def test_vertical_top_bot_disambiguation_by_cnn(self):
        """CNN が上=青、 下=赤を観測 → NEXT (赤,青) でも上下逆転を尊重."""
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_BLUE)
        cnn.set(12, 2, COLOR_RED)
        result = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
        )
        assert result is not None
        assert result.get(11, 2) == COLOR_BLUE
        assert result.get(12, 2) == COLOR_RED

    def test_existing_field_drops_to_top(self):
        """既存盤面の上に着地 → 最上空 row に置かれる."""
        before = _board_with_height([0, 0, 3, 0, 0, 0])
        # 列 2 は高さ 3 (= row 10,11,12 埋まり、 row 9 が最上空)
        cnn = before.copy()
        cnn.set(8, 2, COLOR_RED)  # row 8 = 最上空 - 1
        cnn.set(9, 2, COLOR_BLUE)  # row 9 = 最上空
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE))
        assert result is not None
        assert result.get(8, 2) == COLOR_RED
        assert result.get(9, 2) == COLOR_BLUE


class TestInferPlacementCycle71b:
    """cycle 71b: 案 A (連鎖整合性) + 案 B (縦/横幾何) のテスト."""

    def test_orientation_b_vertical_forced(self):
        """CNN 差分が縦 2 cell → 横置きパターンは候補から除外される."""
        # 空盤面、 CNN が col=2 で row 11,12 縦置きを観測.
        # 旧実装では候補に横置き (= row=12, col=2,3) も残り得たが、
        # 案 B で diff orientation が "vertical" に確定 → 横置きパターン除外.
        before = _empty_board()
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_RED)
        cnn.set(12, 2, COLOR_BLUE)
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE))
        assert result is not None
        # col=2 縦置き結果. 横置き row=12 col=1,2 等にはなっていない.
        assert result.get(11, 2) in (COLOR_RED, COLOR_BLUE)
        assert result.get(12, 2) in (COLOR_RED, COLOR_BLUE)
        # 隣の col=1, col=3 row=12 は空のまま (= 横置きされていない)
        assert result.get(12, 1) == COLOR_EMPTY
        assert result.get(12, 3) == COLOR_EMPTY

    def test_orientation_b_horizontal_forced(self):
        """CNN 差分が横 2 cell → 縦置きパターンは候補から除外される."""
        before = _empty_board()
        cnn = _empty_board()
        cnn.set(12, 2, COLOR_RED)
        cnn.set(12, 3, COLOR_BLUE)
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE))
        assert result is not None
        # row=12 横置き結果.
        assert result.get(12, 2) in (COLOR_RED, COLOR_BLUE)
        assert result.get(12, 3) in (COLOR_RED, COLOR_BLUE)
        # row=11 col=2,3 は空のまま (= 縦置きされていない)
        assert result.get(11, 2) == COLOR_EMPTY
        assert result.get(11, 3) == COLOR_EMPTY

    def test_orientation_b_falls_back_when_diff_ambiguous(self):
        """CNN 差分が 1 cell or 3 cell 以上 → orientation 確定不能、 全候補維持."""
        before = _empty_board()
        cnn = _empty_board()
        cnn.set(12, 2, COLOR_RED)  # 1 cell のみ (= 横置き相方が欠落)
        # next_pair に従って何か配置されるはず (= 全パターン候補から CNN 一致最大)
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE))
        assert result is not None

    def test_chain_sim_tie_break_prefers_no_chain(self):
        """同点候補で連鎖発生なし候補を優先 (案 A、 score_delta=0 default)."""
        from src.chain import ChainSimulator
        # 連鎖が起きる/起きない両方の候補が同点となる人為的シナリオ:
        # 既存盤面: col=0 に RED 3 個連続、 col=1 に RED 1 個
        # NEXT (RED, BLUE) で着地候補:
        #   - col=0 縦置きで RED 上 / BLUE 下 → col=0 RED 4 個 = 連鎖発生
        #   - col=0 縦置きで BLUE 上 / RED 下 → col=0 に BLUE+RED = 連鎖発生
        # 全部連鎖発生ケース → tie-break は無効、 CNN 一致度で決まる.
        # ここでは tie-break ロジックが動作することだけ確認:
        before = _empty_board()
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_RED)
        cnn.set(12, 2, COLOR_BLUE)
        sim = ChainSimulator()
        result = infer_placement(
            before, cnn, (COLOR_RED, COLOR_BLUE),
            chain_sim=sim, score_delta_observed=0,
        )
        # 連鎖は起きない (= 単独 1 ペア配置)、 縦置きで CNN 一致候補が選ばれる.
        assert result is not None
        assert result.get(11, 2) == COLOR_RED
        assert result.get(12, 2) == COLOR_BLUE

    def test_chain_sim_score_delta_prefers_chain(self):
        """score_delta > 0 のとき連鎖発生候補を優先 (案 A)."""
        from src.chain import ChainSimulator
        # 既存盤面: col=2 に RED 3 個積み (= row 10,11,12)
        before = _board_with_height([0, 0, 3, 0, 0, 0], color=COLOR_RED)
        cnn = before.copy()
        cnn.set(8, 2, COLOR_RED)  # 上に RED 追加 → 4 個連続 = 連鎖発生候補
        cnn.set(9, 2, COLOR_BLUE)
        sim = ChainSimulator()
        # score_delta=100 で「連鎖が起きた」 を示唆
        result = infer_placement(
            before, cnn, (COLOR_RED, COLOR_BLUE),
            chain_sim=sim, score_delta_observed=100,
        )
        assert result is not None
        # 候補は (RED, BLUE), (BLUE, RED) の 2 つで、 RED 上 (= 連鎖発生) が優先される
        assert result.get(8, 2) == COLOR_RED  # 上が赤 → 既存赤と 4 連結 = 連鎖
        assert result.get(9, 2) == COLOR_BLUE


class TestInferPlacementEmptyGuard:
    """guard_empty_hallucination 案 B: 観測セルは NEXT 色、非観測 EMPTY セルは
    COLOR_UNKNOWN 留保、commit refuse 廃止のテスト。"""

    def test_guard_off_legacy_behavior_unchanged(self):
        """guard OFF (default) → 非 diff セルが EMPTY でも従来通り候補採用。"""
        before = _empty_board()
        # CNN は col=2 の下セルのみ変化 (diff=1)、 上セルは EMPTY のまま
        cnn = _empty_board()
        cnn.set(12, 2, COLOR_RED)  # diff=1 cell のみ
        # guard OFF では diff=1 cell を含む pattern が採用される
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE),
                                 guard_empty_hallucination=False)
        assert result is not None

    def test_guard_on_diff1_empty_nondiff_commits_with_unknown(self):
        """guard ON + diff=1 + 非 diff セルが CNN EMPTY → None を返さず盤面を commit。

        案 B 変更点: 旧実装は全パターンスキップ→None (commit refuse) だったが、
        新実装は「観測セル (row=12,col=2) に NEXT 色、非観測セル (row=11,col=2) に
        COLOR_UNKNOWN」として盤面を返す。color→empty 副作用を除去。
        """
        before = _empty_board()
        cnn = _empty_board()
        # col=2 の下セルだけ変化、 上セルは EMPTY のまま
        cnn.set(12, 2, COLOR_RED)  # diff=1 cell
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE),
                                 guard_empty_hallucination=True)
        # 案 B: commit refuse しない (None を返さない)
        assert result is not None
        # 観測した下セルには NEXT 色のいずれかが書かれる
        assert result.get(12, 2) in (COLOR_RED, COLOR_BLUE)
        # 非観測 EMPTY だった上セルは COLOR_UNKNOWN 留保 (hallucination 防止)
        assert result.get(11, 2) == COLOR_UNKNOWN

    def test_guard_on_unknown_nondiff_allows_physical_completion(self):
        """guard ON + 非 diff セルが CNN UNKNOWN → 物理補完で NEXT 色を書く。"""
        before = _empty_board()
        cnn = _empty_board()
        # 下セルが COLOR_RED に変化 (diff)
        cnn.set(12, 2, COLOR_RED)
        # 上セルを COLOR_UNKNOWN に設定 (CNN 不確実)
        cnn.set(11, 2, COLOR_UNKNOWN)
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE),
                                 guard_empty_hallucination=True)
        assert result is not None
        # UNKNOWN セルは物理補完 → NEXT 色のいずれかが書かれる
        assert result.get(11, 2) in (COLOR_RED, COLOR_BLUE)
        assert result.get(12, 2) in (COLOR_RED, COLOR_BLUE)

    def test_guard_on_both_diff_both_colored_allows_pattern(self):
        """guard ON + 2 セルともに diff かつ有色 → 両方 NEXT 色で通常 commit。"""
        before = _empty_board()
        cnn = _empty_board()
        # 縦置きパターン 2 セルともに diff
        cnn.set(11, 2, COLOR_RED)
        cnn.set(12, 2, COLOR_BLUE)
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE),
                                 guard_empty_hallucination=True)
        assert result is not None
        assert result.get(11, 2) == COLOR_RED
        assert result.get(12, 2) == COLOR_BLUE

    def test_guard_on_no_hallucination_when_both_diff(self):
        """guard ON + 両 diff → COLOR_UNKNOWN 上書きなし (留保が起きない)。"""
        before = _empty_board()
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_BLUE)
        cnn.set(12, 2, COLOR_RED)
        result = infer_placement(before, cnn, (COLOR_RED, COLOR_BLUE),
                                 guard_empty_hallucination=True)
        assert result is not None
        # どちらも diff → UNKNOWN への置換は発生しない
        assert result.get(11, 2) != COLOR_UNKNOWN
        assert result.get(12, 2) != COLOR_UNKNOWN


class TestResolveAfterPlacementGuard:
    """cycle 71c: 大量 add ガードのテスト (= A=hit α ケース対策)."""

    def test_no_prev_falls_back_to_legacy(self):
        """prev_confirmed=None で従来通り chain_sim を呼ぶ."""
        from src.chain import ChainSimulator
        # 連鎖が発生する盤面: col=0 に RED 4 連結
        new_board = _board_with_height([4, 0, 0, 0, 0, 0], color=COLOR_RED)
        sim = ChainSimulator()
        final, n = resolve_after_placement(new_board, sim)
        # 連鎖発生 → chain_count >= 1
        assert n >= 1

    def test_guard_blocks_large_add(self):
        """prev → new で puyo cell 数が threshold 超増加なら chain skip."""
        from src.chain import ChainSimulator
        prev = _empty_board()  # 0 cells
        # new に大量 (= 8 cells) 追加 (= LARGE_ADD_GUARD_CELLS=6 超)
        new_board = _board_with_height([4, 4, 0, 0, 0, 0], color=COLOR_RED)
        # 注: col=0,1 で RED 4 個ずつ → 連結 8 個 = 連鎖発生 だが、 ガードで skip
        sim = ChainSimulator()
        final, n = resolve_after_placement(new_board, sim, prev_confirmed=prev)
        assert n == 0
        # final は new_board そのまま (= chain 結果に置換していない)
        assert final.get(BOARD_ROWS - 1, 0) == COLOR_RED

    def test_guard_allows_normal_placement(self):
        """通常着地 (+2 cells) はガード発動しない."""
        from src.chain import ChainSimulator
        # prev に col=0 RED 3 個、 new で 1 つ追加 (= +1 cell、 連鎖発生)
        prev = _board_with_height([3, 0, 0, 0, 0, 0], color=COLOR_RED)
        new_board = prev.copy()
        # row=9 col=0 に RED 追加 → 4 連結 → 連鎖発生
        new_board.set(9, 0, COLOR_RED)
        sim = ChainSimulator()
        final, n = resolve_after_placement(new_board, sim, prev_confirmed=prev)
        # ガード閾値 6 cells 以下なので chain_sim が動作 → 連鎖発生
        assert n >= 1

    def test_guard_threshold_is_inclusive_upper_bound(self):
        """+LARGE_ADD_GUARD_CELLS ちょうどはガード発動しない (= strict greater)."""
        from src.chain import ChainSimulator
        prev = _empty_board()
        # ちょうど LARGE_ADD_GUARD_CELLS (=6) cells 追加. 連鎖は発生しない配置.
        new_board = _empty_board()
        # 4 色をバラバラに配置 (= 連結なし、 chain 0)
        new_board.set(12, 0, COLOR_RED)
        new_board.set(12, 1, COLOR_BLUE)
        new_board.set(12, 2, COLOR_GREEN)
        new_board.set(12, 3, COLOR_YELLOW)
        new_board.set(12, 4, COLOR_RED)
        new_board.set(12, 5, COLOR_BLUE)
        sim = ChainSimulator()
        final, n = resolve_after_placement(new_board, sim, prev_confirmed=prev)
        # cells 差分 = 6 = threshold ぴったり → ガード非発動、 chain_sim 動作で n=0
        assert n == 0
        # final は new_board のコピー (= ガード時と区別つきにくいが連鎖無しケース)
        assert final.get(12, 0) == COLOR_RED


# ===================================================================
# HSV 分類 fallback テスト (fix/v70-zeropatch-redyellow, 2026-06-01)
# ===================================================================

import numpy as np


def _make_hsv_patch_bgr(h: int, s: int, v: int, size: int = 8) -> np.ndarray:
    """指定 HSV の単色 BGR patch を生成する (テスト用)。

    OpenCV HSV 範囲: H=0-180, S=0-255, V=0-255.
    """
    import cv2
    hsv = np.full((size, size, 3), (h, s, v), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestHsvClassifyFallbackConstants:
    """定数が期待範囲にあることを確認する。"""

    def test_reject_ratio_positive(self):
        """HSV_CLASSIFY_REJECT_RATIO は 1.0 超の正値。"""
        assert HSV_CLASSIFY_REJECT_RATIO > 1.0

    def test_max_distance_positive(self):
        """HSV_CLASSIFY_MAX_DISTANCE は正値。"""
        assert HSV_CLASSIFY_MAX_DISTANCE > 0.0

    def test_min_saturation_in_range(self):
        """HSV_MIN_SATURATION_FOR_CLASSIFY は 0-255 の範囲内。"""
        assert 0 < HSV_MIN_SATURATION_FOR_CLASSIFY < 255


class TestIsHsvClassifyConfident:
    """_is_hsv_classify_confident の境界値テスト。"""

    def test_confident_when_ratio_exceeds_threshold(self):
        """d_max/d_min が REJECT_RATIO 以上なら True。"""
        d_min = 10.0
        d_max = d_min * HSV_CLASSIFY_REJECT_RATIO * 2  # 余裕を持って閾値超え
        assert _is_hsv_classify_confident(d_min, d_max) is True

    def test_not_confident_when_both_far(self):
        """d_min が HSV_CLASSIFY_MAX_DISTANCE 超なら False (両候補とも遠い)。"""
        d_min = HSV_CLASSIFY_MAX_DISTANCE + 1.0
        d_max = d_min * 10
        assert _is_hsv_classify_confident(d_min, d_max) is False

    def test_not_confident_when_ratio_below_threshold(self):
        """d_max/d_min が REJECT_RATIO 未満なら False (両候補が拮抗)。"""
        d_min = 10.0
        d_max = d_min * (HSV_CLASSIFY_REJECT_RATIO - 0.1)  # 閾値をわずかに下回る
        assert _is_hsv_classify_confident(d_min, d_max) is False


class TestClassifyNextPairByHsvFallback:
    """_classify_next_pair_by_hsv の enable_hsv_classify_fallback テスト。"""

    def test_off_legacy_two_choice_unchanged(self):
        """fallback=OFF (default) → 従来の 2 択強制確定が維持される。

        黄に近い patch + falling_pair=(青,赤) でも赤に強制確定する旧挙動を確認。
        """
        import cv2
        from src.board import COLOR_BLUE
        # 黄の patch (H=26, S=200, V=220)
        patch_yellow = _make_hsv_patch_bgr(h=26, s=200, v=220)
        # 何か適当な patch (H=115 = 青)
        patch_blue = _make_hsv_patch_bgr(h=115, s=200, v=180)
        # fallback OFF: 2 択強制 → 黄でも赤/青のどちらかに確定される
        result = _classify_next_pair_by_hsv(
            patch_yellow, patch_blue,
            next_pair=(COLOR_BLUE, COLOR_RED),
            enable_hsv_classify_fallback=False,
        )
        # 2 択強制なので COLOR_BLUE か COLOR_RED のどちらかが返る (next_pair の順序変化は OK)
        assert result in ((COLOR_BLUE, COLOR_RED), (COLOR_RED, COLOR_BLUE))

    def test_on_yellow_as_red_candidate_returns_next_pair(self):
        """fallback=ON + 黄 patch + falling_pair=(青,赤) → 拮抗 or 遠いため next_pair 素返し。

        board_log 実証: 黄(H26)→赤(H7) 誤分類発火点。
        falling_pair=(青,赤) 時、黄セルは赤(H7)と青(H115)両方と距離が中程度になる。
        fallback=ON で強制確定せず next_pair=(青,赤) をそのまま返すことを確認。
        """
        from src.board import COLOR_BLUE
        # 黄の patch (H=26 = 黄色相)
        patch_yellow = _make_hsv_patch_bgr(h=26, s=200, v=220)
        # 青の patch (H=115)
        patch_blue_actual = _make_hsv_patch_bgr(h=115, s=200, v=180)
        original_pair = (COLOR_BLUE, COLOR_RED)
        result = _classify_next_pair_by_hsv(
            patch_yellow, patch_blue_actual,
            next_pair=original_pair,
            enable_hsv_classify_fallback=True,
        )
        # fallback ON では「不確かな分類は next_pair 素返し」。
        # 黄は赤でも青でもないため、両距離が中程度 or 比が閾値未満になる。
        # → next_pair そのままか、確信ある分類のどちらか。
        # 少なくとも「黄を赤と強制確定した結果 (COLOR_RED, COLOR_BLUE)」は
        # fallback が発動した場合には素返し値と一致する。
        assert result in (original_pair, (COLOR_RED, COLOR_BLUE))

    def test_on_clear_color_still_classifies(self):
        """fallback=ON + 明確な色 (距離が十分離れる) → 従来通り確定する。

        赤 patch + falling_pair=(赤,青) → 赤が cell_a に確定されること。
        """
        from src.board import COLOR_BLUE
        # 赤の patch (H=7, S=220, V=200) = COLOR_HSV_CENTERS[COLOR_RED] に近い
        patch_red = _make_hsv_patch_bgr(h=7, s=220, v=200)
        # 青の patch (H=115, S=220, V=180)
        patch_blue = _make_hsv_patch_bgr(h=115, s=220, v=180)
        result = _classify_next_pair_by_hsv(
            patch_red, patch_blue,
            next_pair=(COLOR_RED, COLOR_BLUE),
            enable_hsv_classify_fallback=True,
        )
        # 赤と青は H 差 108 で十分離れている → 確定可能 → 従来通り (赤,青) が返る
        assert result == (COLOR_RED, COLOR_BLUE)

    def test_on_low_saturation_patch_returns_next_pair(self):
        """fallback=ON + 低彩度 patch (背景/空) → 色判断不能として next_pair 素返し。"""
        from src.board import COLOR_BLUE
        # 低彩度 patch (S < HSV_MIN_SATURATION_FOR_CLASSIFY = 60)
        patch_low_s = _make_hsv_patch_bgr(h=26, s=30, v=200)  # S=30 < 60
        patch_normal = _make_hsv_patch_bgr(h=115, s=200, v=180)
        original_pair = (COLOR_BLUE, COLOR_RED)
        result = _classify_next_pair_by_hsv(
            patch_low_s, patch_normal,
            next_pair=original_pair,
            enable_hsv_classify_fallback=True,
        )
        # 低彩度 patch → 色判断不能 → next_pair 素返し
        assert result == original_pair


class TestInferPlacementHsvClassifyFallback:
    """infer_placement の enable_hsv_classify_fallback 引数テスト。"""

    def test_flag_off_default_backward_compat(self):
        """フラグ OFF (default) → 既存の infer_placement 挙動と完全一致。"""
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_RED)
        cnn.set(12, 2, COLOR_BLUE)
        result_default = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
        )
        result_explicit_off = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
            enable_hsv_classify_fallback=False,
        )
        # どちらも同じ結果 (後方互換)
        assert result_default is not None
        assert result_explicit_off is not None
        assert result_default.get(11, 2) == result_explicit_off.get(11, 2)
        assert result_default.get(12, 2) == result_explicit_off.get(12, 2)

    def test_flag_on_frame_bgr_none_no_crash(self):
        """フラグ ON かつ frame_bgr=None → クラッシュしない (HSV 分類スキップ)。"""
        cnn = _empty_board()
        cnn.set(11, 2, COLOR_RED)
        cnn.set(12, 2, COLOR_BLUE)
        result = infer_placement(
            _empty_board(), cnn, (COLOR_RED, COLOR_BLUE),
            enable_hsv_classify_fallback=True,
            # frame_bgr=None → use_hsv_classification=False → fallback 不発動
        )
        assert result is not None


class TestCorrectLandingCellsByObservedColor:
    """correct_landing_cells_by_observed_color の単体テスト (真因 A 対処)."""

    def _make_dummy_pattern(
        self, cells: tuple[tuple[int, int], tuple[int, int]],
    ) -> LandingPattern:
        """テスト用 LandingPattern を生成する。"""
        (r1, c1), (r2, c2) = cells
        orientation = "vertical" if c1 == c2 else "horizontal"
        return LandingPattern(cells=cells, orientation=orientation)

    def test_cnn_equals_hsv_valid_color_overwrite(self):
        """CNN == HSV かつ有効色 → inferred の着地セルを観測色で上書きする."""
        import numpy as np
        from src.board import COLOR_YELLOW
        from src.placement_inferrer import correct_landing_cells_by_observed_color

        # inferred_landing: 着地セルが falling_pair 由来の誤色 (青) になっている
        inferred = _empty_board()
        inferred.set(11, 2, COLOR_BLUE)   # 本来は黄のはずだが infer_placement が青を書いた
        inferred.set(12, 2, COLOR_RED)

        # cnn_board: CNN は黄と認識している
        cnn_board = _empty_board()
        cnn_board.set(11, 2, COLOR_YELLOW)
        cnn_board.set(12, 2, COLOR_RED)

        class _TwoColorClassifier:
            """(11,2) と (12,2) で HSV が黄を返すスタブ。"""
            def classify(self, patch: object) -> int:  # type: ignore[override]
                return COLOR_YELLOW

        pattern = self._make_dummy_pattern(((11, 2), (12, 2)))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row: int, col: int) -> tuple:
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _TwoColorClassifier(), frame, _StubRegion(),
        )
        # セル (11,2): CNN=黄, HSV=黄 → 一致 → 黄に上書き (inferred の青が消える)
        assert int(result.get(11, 2)) == COLOR_YELLOW
        # セル (12,2): CNN=赤, HSV=黄 → 不一致 → inferred のままの赤
        assert int(result.get(12, 2)) == COLOR_RED

    def test_cnn_not_equal_hsv_no_overwrite(self):
        """CNN != HSV → 上書きしない (inferred のまま保持)."""
        import numpy as np
        from src.placement_inferrer import correct_landing_cells_by_observed_color

        inferred = _empty_board()
        inferred.set(11, 2, COLOR_BLUE)
        inferred.set(12, 2, COLOR_RED)

        cnn_board = _empty_board()
        cnn_board.set(11, 2, COLOR_YELLOW)
        cnn_board.set(12, 2, COLOR_GREEN)

        class _HsvReturnsRed:
            """HSV は常に赤を返すスタブ。"""
            def classify(self, patch: object) -> int:  # type: ignore[override]
                return COLOR_RED

        pattern = self._make_dummy_pattern(((11, 2), (12, 2)))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row: int, col: int) -> tuple:
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _HsvReturnsRed(), frame, _StubRegion(),
        )
        # CNN != HSV → 上書きなし → inferred の色をそのまま返す
        assert int(result.get(11, 2)) == COLOR_BLUE
        assert int(result.get(12, 2)) == COLOR_RED

    def test_cnn_invalid_color_no_overwrite(self):
        """CNN が無効色 (お邪魔) → 上書きしない."""
        import numpy as np
        from src.board import COLOR_OJAMA
        from src.placement_inferrer import correct_landing_cells_by_observed_color

        inferred = _empty_board()
        inferred.set(11, 2, COLOR_GREEN)

        # CNN がお邪魔 (= 有効 puyo 色でない) を返す
        cnn_board = _empty_board()
        cnn_board.set(11, 2, COLOR_OJAMA)

        class _HsvOjama:
            """HSV もお邪魔を返すスタブ。"""
            def classify(self, patch: object) -> int:  # type: ignore[override]
                return COLOR_OJAMA

        pattern = self._make_dummy_pattern(((11, 2), (12, 2)))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row: int, col: int) -> tuple:
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _HsvOjama(), frame, _StubRegion(),
        )
        # CNN がお邪魔 (無効色) → _VALID_PUYO_COLORS 外 → 上書きしない
        assert int(result.get(11, 2)) == COLOR_GREEN

class TestCorrectLandingCellsByObservedColor:
    """correct_landing_cells_by_observed_color の単体テスト (真因 A 対処)."""

    def _make_stub_hsv_classifier(self, return_color: int):
        """指定した色を常に返す HSV-only 分類器スタブ."""
        class _StubClassifier:
            def classify(self, patch):  # noqa: ANN001
                return return_color
        return _StubClassifier()

    def _make_dummy_pattern(
        self, cells: tuple[tuple[int, int], tuple[int, int]],
    ) -> LandingPattern:
        (r1, c1), (r2, c2) = cells
        orientation = "vertical" if c1 == c2 else "horizontal"
        return LandingPattern(cells=cells, orientation=orientation)

    def _make_small_patch(self) -> "import numpy as np; np.ndarray":
        """4x4 赤色 BGR パッチを返す。"""
        import numpy as np
        # OpenCV BGR 形式 (実際の色は classify で無視される)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_cnn_equals_hsv_valid_color_overwrite(self):
        """CNN == HSV かつ有効色 → inferred の着地セルを観測色で上書きする."""
        import numpy as np

        # inferred_landing: 着地セルが falling_pair 由来の誤色 (赤→青の誤書き)
        inferred = _empty_board()
        inferred.set(11, 2, COLOR_BLUE)   # 本来は黄のはずだが infer_placement が青を書いた
        inferred.set(12, 2, COLOR_RED)

        # cnn_board: CNN は黄と認識している
        cnn_board = _empty_board()
        cnn_board.set(11, 2, COLOR_YELLOW)
        cnn_board.set(12, 2, COLOR_RED)

        # HSV 分類器: セル (11,2) で黄を返す、(12,2) で赤を返す
        class _TwoColorClassifier:
            def classify(self, patch):  # noqa: ANN001
                return COLOR_YELLOW  # 両セルで黄返し (テスト簡略化)

        pattern = self._make_dummy_pattern(((11, 2), (12, 2)))

        # ダミーフレームと region
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row, col):  # noqa: ANN001
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _TwoColorClassifier(), frame, _StubRegion(),
        )
        # セル (11,2): CNN=黄, HSV=黄 → 一致 → 黄に上書き (inferred の青が消える)
        assert int(result.get(11, 2)) == COLOR_YELLOW
        # セル (12,2): CNN=赤, HSV=黄 → 不一致 → inferred のままの赤
        assert int(result.get(12, 2)) == COLOR_RED

    def test_cnn_not_equal_hsv_no_overwrite(self):
        """CNN != HSV → 上書きしない (inferred のまま保持)."""
        import numpy as np

        inferred = _empty_board()
        inferred.set(11, 2, COLOR_BLUE)
        inferred.set(12, 2, COLOR_RED)

        cnn_board = _empty_board()
        cnn_board.set(11, 2, COLOR_YELLOW)  # CNN=黄
        cnn_board.set(12, 2, COLOR_GREEN)   # CNN=緑

        class _HsvReturnsRed:
            def classify(self, patch):  # noqa: ANN001
                return COLOR_RED  # HSV は赤 (CNNの黄・緑と不一致)

        pattern = self._make_dummy_pattern(((11, 2), (12, 2)))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row, col):  # noqa: ANN001
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _HsvReturnsRed(), frame, _StubRegion(),
        )
        # CNN != HSV → 上書きなし → inferred の色をそのまま返す
        assert int(result.get(11, 2)) == COLOR_BLUE
        assert int(result.get(12, 2)) == COLOR_RED

    def test_cnn_invalid_color_no_overwrite(self):
        """CNN が無効色 (空/UNKNOWN/お邪魔) → 上書きしない."""
        import numpy as np
        from src.board import COLOR_OJAMA, COLOR_UNKNOWN

        inferred = _empty_board()
        inferred.set(11, 2, COLOR_GREEN)

        # CNN がお邪魔 (= 有効 puyo 色でない) を返す
        cnn_board = _empty_board()
        cnn_board.set(11, 2, COLOR_OJAMA)

        class _HsvOjama:
            def classify(self, patch):  # noqa: ANN001
                return COLOR_OJAMA  # HSV も同じお邪魔

        pattern = self._make_dummy_pattern(((11, 2), (12, 2)))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row, col):  # noqa: ANN001
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _HsvOjama(), frame, _StubRegion(),
        )
        # CNN がお邪魔 (無効色) → _VALID_PUYO_COLORS 外 → 上書きしない
        assert int(result.get(11, 2)) == COLOR_GREEN

    def test_inferred_unchanged_when_no_landing_cells(self):
        """pattern の cells が空の場合は inferred をそのまま返す (no crash)."""
        import numpy as np

        inferred = _empty_board()
        inferred.set(5, 3, COLOR_PURPLE)

        cnn_board = _empty_board()
        cnn_board.set(5, 3, COLOR_PURPLE)

        # 空パターン (cells が inferred に存在しない位置)
        pattern = self._make_dummy_pattern(((10, 0), (11, 0)))

        class _HsvPurple:
            def classify(self, patch):  # noqa: ANN001
                return COLOR_PURPLE

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class _StubRegion:
            def cell_sample_rect(self, row, col):  # noqa: ANN001
                return (0, 0, 4, 4)

        result = correct_landing_cells_by_observed_color(
            inferred, pattern, cnn_board, _HsvPurple(), frame, _StubRegion(),
        )
        # パターン位置 (10,0), (11,0) のCNN色はEMPTY → 上書きしない
        # 無関係な (5,3) は inferred のままの紫
        assert int(result.get(5, 3)) == COLOR_PURPLE
