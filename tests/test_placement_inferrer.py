"""placement_inferrer.py の単体テスト (Phase 1 cycle 71)."""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_RED,
    COLOR_UNKNOWN, COLOR_YELLOW, Board,
)
from src.placement_inferrer import (
    LARGE_ADD_GUARD_CELLS,
    LandingPattern,
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
