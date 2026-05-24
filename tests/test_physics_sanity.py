"""physics_sanity.PhysicsSanityChecker の単体テスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    HIDDEN_ROWS,
    Board,
)
from src.physics_sanity import (
    PhysicsSanityChecker,
    PhysicsViolation,
    ViolationKind,
)


def _make_empty_board() -> Board:
    """すべて空の盤面を作る。"""
    return Board()


class TestAirborneDetection:
    """空中浮遊ぷよ検出のテスト。"""

    def test_no_violation_on_empty_board(self) -> None:
        board = _make_empty_board()
        violations = PhysicsSanityChecker().check(board)
        assert violations == []

    def test_no_violation_for_stacked_column(self) -> None:
        """最下段から連続して積まれた列は違反なし。"""
        board = _make_empty_board()
        board.set(BOARD_ROWS - 1, 0, COLOR_RED)
        board.set(BOARD_ROWS - 2, 0, COLOR_BLUE)
        board.set(BOARD_ROWS - 3, 0, COLOR_GREEN)
        violations = PhysicsSanityChecker().check(board)
        airborne = [v for v in violations if v.kind == ViolationKind.AIRBORNE]
        assert airborne == []

    def test_detect_single_airborne(self) -> None:
        """真下が空の 1 セル → 違反 1 件。"""
        board = _make_empty_board()
        board.set(5, 2, COLOR_RED)  # 空中浮遊 (下が全て空)
        violations = PhysicsSanityChecker().check(board)
        airborne = [v for v in violations if v.kind == ViolationKind.AIRBORNE]
        assert len(airborne) == 1
        assert airborne[0].row == 5
        assert airborne[0].col == 2
        assert airborne[0].color == COLOR_RED

    def test_detect_stacked_airborne(self) -> None:
        """上 2 段が浮いている → 違反 2 件 (上側と下側の境界)。"""
        board = _make_empty_board()
        board.set(3, 1, COLOR_RED)
        board.set(4, 1, COLOR_BLUE)  # この下が空なのでここが違反点
        # row 5-11 は空、row 12 (最下段) も空
        violations = PhysicsSanityChecker().check(board)
        airborne = [v for v in violations if v.kind == ViolationKind.AIRBORNE]
        # (3,1) 直下 (4,1) は非空なので違反ではない
        # (4,1) 直下 (5,1) は空なので違反
        assert len(airborne) == 1
        assert airborne[0].row == 4
        assert airborne[0].col == 1

    def test_hidden_row_excluded(self) -> None:
        """隠し段 (row < HIDDEN_ROWS) のセルは浮遊判定から除外される。"""
        board = _make_empty_board()
        # 隠し段に色、可視段以降は全て空
        for r in range(HIDDEN_ROWS):
            board.set(r, 0, COLOR_RED)
        violations = PhysicsSanityChecker().check(board)
        airborne = [v for v in violations if v.kind == ViolationKind.AIRBORNE]
        assert airborne == []

    def test_floor_row_not_violation(self) -> None:
        """最下段セルは床なので違反にならない。"""
        board = _make_empty_board()
        board.set(BOARD_ROWS - 1, 3, COLOR_RED)
        violations = PhysicsSanityChecker().check(board)
        airborne = [v for v in violations if v.kind == ViolationKind.AIRBORNE]
        assert airborne == []


class TestUnresolvedChainDetection:
    """4+ 連結未消去検出のテスト。"""

    def test_no_violation_on_small_cluster(self) -> None:
        """3 連結は違反にならない。"""
        board = _make_empty_board()
        # 最下段に 3 連結 (赤)
        board.set(BOARD_ROWS - 1, 0, COLOR_RED)
        board.set(BOARD_ROWS - 1, 1, COLOR_RED)
        board.set(BOARD_ROWS - 1, 2, COLOR_RED)
        violations = PhysicsSanityChecker().check(board)
        chains = [v for v in violations if v.kind == ViolationKind.UNRESOLVED_CHAIN]
        assert chains == []

    def test_detect_4_horizontal(self) -> None:
        """4 横並び連結 → UNRESOLVED_CHAIN 違反。"""
        board = _make_empty_board()
        for c in range(4):
            board.set(BOARD_ROWS - 1, c, COLOR_RED)
        violations = PhysicsSanityChecker().check(board)
        chains = [v for v in violations if v.kind == ViolationKind.UNRESOLVED_CHAIN]
        assert len(chains) == 1
        assert chains[0].color == COLOR_RED

    def test_ojama_excluded_from_chain(self) -> None:
        """おじゃまは 4 個連結でも UNRESOLVED_CHAIN にならない。"""
        board = _make_empty_board()
        for c in range(4):
            board.set(BOARD_ROWS - 1, c, COLOR_OJAMA)
        violations = PhysicsSanityChecker().check(board)
        chains = [v for v in violations if v.kind == ViolationKind.UNRESOLVED_CHAIN]
        assert chains == []


class TestCombined:
    """複合ケース。"""

    def test_violations_reported_together(self) -> None:
        """浮遊と 4+ 連結が混在する場合、両方報告される。"""
        board = _make_empty_board()
        # 空中浮遊 (row 3)
        board.set(3, 0, COLOR_BLUE)
        # 最下段 4 連結
        for c in range(4):
            board.set(BOARD_ROWS - 1, c, COLOR_GREEN)
        violations = PhysicsSanityChecker().check(board)
        kinds = {v.kind for v in violations}
        assert ViolationKind.AIRBORNE in kinds
        assert ViolationKind.UNRESOLVED_CHAIN in kinds
