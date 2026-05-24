"""StatefulBoardTracker の単体テスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.stateful_board_tracker import StatefulBoardTracker


def _board(cells: dict[tuple[int, int], int]) -> Board:
    """dict 表記で盤面を作る。指定なしセルは空。"""
    b = Board()
    for (r, c), v in cells.items():
        b.set(r, c, v)
    return b


class TestBootstrap:
    def test_initial_update_accepts_everything(self) -> None:
        """最初の observation はそのまま確定にする。"""
        tr = StatefulBoardTracker()
        obs = _board({(5, 0): COLOR_RED, (5, 1): COLOR_BLUE})
        out = tr.update(obs)
        assert out.get(5, 0) == COLOR_RED
        assert out.get(5, 1) == COLOR_BLUE
        assert tr.initialized

    def test_reset_clears_state(self) -> None:
        tr = StatefulBoardTracker()
        tr.update(_board({(0, 0): COLOR_RED}))
        assert tr.initialized
        tr.reset()
        assert not tr.initialized

    def test_reset_with_board_sets_initial(self) -> None:
        tr = StatefulBoardTracker()
        tr.reset(_board({(1, 1): COLOR_GREEN}))
        assert tr.initialized
        assert tr.current.get(1, 1) == COLOR_GREEN


class TestRejectColorToColor:
    """色→色 直接遷移の棄却。"""

    def test_color_to_different_color_rejected_without_chain(self) -> None:
        """連鎖なしで色が突然変わる → 棄却。"""
        tr = StatefulBoardTracker()
        tr.update(_board({(6, 0): COLOR_RED}))  # bootstrap
        out = tr.update(_board({(6, 0): COLOR_BLUE}))  # CNN 誤認
        assert out.get(6, 0) == COLOR_RED, "色→色は棄却されるはず"
        assert tr.last_stats.rejected == 1

    def test_color_to_color_accepted_with_chain_event(self) -> None:
        """連鎖消去が同時に起きていれば、gravity shift として色→色も accept。"""
        tr = StatefulBoardTracker()
        # 初期状態: 最下段に赤 4 個 + 上に青 1 個
        tr.update(_board({
            (12, 0): COLOR_RED, (12, 1): COLOR_RED,
            (12, 2): COLOR_RED, (12, 3): COLOR_RED,
            (11, 2): COLOR_BLUE,
        }))
        # 連鎖発生: 赤 4 個が消え、青が下に落ちる
        out = tr.update(_board({
            (12, 0): COLOR_EMPTY, (12, 1): COLOR_EMPTY,
            (12, 2): COLOR_BLUE, (12, 3): COLOR_EMPTY,  # 青が 2 段下に
            (11, 2): COLOR_EMPTY,
        }))
        # chain_event=True (c2e=4)
        assert tr.last_stats.chain_event is True
        # 青が落ちた変化は accept
        assert out.get(12, 2) == COLOR_BLUE
        assert out.get(12, 0) == COLOR_EMPTY  # 赤消滅


class TestRejectColorToOjama:
    def test_color_to_ojama_rejected(self) -> None:
        """おじゃまが既存色を上書きしない (色→おじゃま は誤認)。"""
        tr = StatefulBoardTracker()
        tr.update(_board({(5, 0): COLOR_RED}))
        out = tr.update(_board({(5, 0): COLOR_OJAMA}))
        assert out.get(5, 0) == COLOR_RED
        assert tr.last_stats.rejected == 1


class TestAcceptNewDrops:
    def test_empty_to_color_accepted(self) -> None:
        """空→色: 新規落下として受理。"""
        tr = StatefulBoardTracker()
        tr.update(Board())  # bootstrap: 全空
        out = tr.update(_board({(2, 3): COLOR_YELLOW}))
        assert out.get(2, 3) == COLOR_YELLOW
        assert tr.last_stats.empty_to_color == 1
        assert tr.last_stats.accepted == 1

    def test_empty_to_ojama_accepted(self) -> None:
        """空→おじゃま: おじゃま降下として受理。"""
        tr = StatefulBoardTracker()
        tr.update(Board())
        out = tr.update(_board({(0, 0): COLOR_OJAMA}))
        assert out.get(0, 0) == COLOR_OJAMA


class TestChainErase:
    """色→空 の連鎖消去判定。"""

    def test_single_color_to_empty_rejected(self) -> None:
        """1 個だけ色→空は誤認として棄却。"""
        tr = StatefulBoardTracker()
        tr.update(_board({(5, 0): COLOR_RED, (5, 1): COLOR_BLUE}))
        # 単独で赤が消える (実際は CNN が halo で empty に読んだケース)
        out = tr.update(_board({(5, 0): COLOR_EMPTY, (5, 1): COLOR_BLUE}))
        assert out.get(5, 0) == COLOR_RED, "単独 color→empty は棄却"
        assert tr.last_stats.chain_event is False

    def test_four_color_to_empty_accepted_as_chain(self) -> None:
        """4 個同時に色→空 → 連鎖消去として accept。"""
        tr = StatefulBoardTracker()
        tr.update(_board({
            (12, 0): COLOR_GREEN, (12, 1): COLOR_GREEN,
            (12, 2): COLOR_GREEN, (12, 3): COLOR_GREEN,
        }))
        out = tr.update(Board())  # 全空 (消えた)
        assert tr.last_stats.chain_event is True
        for c in range(4):
            assert out.get(12, c) == COLOR_EMPTY


class TestPersistence:
    """ぷよが消えなければ色は維持される。"""

    def test_color_persists_when_observation_noisy(self) -> None:
        """観測が halo ノイズで乱れても、確定色が維持される。"""
        tr = StatefulBoardTracker()
        # 確定状態: 列 0 に赤 5 個
        initial = Board()
        for r in range(8, 13):
            initial.set(r, 0, COLOR_RED)
        tr.update(initial)
        # ノイズ観測: (8,0) だけ halo で緑に読まれる
        noisy = Board()
        for r in range(8, 13):
            noisy.set(r, 0, COLOR_RED)
        noisy.set(8, 0, COLOR_GREEN)  # halo 誤認
        out = tr.update(noisy)
        # chain_event=False, 色→色 棄却
        assert out.get(8, 0) == COLOR_RED
        assert tr.last_stats.color_to_color == 1
        assert tr.last_stats.rejected == 1


class TestTransitionStats:
    def test_stats_populated(self) -> None:
        """TransitionStats が正しくカウントされる。"""
        tr = StatefulBoardTracker()
        tr.update(_board({(12, 0): COLOR_RED, (11, 0): COLOR_BLUE}))
        tr.update(_board({
            (12, 0): COLOR_EMPTY,    # c2e
            (11, 0): COLOR_GREEN,    # c2c (棄却されるが集計はされる)
            (10, 0): COLOR_YELLOW,   # e2c
            (9, 0): COLOR_OJAMA,     # e2c (空→お邪魔)
        }))
        stats = tr.last_stats
        assert stats.color_to_empty == 1
        assert stats.empty_to_color == 2
        assert stats.color_to_color == 1
        # chain_event=False (c2e=1 < 4), c2c 棄却、e2c 2件 accept、c2e 棄却
        assert stats.chain_event is False
        assert stats.accepted == 2  # e2c 2件のみ
        assert stats.rejected == 2  # c2e + c2c


class TestFullBoardScenario:
    """より実戦に近いシナリオ。"""

    def test_ui_halo_injection_rejected(self) -> None:
        """
        安定した盤面に halo が乗って色が変わって見えるケース。
        3 フレーム連続で halo が入っても全て棄却される (色→色)。
        """
        tr = StatefulBoardTracker()
        # 初期: 赤 3 個 (積み上げ途中)
        initial = _board({
            (12, 2): COLOR_RED, (11, 2): COLOR_RED, (10, 2): COLOR_RED,
        })
        tr.update(initial)
        # halo で (10, 2) が緑に見える
        halo = _board({
            (12, 2): COLOR_RED, (11, 2): COLOR_RED, (10, 2): COLOR_GREEN,
        })
        for _ in range(3):
            tr.update(halo)
        # 3 回とも棄却されて赤のまま
        assert tr.current.get(10, 2) == COLOR_RED

    def test_real_puyo_fall_accepted(self) -> None:
        """新規ぷよ落下は受理される。"""
        tr = StatefulBoardTracker()
        tr.update(Board())  # 空盤面
        # ぷよペア落下: (12, 0) と (11, 0) に同時着地
        out = tr.update(_board({
            (12, 0): COLOR_PURPLE,
            (11, 0): COLOR_YELLOW,
        }))
        assert out.get(12, 0) == COLOR_PURPLE
        assert out.get(11, 0) == COLOR_YELLOW
