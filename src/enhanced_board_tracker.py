"""V2.4: V2.1 (NextLink) + V2.3 (Connectivity) を組み合わせた拡張トラッカー。

既存 StatefulBoardTracker を内部に持ち、観測の前段に refiner を 2 段重ねる:

    観測 → V2.1 NextLinkedColorRefiner (前盤面+next_pair で色補正)
         → V2.3 ConnectivityShapeRefiner (孤立異色セルを多数色に)
         → StatefulBoardTracker (物理遷移ルール棄却)
         → 確定状態

V2.2 PairAppearanceConsistency は「相方位置候補の提示」のみで補正は行わない
ため、本トラッカーでは検出ログとしてのみ利用 (last_stats に件数を記録)。

利用例:
    tracker = EnhancedBoardTracker()
    for obs, next_pair in observations_with_next:
        stable = tracker.update(obs, next_pair=next_pair)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import Board
from src.color_oscillation_filter import ColorOscillationFilter
from src.connectivity_refiner import ConnectivityShapeRefiner
from src.next_linked_refiner import NextLinkedColorRefiner
from src.pair_appearance import PairAppearanceConsistency
from src.physical_sanity_refiner import PhysicalSanityRefiner
from src.stateful_board_tracker import StatefulBoardTracker


@dataclass
class EnhancedTrackingStats:
    """1 update 分の補正統計。"""
    next_link_corrected: int = 0
    connectivity_corrected: int = 0
    pair_inconsistent: int = 0  # 1 セル / 3 セル新規出現の検出回数
    oscillation_corrected: int = 0  # B2: 色振動セル UNKNOWN 化数
    sanity_corrected: int = 0   # W6: 4+ 連結補正セル数
    stateful_accepted: int = 0
    stateful_rejected: int = 0


class EnhancedBoardTracker:
    """V2.1 + V2.3 + StatefulBoardTracker を統合した時系列フィルタ。

    next_pair (NextDetector の出力) を update に渡すと、V2.1 が次フレームの
    新規出現セル色を補正してくれる。
    """

    def __init__(
        self,
        next_refiner: NextLinkedColorRefiner | None = None,
        shape_refiner: ConnectivityShapeRefiner | None = None,
        stateful: StatefulBoardTracker | None = None,
        sanity_refiner: PhysicalSanityRefiner | None = None,
        oscillation_filter: ColorOscillationFilter | None = None,
    ) -> None:
        self._next_refiner = next_refiner or NextLinkedColorRefiner()
        self._shape_refiner = shape_refiner or ConnectivityShapeRefiner()
        self._pair_checker = PairAppearanceConsistency()
        self._sanity_refiner = sanity_refiner or PhysicalSanityRefiner()
        self._oscillation_filter = (
            oscillation_filter or ColorOscillationFilter()
        )
        self._stateful = stateful or StatefulBoardTracker()
        self._prev_board: Board | None = None
        self._prev_next_pair: tuple[int, int] | None = None
        self._last_stats: EnhancedTrackingStats = EnhancedTrackingStats()

    @property
    def initialized(self) -> bool:
        return self._stateful.initialized

    @property
    def current(self) -> Board:
        return self._stateful.current

    @property
    def last_stats(self) -> EnhancedTrackingStats:
        return self._last_stats

    def reset(self, board: Board | None = None) -> None:
        """状態リセット。試合間で呼ぶ。"""
        self._stateful.reset(board)
        self._oscillation_filter.reset()
        self._prev_board = board
        self._prev_next_pair = None
        self._last_stats = EnhancedTrackingStats()

    def update(
        self,
        observation: Board,
        next_pair: tuple[int, int] | None = None,
        skip_sanity_refiner: bool = False,
    ) -> Board:
        """観測を 4 段フィルタで処理し、確定状態を返す。

        Args:
            observation: CNN/HSV が予測した新しい盤面。
            next_pair: 観測時刻の **直前** の NextDetector 出力
                (観測フレームに新規落下したペアの真の色)。None なら V2.1 スキップ。
            skip_sanity_refiner: True なら W6 (4+ 連結補正) をスキップ。
                連鎖アニメ中・落下中フレームで指定する。
        """
        stats = EnhancedTrackingStats()

        # 0. B2: 色振動セル UNKNOWN 化 (raw 観測の履歴ベース)
        from src.board import COLOR_UNKNOWN, BOARD_ROWS, BOARD_COLS
        before_osc = observation
        refined = self._oscillation_filter.update(observation)
        # 振動補正セル数を集計
        n_osc = 0
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if (
                    int(before_osc.get(r, c)) != COLOR_UNKNOWN
                    and int(refined.get(r, c)) == COLOR_UNKNOWN
                ):
                    n_osc += 1
        stats.oscillation_corrected = n_osc

        # 1. V2.1: NextLinkedColorRefiner (前盤面 + 直前 next_pair で補正)
        if (
            self._prev_board is not None
            and self._prev_next_pair is not None
        ):
            v21 = self._next_refiner.refine(
                self._prev_board, refined, self._prev_next_pair,
            )
            refined = v21.refined
            stats.next_link_corrected = v21.n_corrected

        # 2. V2.2: ペア整合性チェック (検出のみ、ログ用)
        if self._prev_board is not None:
            pc = self._pair_checker.check(self._prev_board, refined)
            if not pc.is_consistent:
                stats.pair_inconsistent = 1

        # 3. V2.3: ConnectivityShapeRefiner (孤立異色セルを多数色化)
        v23 = self._shape_refiner.refine(refined)
        refined = v23.refined
        stats.connectivity_corrected = v23.n_corrected

        # 4. W6: PhysicalSanityRefiner (4+ 連結 → 1 セル UNKNOWN)
        if not skip_sanity_refiner:
            w6 = self._sanity_refiner.refine(refined)
            refined = w6.refined
            stats.sanity_corrected = w6.n_corrected

        # 5. 既存 StatefulBoardTracker (物理遷移ルール棄却)
        stable = self._stateful.update(refined)
        if self._stateful.last_stats is not None:
            stats.stateful_accepted = self._stateful.last_stats.accepted
            stats.stateful_rejected = self._stateful.last_stats.rejected

        # 6. 浮遊ぷよ削除 (Stateful 後の最終確定盤面に重力詰め)
        # Stateful が「色 → EMPTY」を chain_event=False で棄却した場合に
        # 残ってしまう浮遊ぷよを削除する。
        from src.board_rules import clear_floating_above_gap
        stable = clear_floating_above_gap(
            stable, min_gap=1, skip_hidden=True,
        )
        # _stateful.current にも反映 (内部状態を一致させる)
        self._stateful._current = stable

        # 7. 次フレーム用に保存 (next_pair は **観測時に観測された値** を保存)
        # 「観測時の next_pair が次フレームの落下を予言する」ので、
        # update 時の next_pair が次の update の prev_next_pair になる。
        self._prev_board = stable.copy()
        self._prev_next_pair = next_pair
        self._last_stats = stats
        return stable


__all__ = [
    "EnhancedBoardTracker",
    "EnhancedTrackingStats",
]
