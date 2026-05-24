"""
過去状態を保持しながら物理的に妥当な変化だけを受理する盤面トラッカー。

ぷよぷよのルールに基づくと、単一フレームの CNN 観測に含まれる誤認の多くは
物理的に不可能な状態遷移である:
    - 色 A → 色 B への直接遷移 (空を挟まない色変化はあり得ない)
    - 単独セルの color → empty (連鎖消去なら最低 4 個同時消滅)
    - 色 → OJAMA (お邪魔が既存色を上書きすることはない)

これらを棄却して過去状態を保持することで、halo/UI/瞬間的エフェクトを
ノイズとして除去できる。連鎖消去が検出された場合 (同フレームで 4+ セルが
色→空 になった場合) は連鎖+重力整合として全変化を受理する。

想定利用法:
    tracker = StatefulBoardTracker()
    for obs in cnn_observations:
        stable_board = tracker.update(obs)
    # stable_board は物理的に妥当な変化だけ反映した確定状態

前段に TemporalSmoother を入れて瞬間ノイズを多数決で減らしてから
StatefulBoardTracker に渡す 2 段構えが効果的。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    Board,
)
from src.chain import MIN_ERASE_COUNT


@dataclass(frozen=True)
class TransitionStats:
    """1 回の update における遷移統計 (デバッグ/可視化用)。"""
    color_to_empty: int       # 色→空 の数 (連鎖消去候補)
    empty_to_color: int       # 空→色 の数 (新規落下候補)
    color_to_color: int       # 色→色 直接遷移 (通常は誤認)
    color_to_ojama: int       # 色→お邪魔 (誤認)
    chain_event: bool         # 連鎖消去イベントとみなされたか
    accepted: int             # 実際に反映された変化数
    rejected: int             # 棄却した変化数


class StatefulBoardTracker:
    """
    過去状態を持ちながら物理的に妥当な観測だけを受理する。

    Usage:
        tracker = StatefulBoardTracker()
        stable = tracker.update(observed_board)

    初回 update 時は観測をそのまま確定状態にする (ブート)。
    以降は観測と確定状態の差分を物理ルールでフィルタする。
    """

    def __init__(self) -> None:
        self._current: Board = Board()
        self._initialized: bool = False
        self._last_stats: TransitionStats | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_stats(self) -> TransitionStats | None:
        """直近の update の遷移統計を返す (デバッグ用)。"""
        return self._last_stats

    @property
    def current(self) -> Board:
        """現在の確定状態を返す (コピーではない)。"""
        return self._current

    def reset(self, board: Board | None = None) -> None:
        """確定状態を空に戻す。board 指定時はそれで初期化。"""
        self._current = board if board is not None else Board()
        self._initialized = board is not None
        self._last_stats = None

    def update(
        self,
        observation: Board,
        expected_new_colors: "set[int] | None" = None,
    ) -> Board:
        """
        観測を受け取り、物理ルールで受理できる変化だけ反映して確定状態を返す。

        Args:
            observation: CNN が予測した新しい盤面。
            expected_new_colors: T-v2-K: 新規 EMPTY→色 遷移時に、新しい色が
                この集合になければ棄却する (NextDetector の next/dnext + 既存盤面色)。
                None なら色チェックなし (従来動作)。

        Returns:
            Board: 更新後の確定状態 (内部バッファへの参照、読み取り専用で扱う)。
        """
        if not self._initialized:
            # ブート: 最初の観測をそのまま確定にする
            self._current = self._copy_board(observation)
            self._initialized = True
            self._last_stats = TransitionStats(
                color_to_empty=0, empty_to_color=0,
                color_to_color=0, color_to_ojama=0,
                chain_event=False, accepted=0, rejected=0,
            )
            return self._current

        # 差分分析
        c2e, e2c, c2c, c2o = self._count_transitions(observation)
        chain_event = c2e >= MIN_ERASE_COUNT

        accepted = 0
        rejected = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                old = self._current.get(row, col)
                new = observation.get(row, col)
                if old == new:
                    continue
                if not self._is_allowed(old, new, chain_event):
                    rejected += 1
                    continue
                # T-v2-K: 新規 EMPTY→色 遷移は期待色集合と照合
                if (
                    old == COLOR_EMPTY and new != COLOR_EMPTY
                    and expected_new_colors is not None
                    and new not in expected_new_colors
                ):
                    rejected += 1
                    continue
                self._current.set(row, col, new)
                accepted += 1

        self._last_stats = TransitionStats(
            color_to_empty=c2e, empty_to_color=e2c,
            color_to_color=c2c, color_to_ojama=c2o,
            chain_event=chain_event, accepted=accepted, rejected=rejected,
        )
        return self._current

    def _count_transitions(self, observation: Board) -> tuple[int, int, int, int]:
        """観測との差分を 4 種類に分類してカウントする。"""
        c2e = 0
        e2c = 0
        c2c = 0
        c2o = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                old = self._current.get(row, col)
                new = observation.get(row, col)
                if old == new:
                    continue
                if old != COLOR_EMPTY and new == COLOR_EMPTY:
                    c2e += 1
                elif old == COLOR_EMPTY and new != COLOR_EMPTY:
                    e2c += 1
                elif old != COLOR_EMPTY and new != COLOR_EMPTY:
                    if new == COLOR_OJAMA:
                        c2o += 1
                    else:
                        c2c += 1
        return c2e, e2c, c2c, c2o

    @staticmethod
    def _is_allowed(old: int, new: int, chain_event: bool) -> bool:
        """
        1 セルの遷移 (old → new) を受理するかどうかを返す。

        Rules:
            - old == new: 変化なし (呼び出し元でスキップ想定、ここでは False を返す)
            - EMPTY → 色 (お邪まも含む): accept (新規落下)
            - 色 → EMPTY: chain_event のみ accept (連鎖消去 + gravity shift)
            - 色 → 色 (異色): chain_event のみ accept (gravity shift で上から落ちてきた)
            - 色 → OJAMA: reject (お邪まが色を上書きしない)
        """
        if old == new:
            return False
        if old == COLOR_EMPTY:
            # 空から何かになる: 落下を認める
            return True
        # old != EMPTY
        if new == COLOR_EMPTY:
            # 色が消えた: 連鎖消去時のみ認める
            return chain_event
        # old != EMPTY, new != EMPTY
        if new == COLOR_OJAMA:
            # 色がおじゃまに置き換わることはない
            return False
        # 色 → 色 (異色): gravity shift なら認める
        return chain_event

    @staticmethod
    def _copy_board(src: Board) -> Board:
        """Board の浅いコピー (setセル単位)。"""
        dst = Board()
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                dst.set(row, col, src.get(row, col))
        return dst
