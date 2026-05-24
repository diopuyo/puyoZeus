"""drift detector + re-sync (Phase B-6).

InferenceBoardGenerator が返す推論盤面と CNN 出力の cell 単位差分を監視し、
乖離が連続して閾値を超えた場合に state machine の re-sync を要請する。

新方針 (project_recognition_strategy_pivot) における安全装置:
    state machine の遷移検出が誤った場合、推論盤面と実際 (CNN 出力) が
    乖離する。drift detector が連続乖離を検知したら、上位レイヤーは
    `BoardStateMachine.reset()` + 最新 CNN 盤面で confirmed を上書き
    する形で再同期する。

注意:
    COLOR_UNKNOWN セル (隠し段の量子状態など) は照合対象から除外する。
    NON-STABLE state では推論盤面が動的に変わるが、本 detector は単に
    cell 一致数を見るだけで state は意識しない (caller 側で「STABLE 中の
    乖離のみ評価」「アクション中は drift を許容」など方針を決める前提)。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, Board

# ============================
# 定数
# ============================

# 1 frame の乖離 cell 数閾値。これ以上で drift 候補。
DEFAULT_DRIFT_CELL_THRESHOLD: int = 6

# 連続 drift frame 数。これ以上で re-sync 要請。
DEFAULT_DRIFT_FRAME_THRESHOLD: int = 3


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class DriftResult:
    """1 frame 分の drift 判定結果.

    Attributes:
        mismatch_count: 現 frame の不一致 cell 数 (UNKNOWN 除外)
        consecutive_count: 連続 drift frame 数
        is_drift: 現 frame が drift 判定 (mismatch_count >= cell_threshold)
        needs_resync: state machine reset を要請すべきか
    """

    mismatch_count: int
    consecutive_count: int
    is_drift: bool
    needs_resync: bool


# ============================
# DriftDetector
# ============================


class DriftDetector:
    """推論盤面と CNN 盤面の連続乖離を監視する.

    Usage:
        det = DriftDetector()
        for frame in stream:
            ctx = sm.update(frame_idx, signals)
            inferred = gen.generate(ctx, signals.chain_event, signals.time_sec)
            res = det.update(inferred, signals.cnn_board)
            if res.needs_resync:
                sm.reset(keep_match_state=True)
                det.reset()
    """

    def __init__(
        self,
        cell_threshold: int = DEFAULT_DRIFT_CELL_THRESHOLD,
        frame_threshold: int = DEFAULT_DRIFT_FRAME_THRESHOLD,
    ) -> None:
        if cell_threshold < 1:
            raise ValueError(f"cell_threshold must be >=1 (got {cell_threshold})")
        if frame_threshold < 1:
            raise ValueError(
                f"frame_threshold must be >=1 (got {frame_threshold})"
            )
        self._cell_th = cell_threshold
        self._frame_th = frame_threshold
        self._consec = 0

    @property
    def consecutive_drift_count(self) -> int:
        return self._consec

    @property
    def cell_threshold(self) -> int:
        return self._cell_th

    @property
    def frame_threshold(self) -> int:
        return self._frame_th

    def reset(self) -> None:
        """連続カウンタをクリア (re-sync 後に caller が呼ぶ)."""
        self._consec = 0

    def update(
        self, inferred: Board | None, cnn: Board | None,
    ) -> DriftResult:
        """新 frame を投入、drift 判定を返す."""
        if inferred is None or cnn is None:
            # 比較不能 frame は drift 判定をスキップ、連続カウンタも維持
            return DriftResult(
                mismatch_count=0,
                consecutive_count=self._consec,
                is_drift=False,
                needs_resync=False,
            )
        mismatch = self._count_mismatch(inferred, cnn)
        is_drift = mismatch >= self._cell_th
        if is_drift:
            self._consec += 1
        else:
            self._consec = 0
        needs_resync = self._consec >= self._frame_th
        return DriftResult(
            mismatch_count=mismatch,
            consecutive_count=self._consec,
            is_drift=is_drift,
            needs_resync=needs_resync,
        )

    @staticmethod
    def _count_mismatch(a: Board, b: Board) -> int:
        n = 0
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                ca = a.get(r, c)
                cb = b.get(r, c)
                if ca == COLOR_UNKNOWN or cb == COLOR_UNKNOWN:
                    continue
                if ca != cb:
                    n += 1
        return n


__all__ = [
    "DEFAULT_DRIFT_CELL_THRESHOLD",
    "DEFAULT_DRIFT_FRAME_THRESHOLD",
    "DriftDetector",
    "DriftResult",
]
