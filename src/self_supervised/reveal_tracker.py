"""RevealTracker: 隠し段推論と後続 reveal の照合.

ぷよぷよでは row 0 (= 13 段目) は画面外の隠し段で、回し入れによって
puyo が積まれることがある。本クラスは:

    1. STABLE 時に `infer_hidden_row()` が出した隠し段確率分布を
       PendingInference として一時保存
    2. 後続フレームで「連鎖」「おじゃま落下」「ツモ落下」等で
       盤面状態が遷移した結果、もともと隠し段にあった puyo が
       row 1 (HIDDEN_ROWS == 1 の直下) に押し出されるケースを検出
    3. 押し出された色と、過去の推論確率分布を照合し、
       擬似ラベル (RevealEvent) として返す

ラベルの応用:
    - HiddenRowFineTuner が Platt scaling で `infer_hidden_row()` の
      出力確率を補正
    - 失敗事例 (推論色 != 観測色) は次バージョンの推論ロジック改善材料

設計の出発点:
    - 連鎖中・落下中は中途半端な盤面のため reveal 判定はスキップ
      (was_chain_or_drop=True 直後の STABLE 遷移でのみ有効)
    - 物理推論で「同列で row 0 から row 1 に降りた」と判断できる
      paint がある場合だけ reveal とみなす
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.board import (
    BOARD_COLS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)
from src.probabilistic_board import ProbabilisticBoard

# ============================
# 定数定義
# ============================

# pending を放置する最大秒数 (これを超えたら破棄)
DEFAULT_MAX_PENDING_AGE_SEC: float = 30.0

# 隠し段 reveal 後に row 1 (HIDDEN_ROWS) で観測される行
REVEAL_TARGET_ROW: int = HIDDEN_ROWS  # = 1

# 推論確率がこの値以上の色を「最尤予測」として扱う
PREDICTION_TIEBREAK_EPS: float = 1e-6


# ============================
# データクラス
# ============================


@dataclass
class PendingInference:
    """1 件の隠し段推論を保持する pending エントリ.

    Attributes:
        timestamp: 推論時のフレーム時刻 (秒)
        side: "1P" / "2P"
        inferred_dist: row 0 の各列に対する {color: prob} の dict.
            例: {(0, 2): {COLOR_BLUE: 1.0}}
        state_snapshot: prev_board / cur_board / next_pair などのデバッグ情報
    """

    timestamp: float
    side: str
    inferred_dist: dict[tuple[int, int], dict[int, float]]
    state_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RevealEvent:
    """隠し段推論が後続 reveal で確定した擬似ラベル 1 件.

    Attributes:
        timestamp: reveal 観測時刻
        original_inference_t: 元 PendingInference の timestamp
        side: "1P" / "2P"
        col: reveal が起きた列 (0..5)
        observed_color: 実際に観測された色
        predicted_dist: 元の推論分布 {color: prob}
        match: predicted の最尤色 == observed_color か
    """

    timestamp: float
    original_inference_t: float
    side: str
    col: int
    observed_color: int
    predicted_dist: dict[int, float]
    match: bool


# ============================
# RevealTracker
# ============================


class RevealTracker:
    """隠し段推論と後続 reveal を照合し擬似ラベルを抽出する.

    Usage:
        tracker = RevealTracker()
        # STABLE 確定時:
        tracker.add_inference(t, "1P", pboard, snapshot)
        # 連鎖/落下イベント直後の STABLE 遷移時:
        events = tracker.update(t, "1P", current_board, prev_board, True)
    """

    def __init__(
        self,
        max_pending_age_sec: float = DEFAULT_MAX_PENDING_AGE_SEC,
    ) -> None:
        if max_pending_age_sec <= 0:
            raise ValueError("max_pending_age_sec must be > 0")
        self._max_age = float(max_pending_age_sec)
        self.pending: list[PendingInference] = []

    # ------------------------------------------------------------------
    # 推論追加
    # ------------------------------------------------------------------

    def add_inference(
        self,
        t: float,
        side: str,
        inferred: ProbabilisticBoard,
        state_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """新しい推論結果を pending に追加する.

        infered の隠し段 (row 0) のうち、確率分布として「EMPTY 以外を
        含む可能性 > 0」の列だけを抽出する。
        """
        dist = self._extract_hidden_dist(inferred)
        if not dist:
            # 隠し段に意味のある推論が無ければ pending に追加しない
            return
        entry = PendingInference(
            timestamp=float(t),
            side=str(side),
            inferred_dist=dist,
            state_snapshot=dict(state_snapshot or {}),
        )
        self.pending.append(entry)

    @staticmethod
    def _extract_hidden_dist(
        inferred: ProbabilisticBoard,
    ) -> dict[tuple[int, int], dict[int, float]]:
        """ProbabilisticBoard の row 0 から「色付き確率 > 0」の列を取り出す."""
        out: dict[tuple[int, int], dict[int, float]] = {}
        for col in range(BOARD_COLS):
            cell = inferred.cell(0, col)
            colored_prob = sum(
                p for c, p in cell.probs.items()
                if c not in (COLOR_EMPTY, COLOR_UNKNOWN)
            )
            if colored_prob <= PREDICTION_TIEBREAK_EPS:
                continue
            out[(0, col)] = dict(cell.probs)
        return out

    # ------------------------------------------------------------------
    # reveal 検出
    # ------------------------------------------------------------------

    def update(
        self,
        t: float,
        side: str,
        current_board: Board,
        prev_board: Board,
        was_chain_or_drop: bool,
    ) -> list[RevealEvent]:
        """新 STABLE board で reveal 判定を行う.

        was_chain_or_drop=True 時のみ走査 (連鎖/落下イベント直後に呼ぶこと).
        該当する pending を消化し、RevealEvent を返す。
        """
        # 古い pending を破棄
        self.cleanup_stale(t)
        if not was_chain_or_drop:
            return []
        # 物理推論: row 1 で「prev EMPTY → cur 非 EMPTY (色付き)」になった列
        revealed_cols = self._detect_reveal_columns(prev_board, current_board)
        if not revealed_cols:
            return []
        # 該当 side の pending を新しい順に走査
        events: list[RevealEvent] = []
        consumed_idx: list[int] = []
        for idx in range(len(self.pending) - 1, -1, -1):
            entry = self.pending[idx]
            if entry.side != side:
                continue
            # entry の inferred_dist 列のうち revealed_cols と重なるものを照合
            matched = self._match_entry(entry, revealed_cols, t)
            if not matched:
                continue
            events.extend(matched)
            consumed_idx.append(idx)
            # 1 reveal イベントで消化したら次の pending には進まない
            # (古い pending は別タイミングで再消化される)
            break
        # 消化済を pending から削除
        for idx in consumed_idx:
            del self.pending[idx]
        return events

    @staticmethod
    def _detect_reveal_columns(
        prev_board: Board, current_board: Board,
    ) -> dict[int, int]:
        """row 1 (HIDDEN_ROWS) で reveal とみなせる列と観測色を返す.

        判定条件 (was_chain_or_drop=True 直後に呼ばれる前提):
            - cur row 1 が EMPTY 以外、かつ OJAMA でない
            - prev row 1 != cur row 1 (色が変化した、または初出現)
            - これにより、連鎖でもとの puyo が消えて隠し段から色付き puyo が
              押し出されてきたケース、および直接出現ケース両方を捉える。
            - prev row 1 と cur row 1 が同色 (連鎖で何も変化なし) は除外
        """
        out: dict[int, int] = {}
        for col in range(BOARD_COLS):
            try:
                prev_color = int(prev_board.get(REVEAL_TARGET_ROW, col))
                cur_color = int(current_board.get(REVEAL_TARGET_ROW, col))
            except (IndexError, ValueError):
                continue
            if cur_color in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA):
                continue
            if prev_color == cur_color:
                continue
            # cur が「色付き、prev と異なる」なら reveal 候補
            out[col] = cur_color
        return out

    @staticmethod
    def _match_entry(
        entry: PendingInference,
        revealed_cols: dict[int, int],
        t: float,
    ) -> list[RevealEvent]:
        """1 件の pending と reveal 列を照合して RevealEvent 一覧を生成."""
        events: list[RevealEvent] = []
        for (row, col), dist in entry.inferred_dist.items():
            if row != 0:
                continue
            if col not in revealed_cols:
                continue
            observed = revealed_cols[col]
            predicted_color = _argmax_color(dist)
            match = (predicted_color == observed)
            events.append(RevealEvent(
                timestamp=float(t),
                original_inference_t=entry.timestamp,
                side=entry.side,
                col=int(col),
                observed_color=int(observed),
                predicted_dist=dict(dist),
                match=bool(match),
            ))
        return events

    # ------------------------------------------------------------------
    # 期限切れ pending の破棄
    # ------------------------------------------------------------------

    def cleanup_stale(self, current_t: float) -> int:
        """max_pending_age_sec 経過した pending を破棄する.

        Returns:
            削除された件数
        """
        cutoff = float(current_t) - self._max_age
        before = len(self.pending)
        self.pending = [
            e for e in self.pending if e.timestamp >= cutoff
        ]
        return before - len(self.pending)

    def clear(self) -> None:
        """全 pending を破棄 (試合切替時など)."""
        self.pending.clear()

    def __len__(self) -> int:
        return len(self.pending)


# ============================
# helper
# ============================


def _argmax_color(dist: dict[int, float]) -> int:
    """確率分布から最尤色を返す. 空 dict の場合は EMPTY."""
    if not dist:
        return COLOR_EMPTY
    # EMPTY と UNKNOWN は除外して最尤色を取る (色付き予測を比較対象とする)
    color_only = {
        c: p for c, p in dist.items()
        if c not in (COLOR_EMPTY, COLOR_UNKNOWN)
    }
    if color_only:
        return max(color_only.items(), key=lambda kv: kv[1])[0]
    return max(dist.items(), key=lambda kv: kv[1])[0]


__all__ = [
    "DEFAULT_MAX_PENDING_AGE_SEC",
    "PendingInference",
    "REVEAL_TARGET_ROW",
    "RevealEvent",
    "RevealTracker",
]
