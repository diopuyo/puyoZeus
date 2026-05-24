"""HiddenRowValidator: 隠し段推論の自己教師あり学習用 validator.

ロジック:
    1. 各 STABLE frame で `infer_hidden_row()` の確率分布を取得
    2. RevealTracker に PendingInference として蓄積
    3. 「直前 STABLE → CHAIN/OJAMA/TSUMO 経由 → 次 STABLE」の遷移時に
       row 1 (HIDDEN_ROWS) で revealed 列が観測されたら照合
    4. 照合結果 (RevealEvent) を PseudoLabelSample 形式で emit

擬似ラベル形式:
    component="hidden_row"
    input_data={
        "predicted_dist": {color: prob},
        "side": "1P"/"2P",
        "col": int,
        "predicted_color": int,
    }
    label=observed_color (int)
    confidence=match なら 1.0、不一致なら 0.0
        (calibration 学習側で重み付けせず、両方とも記録する)
    metadata={frame_idx, original_t, t_sec, ...}
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.board import (
    HIDDEN_ROWS,
    Board,
)
from src.board_state_machine import BoardState
from src.hidden_row_inferrer import infer_hidden_row
from src.self_supervised.cross_validator import CrossValidator
from src.self_supervised.pseudo_label import PseudoLabelSample
from src.self_supervised.reveal_tracker import (
    DEFAULT_MAX_PENDING_AGE_SEC,
    RevealEvent,
    RevealTracker,
)

# 擬似ラベル component 名 (pseudo_label モジュールには未定義のため文字列直書き)
COMPONENT_HIDDEN_ROW: str = "hidden_row"

# 配列に収まる最大 pending 推論数 (古いものから dropping)
MAX_PENDING_PER_SIDE: int = 256

# 照合一致時 / 不一致時の信頼度
MATCH_CONFIDENCE: float = 1.0
MISMATCH_CONFIDENCE: float = 0.0

# Phase I 改良: row 1 (HIDDEN_ROWS) で色付き puyo を観測した時点で
# 直近 pending との照合を許可 (action_seen 厳格条件を緩和)。
# 5fps サンプリングで STABLE → CHAIN → STABLE 遷移を捉え損ねた場合の救済。
LENIENT_REVEAL_CONFIDENCE: float = 0.85


class HiddenRowValidator(CrossValidator):
    """隠し段推論の自己教師あり学習用 validator.

    State machine 駆動で動作:
        - state==STABLE → 推論を取得し pending に追加
        - 直前 frame が CHAIN/TSUMO_FALL/OJAMA_FALL かつ現在 STABLE
          → reveal 判定を実行
    """

    def __init__(
        self,
        max_pending_age_sec: float = DEFAULT_MAX_PENDING_AGE_SEC,
        enable_lenient_reveal: bool = True,
    ) -> None:
        super().__init__()
        self._tracker_1p = RevealTracker(max_pending_age_sec=max_pending_age_sec)
        self._tracker_2p = RevealTracker(max_pending_age_sec=max_pending_age_sec)
        # 直前 frame の側別 (board, state, next_pair) を保持
        self._prev_state_1p: BoardState | None = None
        self._prev_state_2p: BoardState | None = None
        self._prev_board_1p: Board | None = None
        self._prev_board_2p: Board | None = None
        self._prev_next_pair_1p: tuple[int, int] | None = None
        self._prev_next_pair_2p: tuple[int, int] | None = None
        # アクション系 state を経由したフラグ (chain/ojama/tsumo を辿ったか)
        self._action_seen_1p: bool = False
        self._action_seen_2p: bool = False
        # Phase I 改良: lenient モードでは row 1 色変化を観測したら
        # action_seen に関わらず reveal 候補化 (5fps サンプリング救済策)
        self._enable_lenient_reveal = bool(enable_lenient_reveal)

    def reset(self) -> None:
        super().reset()
        self._tracker_1p.clear()
        self._tracker_2p.clear()
        self._prev_state_1p = None
        self._prev_state_2p = None
        self._prev_board_1p = None
        self._prev_board_2p = None
        self._prev_next_pair_1p = None
        self._prev_next_pair_2p = None
        self._action_seen_1p = False
        self._action_seen_2p = False

    # ------------------------------------------------------------------
    # CrossValidator API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        pipeline_result: Any,
        frame_bgr: np.ndarray | None,
    ) -> None:
        """1 frame の更新."""
        if not getattr(pipeline_result, "is_match_active", False):
            return
        self._process_side(frame_idx, t_sec, "1P", pipeline_result.p1)
        self._process_side(frame_idx, t_sec, "2P", pipeline_result.p2)

    # ------------------------------------------------------------------
    # 1 サイド分の処理
    # ------------------------------------------------------------------

    def _process_side(
        self,
        frame_idx: int,
        t_sec: float,
        side: str,
        side_result: Any,
    ) -> None:
        """1 サイド分の状態更新 + reveal 検査."""
        state = getattr(side_result, "state", None)
        if state is None:
            return
        confirmed = getattr(side_result, "confirmed_board", None)
        next_pair = self._get_next_pair(side_result)
        # アクション系 state を踏破したか記録
        self._update_action_flag(side, state)
        # 最初に reveal 判定 (前 STABLE → 今 STABLE 遷移時のみ実行)
        if state == BoardState.STABLE and confirmed is not None:
            self._maybe_run_reveal_check(
                frame_idx, t_sec, side, confirmed,
            )
            # 続いて新しい推論を pending に追加
            self._maybe_add_inference(
                frame_idx, t_sec, side, confirmed, next_pair,
            )
        # state を更新 (最後)
        self._save_prev_state(side, state, confirmed, next_pair)

    def _update_action_flag(self, side: str, state: BoardState) -> None:
        """state がアクション系なら action_seen フラグを True に."""
        if state in (
            BoardState.CHAIN,
            BoardState.TSUMO_FALL,
            BoardState.OJAMA_FALL,
        ):
            if side == "1P":
                self._action_seen_1p = True
            else:
                self._action_seen_2p = True

    def _maybe_add_inference(
        self,
        frame_idx: int,
        t_sec: float,
        side: str,
        confirmed: Board,
        next_pair: tuple[int, int] | None,
    ) -> None:
        """STABLE 検出時に隠し段推論を tracker に追加."""
        prev_board = (
            self._prev_board_1p if side == "1P" else self._prev_board_2p
        )
        prev_next_pair = (
            self._prev_next_pair_1p if side == "1P"
            else self._prev_next_pair_2p
        )
        if prev_board is None:
            return
        # silent skip: 推論失敗時は擬似ラベル収集を諦め、運用継続
        try:
            pboard, _result = infer_hidden_row(
                prev_board, confirmed, prev_next_pair,
                apply_calibration=False,
            )
        except Exception:
            return
        snapshot: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "prev_next_pair": (
                list(prev_next_pair) if prev_next_pair is not None else None
            ),
        }
        tracker = (
            self._tracker_1p if side == "1P" else self._tracker_2p
        )
        tracker.add_inference(t_sec, side, pboard, snapshot)
        # サイズ上限を超えたら古いものを切り捨て
        if len(tracker.pending) > MAX_PENDING_PER_SIDE:
            tracker.pending = tracker.pending[-MAX_PENDING_PER_SIDE:]

    def _maybe_run_reveal_check(
        self,
        frame_idx: int,
        t_sec: float,
        side: str,
        current_board: Board,
    ) -> None:
        """前 STABLE → アクション → 今 STABLE の遷移時に reveal 判定.

        改良: lenient モードでは action_seen=False でも、prev → cur で
        row 1 の色変化があれば reveal 候補として走査する。
        """
        prev_state = (
            self._prev_state_1p if side == "1P" else self._prev_state_2p
        )
        prev_board = (
            self._prev_board_1p if side == "1P" else self._prev_board_2p
        )
        action_seen = (
            self._action_seen_1p if side == "1P" else self._action_seen_2p
        )
        if prev_state is None or prev_board is None:
            return
        # 厳格モード: action_seen 経由のみ
        # lenient モード: row 1 色変化があれば action_seen 不問
        if not action_seen and not self._enable_lenient_reveal:
            return
        if (
            not action_seen
            and not _row1_color_changed(prev_board, current_board)
        ):
            return
        tracker = (
            self._tracker_1p if side == "1P" else self._tracker_2p
        )
        # action_seen=False の場合は lenient confidence で emit するため、
        # tracker.update の結果を後で marshal し直す
        events = tracker.update(
            t_sec, side, current_board, prev_board,
            was_chain_or_drop=True,
        )
        is_lenient_path = not action_seen
        for ev in events:
            self._emit_reveal(frame_idx, ev, lenient=is_lenient_path)
        # アクションフラグをリセット (この遷移は消費した)
        if side == "1P":
            self._action_seen_1p = False
        else:
            self._action_seen_2p = False

    def _emit_reveal(
        self,
        frame_idx: int,
        event: RevealEvent,
        lenient: bool = False,
    ) -> None:
        """RevealEvent を PseudoLabelSample に変換して emit.

        lenient=True の場合は action_seen 経由でない reveal なので、
        confidence を抑え、source を区別する。
        """
        predicted_color = _argmax_color(event.predicted_dist)
        if lenient:
            # match/mismatch のいずれも lenient_confidence に丸める
            conf = LENIENT_REVEAL_CONFIDENCE
            source = "reveal_track_lenient"
        else:
            conf = MATCH_CONFIDENCE if event.match else MISMATCH_CONFIDENCE
            source = "reveal_track"
        sample = PseudoLabelSample(
            component=COMPONENT_HIDDEN_ROW,
            timestamp=float(event.timestamp),
            input_data={
                "predicted_dist": {
                    int(k): float(v) for k, v in event.predicted_dist.items()
                },
                "side": event.side,
                "col": int(event.col),
                "predicted_color": int(predicted_color),
            },
            label=int(event.observed_color),
            confidence=conf,
            metadata={
                "frame_idx": int(frame_idx),
                "original_inference_t": float(event.original_inference_t),
                "match": bool(event.match),
                "source": source,
                "reveal_target_row": int(HIDDEN_ROWS),
            },
        )
        self._emit(sample)

    # ------------------------------------------------------------------
    # state 保存
    # ------------------------------------------------------------------

    def _save_prev_state(
        self,
        side: str,
        state: BoardState,
        confirmed: Board | None,
        next_pair: tuple[int, int] | None,
    ) -> None:
        """直前状態を保存 (board は STABLE 時のみ更新)."""
        if side == "1P":
            self._prev_state_1p = state
            if state == BoardState.STABLE and confirmed is not None:
                self._prev_board_1p = confirmed.copy()
                self._prev_next_pair_1p = next_pair
        else:
            self._prev_state_2p = state
            if state == BoardState.STABLE and confirmed is not None:
                self._prev_board_2p = confirmed.copy()
                self._prev_next_pair_2p = next_pair

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_next_pair(side_result: Any) -> tuple[int, int] | None:
        """SideResult から next_pair を安全に取り出す.

        SideResult には next_pair 属性が無い場合があるため、
        chain_event や signals 等の派生属性も試す。
        """
        # 直接属性
        np_pair = getattr(side_result, "next_pair", None)
        if isinstance(np_pair, tuple) and len(np_pair) == 2:
            return (int(np_pair[0]), int(np_pair[1]))
        # signals 経由
        signals = getattr(side_result, "signals", None)
        if signals is not None:
            sp = getattr(signals, "next_pair", None)
            if isinstance(sp, tuple) and len(sp) == 2:
                return (int(sp[0]), int(sp[1]))
        return None


# ============================
# helper
# ============================


def _argmax_color(dist: dict[int, float]) -> int:
    """確率分布から最尤色を返す. 空 dict の場合は 0."""
    if not dist:
        return 0
    return max(dist.items(), key=lambda kv: kv[1])[0]


def _row1_color_changed(prev: Board, cur: Board) -> bool:
    """row 1 (HIDDEN_ROWS) のいずれかの列で色が変化したか.

    lenient reveal モードで使用。EMPTY/UNKNOWN/OJAMA は無視し、
    色付き出現または色付き → 別色付きの変化を「reveal 候補あり」とみなす。
    """
    from src.board import (
        BOARD_COLS as _BOARD_COLS,
        COLOR_EMPTY as _EMPTY,
        COLOR_OJAMA as _OJAMA,
        COLOR_UNKNOWN as _UNK,
    )
    for col in range(_BOARD_COLS):
        try:
            pv = int(prev.get(HIDDEN_ROWS, col))
            cv = int(cur.get(HIDDEN_ROWS, col))
        except (IndexError, ValueError):
            continue
        if cv in (_EMPTY, _UNK, _OJAMA):
            continue
        if pv == cv:
            continue
        return True
    return False


__all__ = [
    "COMPONENT_HIDDEN_ROW",
    "HiddenRowValidator",
    "LENIENT_REVEAL_CONFIDENCE",
    "MATCH_CONFIDENCE",
    "MAX_PENDING_PER_SIDE",
    "MISMATCH_CONFIDENCE",
]
