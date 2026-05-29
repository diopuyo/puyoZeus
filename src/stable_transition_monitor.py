"""StableTransitionMonitor: STABLE 遷移間の物理事由なき大幅ぷよ減少を検知。

Phase 1 / Task C2 (2026-05-28)。

PuyoErasureMonitor が blind な「初回から空」 型 fail-silent を補完する。
STABLE 終了時に board を記録し、 NON-STABLE 中の連鎖 / ojama イベントなしで
次の STABLE 復帰時にぷよ数が大幅減少していたら Alert を生成する。

設計:
    - stateless 原則: state は外部から渡す (reset() で一掃)
    - backwards compat: SideResult.transition_drop_alerts は optional フィールド
    - 1 関数 50 行以内
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_UNKNOWN,
    Board,
)

# ============================
# 定数定義 (= マジックナンバー禁止)
# ============================

# 1 ツモ = 2 cell なので 2 以内の減少は正常範囲。 3 以上で異常判定。
STABLE_TRANSITION_DROP_THRESHOLD: int = 2

# NON-STABLE 中のイベントを参照するウィンドウ (秒)。
# 直前 3 秒以内に連鎖 / ojama イベントがあれば減少を許容する。
NON_STABLE_EVENT_WINDOW_SEC: float = 3.0

# イベント種別 (= on_non_stable_event の event_type として受け入れる文字列)
EVENT_CHAIN_START: str = "chain_start"
EVENT_OJAMA_LAND: str = "ojama_land"
# その他の物理イベント (= 連鎖終了、 落下等) も許容として受け入れる prefix
EVENT_CHAIN_PREFIX: str = "chain"
EVENT_OJAMA_PREFIX: str = "ojama"


# ============================
# Alert データクラス
# ============================

@dataclass
class TransitionDropAlert:
    """STABLE → STABLE 間で物理事由なきぷよ減少 alert。"""
    frame_idx: int              # 復帰 STABLE の frame index
    t_sec: float                # 復帰 STABLE の時刻
    prev_count: int             # 前回 STABLE のぷよ数
    curr_count: int             # 今回 STABLE のぷよ数
    drop: int                   # 減少数 (prev - curr)
    prev_frame_idx: int         # 前回 STABLE 終了時の frame index

    def to_tuple(self) -> tuple:
        """scripts / tests から参照しやすいシンプルな tuple 表現。"""
        return (self.frame_idx, self.t_sec, self.prev_count, self.curr_count, self.drop)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t_sec": round(self.t_sec, 2),
            "prev_count": self.prev_count,
            "curr_count": self.curr_count,
            "drop": self.drop,
            "prev_frame_idx": self.prev_frame_idx,
        }


# ============================
# モニター本体
# ============================

class StableTransitionMonitor:
    """STABLE 終了〜次の STABLE 開始間の物理事由なきぷよ大幅減少を検知。

    Usage:
        monitor = StableTransitionMonitor()
        # STABLE 終了時:
        monitor.on_stable_end(frame_idx, board)
        # NON-STABLE 中のイベント:
        monitor.on_non_stable_event("chain_start", frame_idx)
        # 次の STABLE 開始時:
        alerts = monitor.on_stable_start(frame_idx, t_sec, board)
    """

    def __init__(self) -> None:
        # 前回 STABLE 終了時の board スナップショット (None = 未記録)
        self._last_stable_board: Board | None = None
        self._last_stable_frame_idx: int = -1
        # NON-STABLE 中に観測した物理イベントのリスト [(t_sec, event_type)]
        self._non_stable_events: list[tuple[float, str]] = []
        # 累積 alert リスト (to_dict() で参照可能)
        self._alerts: list[TransitionDropAlert] = []

    def on_stable_end(self, frame_idx: int, board: Board) -> None:
        """STABLE 終了時に board をスナップショット記録する。"""
        self._last_stable_board = board
        self._last_stable_frame_idx = frame_idx
        # 新しい NON-STABLE 区間のイベントバッファをクリア
        self._non_stable_events = []

    def on_non_stable_event(self, event_type: str, frame_idx: int, t_sec: float = 0.0) -> None:
        """NON-STABLE 中の物理イベント (連鎖 / ojama 等) を記録する。"""
        self._non_stable_events.append((t_sec, event_type))

    def on_stable_start(
        self,
        frame_idx: int,
        t_sec: float,
        board: Board,
    ) -> list[TransitionDropAlert]:
        """STABLE 復帰時にぷよ数を比較し、 異常なら Alert リストを返す。

        Args:
            frame_idx: 復帰 STABLE の frame index。
            t_sec: 復帰 STABLE の時刻 (秒)。
            board: 復帰 STABLE の確定盤面。

        Returns:
            Alert リスト (= 通常は空)。
        """
        if self._last_stable_board is None:
            # 初回 STABLE = 比較対象なし → alert なし
            return []
        prev_count = _count_puyo(self._last_stable_board)
        curr_count = _count_puyo(board)
        drop = prev_count - curr_count
        alerts: list[TransitionDropAlert] = []
        if drop > STABLE_TRANSITION_DROP_THRESHOLD:
            # 直前 NON-STABLE 区間に物理イベントがあれば許容
            if not _has_physics_event(self._non_stable_events, t_sec):
                alert = TransitionDropAlert(
                    frame_idx=frame_idx,
                    t_sec=t_sec,
                    prev_count=prev_count,
                    curr_count=curr_count,
                    drop=drop,
                    prev_frame_idx=self._last_stable_frame_idx,
                )
                alerts.append(alert)
                self._alerts.append(alert)
        return alerts

    def reset(self) -> None:
        """試合リセット時に全 state を消去する。"""
        self._last_stable_board = None
        self._last_stable_frame_idx = -1
        self._non_stable_events = []
        self._alerts = []

    def get_all_alerts(self) -> list[TransitionDropAlert]:
        """蓄積済みの全 alert を返す。"""
        return list(self._alerts)

    def to_dict(self) -> dict[str, Any]:
        """デバッグ / シリアライズ用 dict 表現。"""
        return {
            "alert_count": len(self._alerts),
            "alerts": [a.to_dict() for a in self._alerts],
            "has_last_stable_board": self._last_stable_board is not None,
            "last_stable_frame_idx": self._last_stable_frame_idx,
            "pending_events": len(self._non_stable_events),
        }


# ============================
# ユーティリティ (= モジュールプライベート)
# ============================

def _count_puyo(board: Board) -> int:
    """Board の非 EMPTY・非 UNKNOWN cell 数を返す。"""
    count = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            val = int(board.get(r, c))
            if val not in (COLOR_EMPTY, COLOR_UNKNOWN):
                count += 1
    return count


def _has_physics_event(
    events: list[tuple[float, str]],
    stable_start_t: float,
) -> bool:
    """NON-STABLE 区間に物理イベント (連鎖 / ojama) があれば True を返す。

    直前 NON_STABLE_EVENT_WINDOW_SEC 以内のイベントのみ参照する。
    """
    window_start = stable_start_t - NON_STABLE_EVENT_WINDOW_SEC
    for t, ev in events:
        if t < window_start:
            continue
        ev_lower = ev.lower()
        if ev_lower.startswith(EVENT_CHAIN_PREFIX) or ev_lower.startswith(EVENT_OJAMA_PREFIX):
            return True
    return False
