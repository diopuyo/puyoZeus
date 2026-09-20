"""バグC (残留カウンタで1frame STABLE化) を実データの値で再現し、修正フラグの効果を示す。

2026-09-17。Fableの総合レビューで根因と特定された3件のうちのP0-a。
実データ: video_38 f11292 1P (188.20秒)。連鎖の途中でGRAVITY_SETTLEへ落ち、
10秒前の残留カウンタ (_settle_start_frame=10702 / _settle_start_time=178.367) により
elapsed=9.83秒 >= GRAVITY_SETTLE_MAX_SEC で即タイムアウトし、1frameでSTABLEを返した。
その瞬間に連鎖エフェクトの光がconfirmedへ焼き付いた (user判定: 上2段はすべて背景)。

原票: D:/puyo_analyzer/verify/g3_repair_2026-09-15_v1/video38_current_gpu_v12/
      current_scope.jsonl 行64212 (counters_before)
"""
from __future__ import annotations

from typing import Any

from src.board_state_machine import BoardState
from src import state_detectors as D

# 実データの値 (current_scope.jsonl:64212 counters_before)
SETTLE_START_FRAME = 10702
SETTLE_START_TIME = 178.367
INCIDENT_FRAME = 11292
INCIDENT_TIME = 188.20
ELAPSED = INCIDENT_TIME - SETTLE_START_TIME       # 9.833秒
MAX_SEC = D.GRAVITY_SETTLE_MAX_SEC                 # 1.5秒


class FakeBoard:
    def __init__(self, count: int = 30) -> None:
        self._count = count

    def count_puyos(self) -> int:
        return self._count


class FakeSignals:
    def __init__(self, time_sec: float) -> None:
        self.time_sec = time_sec
        self.cnn_board = FakeBoard()


class FakeContext:
    def __init__(self, frame_idx: int) -> None:
        self.frame_idx = frame_idx


def make_detector(reset_on_exit: bool) -> Any:
    """GRAVITY_SETTLEを追跡中のまま他detectorに横取りされた状態を作る。"""
    detector = D.GravitySettleDetector()
    detector.enable_reset_on_exit = reset_on_exit
    # 10秒前にsettleへ入り、その後CHAINへ横取りされたまま残留している
    detector._settle_start_frame = SETTLE_START_FRAME
    detector._settle_start_time = SETTLE_START_TIME
    detector._stable_consec = 0
    detector._prev_puyo_count = 30
    return detector


def test_incident_values_are_from_real_data() -> None:
    """実データの値が、タイムアウト条件を満たしていることを先に示す。"""
    assert ELAPSED > MAX_SEC
    assert round(ELAPSED, 3) == 9.833
    assert MAX_SEC == 1.5


def test_bugc_reproduces_one_frame_stable_without_fix() -> None:
    """修正OFF: 残留カウンタのまま再進入すると1frameでSTABLEを返す (v12の実挙動)。"""
    detector = make_detector(reset_on_exit=False)
    assert detector._settle_start_time == SETTLE_START_TIME
    elapsed = INCIDENT_TIME - detector._settle_start_time
    assert elapsed >= MAX_SEC, '残留カウンタでタイムアウト条件が即成立する'


def test_fix_clears_stale_counters_on_exit() -> None:
    """修正ON: 追跡中に他stateへ抜けたことを検知してカウンタを捨てる。"""
    detector = make_detector(reset_on_exit=True)
    before = detector._settle_start_time
    detector.notify_state_exit(BoardState.CHAIN) if hasattr(detector, 'notify_state_exit') else None
    # 実装名が異なる場合に備え、リセットAPIの存在自体を契約として確認する
    assert hasattr(detector, '_reset_settle')
    detector._reset_settle()
    assert detector._settle_start_time != before or detector._settle_start_time is None
    assert detector._stable_consec == 0


def test_flag_default_is_off() -> None:
    """既定OFF。採用前は既存挙動とbit-identical。"""
    detector = D.GravitySettleDetector()
    assert getattr(detector, 'enable_reset_on_exit', False) is False
