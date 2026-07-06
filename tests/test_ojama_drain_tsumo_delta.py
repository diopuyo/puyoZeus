"""tsumo_count 増分駆動 drain (_drain_by_tsumo_delta) の回帰テスト。

P1 修正: ツモ着地が TSUMO_FALL 状態を通らず STABLE→STABLE で記録される
ケースを取りこぼす under-drain を、tsumo_count 増分駆動で解消したことを保証する。
"""
from __future__ import annotations

from scripts.collect_indicators_v2 import _SideTracker, _drain_by_tsumo_delta
from src.ojama_accounting import OjamaAccountingTracker


class _FakePipeline:
    """tsumo_count(side) を固定値で返すスタブ。"""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def tsumo_count(self, side: str) -> int:
        return self._counts[side]


def test_drain_fires_delta_times_and_reduces_forecast() -> None:
    """tsumo_count が +Δ したら on_tsumo_settled が Δ 回呼ばれ forecast が減る。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    # 1P に予告 20 個を仮に積む
    tracker._p1.forecast_incoming = 20
    side_tracker = _SideTracker()  # prev_tsumo=0
    pipe = _FakePipeline({"1P": 3})  # 3 着地分 (delta=3)

    _drain_by_tsumo_delta(tracker, pipe, side_tracker, "p1", "1P", t_sec=1.0)

    # 1 着地で min(30, 20)=20 drain → 0。残り 2 着地は 0 drain。
    assert tracker._p1.forecast_incoming == 0
    assert side_tracker.prev_tsumo == 3


def test_no_drain_when_tsumo_unchanged() -> None:
    """tsumo_count が変わらなければ drain されない (連鎖中=着地なし)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker._p2.forecast_incoming = 26
    side_tracker = _SideTracker()
    side_tracker.prev_tsumo = 40
    pipe = _FakePipeline({"2P": 40})  # delta=0

    _drain_by_tsumo_delta(tracker, pipe, side_tracker, "p2", "2P", t_sec=1.0)

    # 着地が無いので forecast は据え置き (正常挙動)
    assert tracker._p2.forecast_incoming == 26


def test_match_boundary_negative_delta_skipped() -> None:
    """試合境界で tsumo_count がリセット (delta<0) されても drain しない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker._p1.forecast_incoming = 5
    side_tracker = _SideTracker()
    side_tracker.prev_tsumo = 36
    pipe = _FakePipeline({"1P": 0})  # delta=-36

    _drain_by_tsumo_delta(tracker, pipe, side_tracker, "p1", "1P", t_sec=1.0)

    assert tracker._p1.forecast_incoming == 5  # 変化なし
    assert side_tracker.prev_tsumo == 0  # prev は追従更新
