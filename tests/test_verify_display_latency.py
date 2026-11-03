"""仮想時計の比較が実際の送信待機を通り、未送信判定も数えること。"""
from __future__ import annotations

from scripts.verify_display_latency import run_delivery, summary
from tests.phase_j.test_live_publish import row


def event(stamp: float, frame: int, complete: float) -> tuple[float, dict]:
    value = dict(row(), frame=frame, t_sec=float(frame), captured_at=complete,
                 recognized_at=complete, evaluated_at=stamp, completed_at=complete,
                 judgment_id=(0, frame), match_active=True,
                 display_layers=dict(current_p1=.6, predicted_p1=None, displayed_p1=.6,
                    includes_prediction=False, prediction_discard_reason=None))
    return stamp, value


def test_replay_measures_wait_instead_of_assuming_zero() -> None:
    old = [event(10.1, 1, 10.), event(10.2, 2, 10.1)]
    new = [event(10., 1, 10.), event(10.1, 2, 10.1)]
    before, old_rows = run_delivery(old, 10., 11., False)
    after, new_rows = run_delivery(new, 10., 11., True)
    old_summary = summary(before, old, 60.)
    new_summary = summary(after, new, 60.)
    assert [r['raw_probability'] for r in old_rows] == [r['raw_probability'] for r in new_rows]
    assert old_summary['delay_ms']['mean'] > 0.
    assert new_summary['delay_ms']['mean'] == 0.
    assert new_summary['delivered'] == 2
    assert new_summary['duplicate_judgment_sends'] == 0


def test_unavailable_delivery_is_counted_in_denominator() -> None:
    events = [event(10., 1, 10.), event(10., 2, 10.)]
    sent, _ = run_delivery(events, 10., 11., True)
    report = summary(sent, events, 60.)
    assert report['judgments'] == 2
    assert report['delivered'] == 1
