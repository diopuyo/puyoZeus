"""非同期を同期SSEへ誤対応せず、全観測の台帳時間も数える。"""
import json
from pathlib import Path

from scripts.b3_realtime_report import breakdown, placement_latencies


def test_async_does_not_use_earlier_sse() -> None:
    metrics = dict(sse_sent=[dict(frame=20, sent_at=1.)])
    updates = [dict(frame=10, captured_at=0., completed_at=2.)]
    assert placement_latencies(metrics, updates, True) == ([], 0)
    assert placement_latencies(metrics, updates, False) == ([1000.], 1)


def test_all_observation_ledger_is_separate(tmp_path: Path) -> None:
    rows = [dict(frame=i, ledger_ms=2., state_ms=1.) for i in range(4)]
    (tmp_path/'b3_state_timing.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    result = breakdown(tmp_path, [dict(ledger_ms=2., search_ms=4., accepted=False)], 1, 3)
    assert result['observations'] == 2
    assert result['all_observations']['ledger_ms']['total'] == 4.
    assert result['computed']['search_ms']['total'] == 4.
    assert result['accepted'] == 0 and result['rejected'] == 1
