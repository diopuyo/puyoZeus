"""監査器が先読み・欠測・同一撃ち合い比較を検出することを確かめる。"""
import numpy as np

from scripts.prefire_v5_report import audit, agreement


def table() -> dict[str, np.ndarray]:
    """0.5秒前と直前の2行。"""
    return {key: np.array(values) for key, values in dict(
        t_sec=[1.5, 1.9], game_idx=[1, 1], p_current=[0.5, 0.5], p_shown=[0.7, 0.3],
        v_1p=[0.7, 0.3], v_2p=[0.7, 0.3], ready_sec=[1.0, 1.7], used=[1, 1], held=[0, 0],
        status_1p=[0, 0], status_2p=[0, 0]).items()}


def test_audit_detects_reference_even_without_display() -> None:
    rows = table()
    rows['ready_sec'][0] = 2.0
    rows['used'][0] = 0
    result = audit(rows)
    assert result['early_referenced'] == 1
    assert result['early_used'] == 0


def test_paired_same_exchange() -> None:
    events = [dict(game=1, trigger=2.0, prev_close=0.0, target=0.9)]
    assert agreement(table(), events) == dict(denominator=1, last_hits=0, half_second_hits=1,
        exchanges=1, with_both_endpoints=1, last_scored=1, half_second_scored=1, both_scored=1)


def test_missing_endpoint_excludes_both() -> None:
    rows = table()
    rows['used'][1] = 0
    events = [dict(game=1, trigger=2.0, prev_close=0.0, target=0.9)]
    assert agreement(rows, events)['denominator'] == 0
