"""表示用時刻の射影だけを作り、使わない全履歴の深いコピーを避ける。"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

EVENT_FIELDS = ('trigger_sec', 'closed_sec')
CHAIN_FIELDS = ('observed_sec', 'end_signal_sec')
LANDING_FIELDS = ('fall_start_sec', 't_sec')


def selected(value: Any, names: tuple[str, ...]) -> dict:
    """dictの欠測とdataclassの値を、元のgetと同じNoneへ揃える。"""
    if isinstance(value, dict):
        return {name: value.get(name) for name in names}
    return {name: getattr(value, name) for name in names}


def physical_mapping(record: Any) -> dict:
    # 元実装でもdictはasdictを通さない。順序とcurrentの重複も変更しない。
    if isinstance(record, dict):
        return record
    return dict(selected(record, EVENT_FIELDS),
        chains=[selected(chain, CHAIN_FIELDS) for chain in record.chains],
        landings=[selected(landing, LANDING_FIELDS) for landing in record.landings])


def reference_state(tracker: Any) -> dict:
    """unique・型変換・表示側の合図数の計算は元の関数のまま維持する。"""
    from src.b3_stage2_trace import physical_times
    records = list(getattr(tracker, 'records', ()))
    if tracker.current is not None:
        records.append(tracker.current)
    mappings = [physical_mapping(record) for record in records]
    return dict(source=tracker.source, probability=tracker.probability,
                physical=physical_times(mappings).tolist())


@contextmanager
def fast_reference() -> Iterator[None]:
    from src.phase_j import live_b3
    with patch.object(live_b3, 'reference_state', reference_state):
        yield
