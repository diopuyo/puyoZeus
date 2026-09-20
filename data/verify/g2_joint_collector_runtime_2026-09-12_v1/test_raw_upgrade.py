"""原cold collector生成・元raw code認証・観測器復元を同時検査する。"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import pytest
import raw_upgrade as R

sys.path.insert(0, str(R.ROOT.parent / 'g2_empty_tail_desync_trigger_2026-09-11_v1'))
from test_collector_continuous import real


def test_original_generator_and_upgraded_observer(real: Any) -> None:
    collector, bounded = real
    original = collector.EventAccountingRecorder
    loop = R.build(collector, bounded.O.SOURCE)
    try:
        R.B.R.validate(loop)
        assert collector.EventAccountingRecorder is original
        assert type(loop.runtime_state['accounting_recorder']) is R.U.U.modern()
        assert loop.runtime_state['accounting_recorder']._observed_frame_count == 0
        assert loop.configuration['enable_event_accounting_sidecar'] is True
        assert loop.source_collector is collector
    finally:
        loop.generator.close()
    with pytest.raises(ValueError, match='raw_generator_closed'):
        R.B.R.validate(loop)


def test_accounting_cannot_be_disabled(real: Any) -> None:
    collector, bounded = real
    with pytest.raises(ValueError, match='joint_accounting_required'):
        R.build(collector, bounded.O.SOURCE, dict(enable_event_accounting_sidecar=False))
