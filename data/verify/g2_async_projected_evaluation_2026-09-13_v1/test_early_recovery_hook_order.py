"""原Recoveryの後付けを早期Oの寿命へ結合し、中段解除の旧反例を再現する。"""
from contextlib import ExitStack
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import second_observation as O
import early_probability_capture as E
from test_early_probability_capture import prepare, replace
from test_second_observation import context, generated, live, saved

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('early_real_recovery',
    ROOT / 'g2_reset_recovery_candidate_2026-09-10_v1/recovery_connection.py')
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def test_original_recovery_after_early_observer_blocks_middle_close(context: Any) -> None:
    value = context
    prepare(value)
    original = value.journal.complete_step
    with ExitStack() as outer:
        early = ExitStack()
        capture = E.Capture(early, value.journal, value.state, O, replace)
        evidence = capture.evidence
        sink = N(before=lambda *args: None, after=lambda *args: None, stream=io.StringIO(), rows=0)
        recovery = N(journal=value.journal, failure=None, complete=lambda *args: None)
        RECOVERY.recording(outer, sink, recovery)
        generated(value.journal, value.pipe, value.result, value.row)
        with pytest.raises(ValueError, match='second_foreign_hook'):
            early.close()
        assert evidence.closed and evidence.error is not None and capture.closed
    # 中段close失敗で元Oが残る事実も保持。outerの復元だけでは正常に戻らない。
    assert value.journal.complete_step != original
