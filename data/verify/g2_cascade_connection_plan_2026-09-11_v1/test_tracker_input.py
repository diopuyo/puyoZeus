"""原fixture/constructor/updateでtracker存在と非発火対照だけを確認する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from typing import Any
import pytest
import tracker_input as T

SOURCE = Path(__file__).resolve().parents[3] / 'tests/test_next_enqueue_live_shadow_v1.py'
spec = importlib.util.spec_from_file_location('_g2_tracker_original_fixture', SOURCE)
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)


def test_original_constructor_tracker_and_empty_updates() -> None:
    evidence: dict[str, Any] = {}
    with contextlib.contextmanager(F.frozen.__wrapped__)() as frozen:
        constructor = frozen.RecognitionPipeline.__init__
        with pytest.MonkeyPatch.context() as patch:
            supplied = T.transport(F.real.__wrapped__, evidence)
            with contextlib.contextmanager(supplied)(frozen, patch) as real:
                pipe, _, _, image, _ = real
                for frame in range(34772, 34788, 2):
                    pipe.update(frame, frame / 60, image)
                    value = T.observe(pipe)
                    assert not value['active_origin_1p'] and not value['active_origin_2p']
        assert frozen.RecognitionPipeline.__init__ is constructor
    assert evidence['tracker_constructor_calls'] == 1 and evidence['tracker_constructor_restored']
