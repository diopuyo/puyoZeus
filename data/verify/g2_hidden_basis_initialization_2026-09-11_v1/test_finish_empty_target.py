"""空の対象票をsubsetと見なす旧穴を実保存で再現する。"""
from __future__ import annotations
from typing import Any
import test_probabilistic_finish as T
import probabilistic_finish_v1 as OLD


def test_old_empty_target_passed(monkeypatch: Any) -> None:
    read = OLD.read
    monkeypatch.setattr(OLD, 'read', lambda output, name:
        {} if name == 'PROBABILISTIC_TARGET_RESULT.json' else read(output, name))
    _, report = OLD.saved(T.OUTPUT, T.S)
    assert report['saved_evidence_verified']
