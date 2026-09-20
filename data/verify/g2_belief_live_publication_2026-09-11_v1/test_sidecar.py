"""人工scorer結果の全world保存・排他性を検査。実採録合格ではない。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from test_live_binding import saved, live
import sidecar as S

ARTIFICIAL_MODEL_SHA = '0' * 64


def result(live: Any) -> tuple[Any, Any]:
    return S.L.evaluate(live.capture, lambda inputs: lambda batch: np.full(len(batch), 0.6))


def test_saved_worlds(live: Any, tmp_path: Path) -> None:
    bound, score = result(live)
    path = tmp_path/'evaluation.json'
    digest = S.save(path, bound, score, ARTIFICIAL_MODEL_SHA)
    packet = json.loads(path.read_bytes())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert tuple(S.S.decode(v) for v in packet['states']) == bound.values
    assert len(packet['states'][0]['hidden_worlds']) == 49
    assert packet['ledger_connection'] == 'NOT_CONNECTED' and not packet['training_permission']
    with pytest.raises(FileExistsError):
        S.save(path, bound, score, ARTIFICIAL_MODEL_SHA)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_foreign_result(live: Any, tmp_path: Path) -> None:
    bound, score = result(live)
    with pytest.raises(ValueError, match='sidecar_result_scope'):
        S.save(tmp_path/'bad.json', bound, replace(score, frame=score.frame+2), ARTIFICIAL_MODEL_SHA)
    assert not (tmp_path/'bad.json').exists()
