"""v70原保存の旧scope全被覆を新derive本体で検査。Stage1全体合格ではない。"""
from __future__ import annotations
from dataclasses import make_dataclass
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import probability_stage as S

BASE = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'
ADAPTER_SHA = 'f133e45de3c812f3c447464fbd18f6949c3324500617b1862a861003cba55e53'
FIRST = 34796
PROOF = 34930


class ReachedOriginalDerive(Exception):
    """後段を実行していないことを区別する正常対照の停止。"""


@pytest.mark.parametrize('defect', [None, 'first', 'last', 'scope', 'post_reset', 'hash'])
def test_actual_saved_old_coverage(defect: str | None) -> None:
    full = [json.loads(line) for line in (BASE / 'prefix_cpu_v70_lower_finish' /
            'directional_history.jsonl').read_text().splitlines()]
    rows = [r for r in full if 'decision' in r]
    expected = rows[0]['decision']['history_state']['scope']
    scope_type = make_dataclass('SavedScope', [(name, Any) for name in expected])
    old = N(baseline_through=N(frame=FIRST), scope=scope_type(**expected))
    lease = N(archive=N(binding=N(owner=N(state=old))), empty_evidence=N(frame=PROOF))
    original = N(STAGE=N(ADAPTER=BASE / 'multiscope_stage/adapt.py', ADAPTER_SHA=ADAPTER_SHA),
                 Q=N(E=N(STRIDE=2), module=lambda *args: N()))
    if defect == 'first':
        rows.pop(0)
    elif defect == 'last':
        rows.pop()
    elif defect == 'scope':
        rows[-1]['decision']['history_state']['scope']['reset_epoch'] += 1
    elif defect == 'post_reset':
        rows[-1]['scope']['frame_idx'] += 2
    elif defect == 'hash':
        original.STAGE.ADAPTER_SHA = '0' * 64
    def derive(*args: Any) -> None:
        raise ReachedOriginalDerive()
    error = ReachedOriginalDerive if defect is None else AssertionError
    with pytest.raises(error):
        S.derive(original, N(derive=derive), None, None, None, rows, lease)
