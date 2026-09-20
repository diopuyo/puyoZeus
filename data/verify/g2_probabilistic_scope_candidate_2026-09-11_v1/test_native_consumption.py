"""保存済み原Jの消費票で抽出を検査。保存票読取りを実caller認証とは呼ばない。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import pytest
import native_consumption as N

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v23/atomic_journal.jsonl'


def saved() -> dict:
    with SOURCE.open(encoding='utf-8') as stream:
        for line in stream:
            if 'fifo_after' in line:
                return json.loads(line)
    raise AssertionError('保存済み原FIFO消費が必要')


def test_original_native_same_color_consumption() -> None:
    item = saved()
    old = deepcopy(item)
    event = N.extract(item)
    assert event.pair == (5, 5) and item == old
    assert not event.physical_landing_certified and not event.accounting_permission
    assert event.occurrence_token.endswith(':enqueue:28:slot:0')


@pytest.mark.parametrize('case', ('missing_before', 'missing_after', 'duplicate_after',
    'wrong_committed', 'wrong_tail', 'missing_token', 'wrong_token', 'wrong_tokens_length'))
def test_invalid_native_pair_refused(case: str) -> None:
    item = saved()
    start = next(e for e in item['events'] if e['stage'] == 'fifo_before')
    end = next(e for e in item['events'] if e['stage'] == 'fifo_after')
    if case == 'missing_before': item['events'].remove(start)
    elif case == 'missing_after': item['events'].remove(end)
    elif case == 'duplicate_after': item['events'].append(deepcopy(end))
    elif case == 'wrong_committed': end['committed'] = [4, 5]
    elif case == 'wrong_tail': end['accounting']['pending_tsumo'] = []
    elif case == 'missing_token': end['enqueue_occurrence_token'] = None
    elif case == 'wrong_token': end['enqueue_occurrence_token'] = 'invented'
    else: start['fifo_occurrence_tokens'] = []
    with pytest.raises(ValueError): N.extract(item)


def test_no_fifo_event_is_not_a_consumption() -> None:
    assert N.extract(dict(token='no-event', events=[])) is None
