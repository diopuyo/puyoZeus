"""元v67保存票の整数公開境界を検査。確率分布・実liveの合格ではない。"""
from __future__ import annotations
import json
from typing import Any
import pytest
import probability_boundary as P
from test_saved_split_timeline import SAVED, rows


@pytest.fixture
def saved() -> dict[str, Any]:
    history = rows('directional_history.jsonl')
    status = json.loads((SAVED / 'PROBABILISTIC_TRACKING_STATUS.json').read_bytes())
    reset = json.loads((SAVED / 'LIVE_EMPTY_RESET.json').read_bytes())
    return dict(consumer=json.loads((SAVED / 'POSTCOMMIT_CONSUMER_ROWS.json').read_bytes()),
                history=history, journal=rows('atomic_journal.jsonl'), recovery=reset['recovery'],
                tracking=rows('PROBABILISTIC_TRACKING.jsonl'),
                history_first=34796, end_frame=34980,  # 親検収済みv67限定の固定範囲。
                proof_frame=max(row['scope']['frame_idx'] for row in history if 'decision' in row),
                basis_frame=status['activation']['frame'], scope=tuple(history[-1]['scope']))


def test_original_saved_boundary(saved: dict[str, Any]) -> None:
    report = P.check(**saved)
    assert report['waiting_updates'] == 7 and report['tracking_updates'] == 18
    assert report['legacy_publication_join_verified'] and not report['quality_gate_clear']


@pytest.mark.parametrize('case', ('missing', 'duplicate', 'baseline', 'scope', 'token', 'publication',
                                 'old_gap', 'journal_exception', 'consumer_changed', 'truncated_end'))
def test_saved_corruption_rejected(saved: dict[str, Any], case: str) -> None:
    if case == 'missing': saved['tracking'].pop(0)
    elif case == 'duplicate': saved['tracking'].append(saved['tracking'][-1])
    elif case == 'baseline': saved['recovery'].append(dict(kind='new_baseline'))
    elif case == 'scope': saved['tracking'][0]['scope']['run_id'] = 'foreign'
    elif case == 'token': saved['tracking'][0]['journal_token'] = 'step:wrong'
    elif case == 'publication': saved['consumer'][-1]['tickets_this_update'] = 1
    elif case == 'old_gap': saved['history'].pop(0)
    elif case == 'truncated_end':
        saved['consumer'].pop()
        saved['tracking'].pop()
    elif case == 'journal_exception':
        step = next(row for row in saved['journal'] if row['kind'] == 'step'
                    and row['side'] == '1P' and row['frame_idx'] == saved['basis_frame'] + P.STRIDE)
        step['exception'] = 'failed'
    elif case == 'consumer_changed': saved['consumer'][-1]['same_result_identity'] = False
    with pytest.raises(AssertionError):
        P.check(**saved)
