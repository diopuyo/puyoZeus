"""元world/PBを再用。保存連結の改変対照であり実factory認証ではない。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import pytest
import empty_world as W
import world_inputs as I


@pytest.fixture(scope='module')
def saved() -> Any:
    return I.inputs()


def test_saved_four_worlds_and_original_probabilities(saved: Any) -> None:
    result=W.verify_world(**saved)
    assert result['prepared_worlds']==4 and result['empty_tail_geometry_verified']
    assert result['world_PB_verified'] and not result['runtime_finalization_allowed']
    assert not result['proof_kinds_renamed']


@pytest.mark.parametrize('case',['pair','tail_digest','tail_frame','prefix','raw_capture','hidden_source','current_source','stale_old_source'])
def test_saved_world_mismatch_rejected(saved: Any, case: str) -> None:
    value=deepcopy(saved)
    row=next(r for r in value['history_rows'] if r['prepared'] and r['prepared']['kind']==W.KIND)
    proof=row['prepared']
    if case=='pair': proof['pair']=[1,1]
    elif case=='tail_digest': proof['previous_private_placement']='foreign'
    elif case=='tail_frame': proof['previous_private_frame']-=2
    elif case=='prefix': proof['prefix_source']['run_id']='foreign'
    elif case=='raw_capture': proof['raw_capture']['captured_frame']-=2
    elif case=='hidden_source':
        next(r for r in value['hidden_history'] if r['kind']==W.KIND)['source']['token']='foreign'
    elif case in ('current_source','stale_old_source'):
        import json
        row=next(r for r in value['hidden_events'] if 'current' in r)
        source=json.loads(row['current']['evidence_json'])
        if case=='current_source': source[W.SOURCE_KEY]['token']='foreign'
        else: source['private_suffix_committed_source']=deepcopy(source[W.SOURCE_KEY])
        row['current']['evidence_json']=json.dumps(source)
    with pytest.raises((AssertionError,ValueError,KeyError)):
        W.verify_world(**value)
