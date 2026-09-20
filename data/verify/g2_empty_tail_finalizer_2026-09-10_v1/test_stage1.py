"""元保存fixture/APIを再用し、新空tail構造と旧結果の不変を検査する。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any
import pytest
import empty_stage1 as C

RUN = C.ROOT.parent/'g2_empty_tail_next_2026-09-10_v1/prefix_cpu_v2'
TAIL, START, PLACED = 34896, 34898, 34902
RESULTS: list[Any] = []


def read(path: Path) -> Any:
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines()]
    return json.loads(path.read_bytes())


@pytest.fixture(scope='module')
def saved() -> Any:
    import empty_world as W
    # 実finishと同じくworldのcold依存を原legalより先に一回だけ解決する。
    with W.loaded() as world:
        with world.libraries().G.libraries():
            pass
    with C.libraries() as lib, lib.F.session() as old:
        fixture = old.F.load('_empty_original_test_fixture', lib.F.OLD/'test_stage1.py')
        normal, conditional = fixture.data('cpu_v2'), fixture.data('prefix_cpu_v2')
        prior = sys.modules.get('compat')
        try:
            sys.modules['compat'] = lib.P
            private_fixture = old.F.load('_empty_private_test_fixture', C.PRIVATE/'test_stage1.py')
        finally:
            if prior is None: sys.modules.pop('compat', None)
            else: sys.modules['compat'] = prior
        private = private_fixture.private_data(old, fixture, normal['legal'])
        load = lib.F.clone(private_fixture.private_data, RUN=RUN)
        cached = sys.modules.pop('_private_suffix_saved_goals', None)
        try:
            empty = load(old, fixture, normal['legal'])
        finally:
            sys.modules.pop('_private_suffix_saved_goals', None)
            if cached is not None: sys.modules['_private_suffix_saved_goals'] = cached
        empty['empty_evidence'] = read(RUN/'EMPTY_TAIL_LIVE_EVIDENCE.json')
    return normal, conditional, private, empty


def changed(value: Any) -> Any:
    keys = ('rows','journal_rows','firing_rows','conditional_rows','empty_evidence')
    return value | {key:deepcopy(value[key]) for key in keys if key in value}


def at(value: Any, frame: int) -> Any:
    return next(row for row in value['rows'] if row['scope']['frame_idx'] == frame)


def step(value: Any, frame: int) -> Any:
    return next(row for row in value['journal_rows']
                if row['kind'] == 'step' and row['side'] == '1P' and row['frame_idx'] == frame)


def event(value: Any, frame: int, stage: str) -> Any:
    return next(row for row in step(value, frame)['events'] if row['stage'] == stage)


def test_saved76(saved: Any) -> None:
    value = saved[3]
    before = json.dumps({k:value[k] for k in ('rows','journal_rows','empty_evidence')}, sort_keys=True)
    result = C.audit_stage1(**value)
    assert len(result['history_consumed']) == len(result['conditional_stage1']['placements']) == 4
    assert result['empty_tail_structure_verified'] and not result['empty_tail_samecall_verified']
    assert result['empty_tail_saved_link']['prefixes'] == 2
    assert result['current_proofs'] == [] and result['issued'] == result['released'] == 0
    for key in ('same_live_controller_verified','runtime_finalization_allowed','world_PB_verified',
                'physical_certified','current_permission','production_permission'):
        assert result[key] is False
    assert len(result['conditional_stage1']['empty_tail']['links']) == 1
    assert before == json.dumps({k:value[k] for k in ('rows','journal_rows','empty_evidence')}, sort_keys=True)
    RESULTS.append(dict(case='saved76_structure', result=result))


@pytest.mark.parametrize('index', [0,1,2])
def test_old_complete_delegate(saved: Any, index: int) -> None:
    with C.libraries() as lib:
        expected = lib.P.audit_stage1(**saved[index])
    assert C.audit_stage1(**saved[index]) == expected
    RESULTS.append(dict(case='old_delegate', index=index, equal=True))


@pytest.mark.parametrize('case', ['missing','extra','unknown','no_start','reused_start','future_head',
    'capture_digest','capture_state','tail_not_empty','tail_two_slots','capture_J','scope','binding',
    'added_missing','added_wrong','enqueue_before','enqueue_after','start_time','start_action',
    'old_counter','old_slot','proof_kind','proof_pair','proof_source','proof_before','proof_basis',
    'proof_frame','authority','observed_flag','prepop','missing_row','early_pop','native_counter',
    'duplicate_pop','broken_generation','bool_generation'])
def test_reject(saved: Any, case: str) -> None:
    value = changed(saved[3]); evidence = value['empty_evidence']
    capture, start = evidence['basis_events']; proof = at(value, PLACED)['prepared']
    enqueue = next(r for r in value['journal_rows'] if r['kind']=='enqueue'
                   and r['frame_idx']==START and r['side']=='1P')
    mutate(value, evidence, capture, start, proof, enqueue, case)
    with pytest.raises((ValueError,KeyError,TypeError,AssertionError)) as error:
        C.audit_stage1(**value)
    RESULTS.append(dict(case=case, rejected=type(error.value).__name__, reason=str(error.value)))


def mutate(value: Any, evidence: Any, capture: Any, start: Any, proof: Any, enqueue: Any, case: str) -> None:
    if case == 'missing': value['empty_evidence'] = None
    elif case == 'extra': evidence['prepop_checks'].append(deepcopy(evidence['prepop_checks'][0]))
    elif case == 'unknown': start['kind'] = 'unknown/v1'
    elif case == 'no_start': evidence['basis_events'].pop()
    elif case == 'reused_start': evidence['basis_events'][1] = deepcopy(capture)
    elif case == 'future_head': capture['head'] = start['head']
    elif case == 'capture_digest': capture['digest'] = '0'*64
    elif case == 'capture_state': capture['state']['action'] += 1
    elif case == 'tail_not_empty': event(value, TAIL, 'fifo_after')['accounting']['pending_tsumo'] = [[1,1]]
    elif case == 'tail_two_slots': event(value, TAIL, 'fifo_before')['fifo_occurrence_tokens'].append('extra')
    elif case == 'capture_J': capture['journal_content'][1][0][1] = 'step:foreign'
    elif case == 'scope': start['scope'][5] += 1
    elif case == 'binding': start['binding_id'] += 1
    elif case == 'added_missing': at(value, START)['added'] = []
    elif case == 'added_wrong': at(value, START)['added'][0] += ':foreign'
    elif case == 'enqueue_before': enqueue['before']['pending_tsumo'] = [[1,1]]
    elif case == 'enqueue_after': enqueue['after']['pending_tsumo'][0] = [1,1]
    elif case == 'start_time': start['clock'] += 1
    elif case == 'start_action': at(value, START)['decision']['history_state']['action'] += 1
    elif case == 'old_counter': start['state']['counter'][0] += 1
    elif case == 'old_slot': at(value, PLACED)['decision']['history_state']['current'] = {}
    elif case == 'proof_kind': proof['kind'] = 'hidden_private_suffix_history/v1'
    elif case == 'proof_pair': proof['pair'] = [1,1]
    elif case == 'proof_source': proof['prefix_source']['old_token'] += ':foreign'
    elif case == 'proof_before': proof['before_grid'][0][0] = 1
    elif case == 'proof_basis': proof['previous_private_placement'] = '0'*64
    elif case == 'proof_frame': proof['previous_private_frame'] -= 2
    elif case == 'authority': proof['current_permission'] = True
    elif case == 'observed_flag': proof['inferred_grid_is_observed'] = True
    elif case == 'prepop': evidence['prepop_checks'][0]['event_id'] = '0'*64
    elif case == 'missing_row': value['rows'].remove(at(value, START))
    elif case == 'early_pop': at(value, PLACED-2)['decision']['history_consumed'] = True
    elif case == 'native_counter': event(value, PLACED, 'fifo_after')['accounting']['tsumo_count'] = {'2':1}
    elif case == 'duplicate_pop': step(value, PLACED)['events'].append(deepcopy(event(value,PLACED,'fifo_after')))
    elif case == 'broken_generation': step(value, START)['generation_after']['action_revision'] += 2
    elif case == 'bool_generation': step(value, START)['generation_after']['action_revision'] = True
    else: raise AssertionError(case)


def test_installed_restores_and_denies_evaluate(saved: Any) -> None:
    with C.libraries() as lib, lib.F.session() as original:
        modules = [(m,dict(vars(m))) for m in (original,original.J,original.S)]
        with C.installed(original, empty_evidence=saved[3]['empty_evidence']) as facade:
            value = {k:v for k,v in saved[3].items() if k!='empty_evidence'}
            assert facade.audit_stage1(**value)['empty_tail_structure_verified']
            with pytest.raises(ValueError, match='empty_world_and_samecall_required'):
                facade.evaluate(**value)
        assert all(set(vars(m)) == set(before) and all(vars(m)[k] is v for k,v in before.items())
                   for m,before in modules)


def test_world_saved_separate(saved: Any) -> None:
    import empty_world as W
    import world_inputs as I
    report = W.verify_world(**I.inputs())
    assert report['world_PB_verified']
    RESULTS.append(dict(case='world_separate', result=report))


def tagged(value: Any) -> Any:
    """改変対照の再符号化だけ。元SP.content同形式で意味検査を迂回しない。"""
    if type(value) is dict: return ['dict',[[k,tagged(v)] for k,v in sorted(value.items())]]
    if type(value) in (tuple,list): return [type(value).__name__,[tagged(v) for v in value]]
    return [type(value).__name__,value]


@pytest.mark.parametrize('case', ['baseline_kind','baseline_scope','old_policy','prefix_support'])
def test_coherent_content_mutation(saved: Any, case: str) -> None:
    value = changed(saved[3]); capture = value['empty_evidence']['basis_events'][0]
    if case == 'prefix_support':
        source = C.J.thaw(capture['source_content'])
        source[1]['final'] = tuple(tuple(1 if (r,c)==(0,0) else v for c,v in enumerate(row))
                                   for r,row in enumerate(source[1]['final']))
        capture['source_content'] = tagged(source)
    else:
        policy = C.J.thaw(capture['policy_content'])
        target = policy[1] if case == 'old_policy' else policy[0]
        if case == 'baseline_kind': target['proof']['kind'] = 'foreign'
        elif case == 'baseline_scope': target['proof']['live_scope'] = tuple([*capture['scope'][:4],1,*capture['scope'][5:]])
        else: target['proof']['inferred_path']['physical_certified'] = True
        from hashlib import sha256
        text = json.dumps(target['proof'],sort_keys=True,separators=(',',':'),allow_nan=False)
        target['proof_sha'] = sha256(text.encode()).hexdigest()
        capture['policy_content'] = tagged(policy)
    with pytest.raises((ValueError,KeyError,AssertionError)) as error:
        C.audit_stage1(**value)
    RESULTS.append(dict(case=case,coherent_content=True,reason=str(error.value)))
