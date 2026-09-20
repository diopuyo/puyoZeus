"""原bindingとの同値対照と原frozen環境でのモデル非importを確認する。"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from typing import Any
import pytest
import pure_contract as P

PROJECT=P.ROOT.parents[2]
SAVED=P.ROOT.parent/'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v46/provisional_context.jsonl'


def load(name: str,path: Path) -> Any:
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def original() -> Any:
    return load('_pure_contract_original_reference',P.SOURCE)


@pytest.fixture
def row() -> Any:
    with SAVED.open() as stream:
        for line in stream:
            value=json.loads(line)
            if value['frame_idx']==35370: return value
    raise AssertionError('saved_reference_missing')


def result(api: Any,row: Any) -> Any:
    registration=dict(source_id=row['source_id'],run_id=row['run_id'],
        time_base_numerator=1,time_base_denominator=60,ledger_connection='NOT_CONNECTED')
    try:
        api._identity(row,registration)
        api._update(row)
        api._generation(row)
    except Exception as error: return type(error).__name__,str(error)
    return 'PASS',api.digest(row)


@pytest.mark.parametrize('case',range(8))
def test_exact_original_result(original: Any,row: Any,case: int) -> None:
    if case==1: row['frame_idx']=True
    if case==2: row['available_frame']-=2
    if case==3: row['update']['returned']=False
    if case==4: row['update']['returned_frame_idx']+=2
    if case==5: row['update']['match_end_locked']=True
    if case==6: row['generation']['after']['value']['2P']['reset_epoch']+=1
    if case==7: row['time_sec']=float('nan')
    actual=result(P,row)
    assert actual==result(original,row)
    assert (actual[0]=='PASS')==(case==0)
    for name in ('_identity','_update','_generation'):
        assert getattr(P,name).__code__.co_code==getattr(original,name).__code__.co_code


def test_carrier_is_unconverted_snapshot(row: Any) -> None:
    saved=deepcopy(row)
    carrier=P._inputs(row)
    row['frame_idx']+=2
    assert json.loads(carrier.source_json)==saved
    assert not hasattr(carrier,'boards')


def frozen_check(row: Any) -> None:
    fixture=load('_pure_contract_frozen_fixture',
        PROJECT/'tests/test_diagnose_video38_next_enqueue_live_shadow_v1.py')
    generator=fixture.frozen.__wrapped__()
    collector=next(generator)
    old={k:v for k,v in sys.modules.items() if k=='src' or k.startswith('src.')}
    update=collector.RecognitionPipeline.update
    try:
        api=load('_pure_contract_cold_reference',P.ROOT/'pure_contract.py')
        assert result(api,row)[0]=='PASS'
        assert json.loads(api._inputs(row).source_json)==row
        now={k:v for k,v in sys.modules.items() if k=='src' or k.startswith('src.')}
        assert old.keys()==now.keys() and all(now[k] is v for k,v in old.items())
        assert collector.RecognitionPipeline.update is update
    finally: generator.close()


def test_actual_frozen_import_without_models() -> None:
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--frozen'],
        cwd=PROJECT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr


if __name__=='__main__':
    assert sys.argv[1:]==['--frozen']
    frozen_check(row.__wrapped__())
