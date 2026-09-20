"""chain-end v2 adapterの製造CPU検証。独立QA・実認識の合格ではない。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from scripts import diagnose_video38_chain_end_shadow_v2 as target


@pytest.fixture
def short_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """人工3frameの母数を実1350frameと混同しない。"""
    monkeypatch.setattr(target.common, "INTERVALS", {target.MODE: (0.0, 0.1)})


def values() -> list[dict[str, Any]]:
    """必須3kind/左右を旧順序で生成する。"""
    return [{"kind": kind, "frame_idx": frame, "time_sec": frame / 60,
             "side": side, "value": 1} for frame in (0, 2, 4) for side in ("1P", "2P")
            for kind in ("frame_side", "raw_score_ocr", "completion_frame_observation")]


def decision(*, newer: bool, frame: int = 0) -> dict[str, Any]:
    """固定sourceから導出した既知caller差を使う。"""
    contract = target.caller_contract()
    metadata = contract["after"] if newer else contract["before"][0]
    return {"kind": contract["kind"], "decision_function": contract["decision_function"],
            "frame_idx": frame, "time_sec": frame / 60, "side": None,
            "args": [], "kwargs": {"current_next": [5, 5], "start_next": [5, 5]},
            "result": False, **metadata}


def marker() -> dict[str, Any]:
    """保存契約専用の人工介入で、実認識の変更とは呼ばない。"""
    return {"kind": "boundary_repair_mutation", "repair": target.MODE, "frame_idx": 2,
            "time_sec": 2 / 60, "side": "2P", "before": {}, "after": {},
            "evidence": {"synthetic_storage_fixture": True}}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """pytest一時directoryへだけ人工入力を書く。"""
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare(tmp_path: Path, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """原文を別pathへ固定して実比較器を呼ぶ。"""
    paths = (tmp_path / "old.jsonl", tmp_path / "new.jsonl")
    for path, rows in zip(paths, (before, after), strict=True):
        write_rows(path, rows)
    return target.compare_rows(*paths, target.MODE)


def test_declared_caller_diff_is_retained_not_raw_bitexact(tmp_path: Path, short_interval: None) -> None:
    report = compare(tmp_path, values() + [decision(newer=False)], values() + [decision(newer=True)])
    assert report["pre_mutation_prefix_bit_exact"] is False
    assert report["pre_mutation_prefix_exact_except_declared_caller_metadata"] is True
    assert report["all_legacy_rows_bit_exact"] is False and report["mutation_count"] == 0
    assert report["status"] == "NO_INTERVENTION_NOT_QUALITY_CLEAR"
    change = next(iter(report["complete_changed_rows"].values()))
    assert change["declared_caller_metadata_only"] and set(change["fields"]) == target.CALLER_FIELDS
    assert change["before"]["row"] == decision(newer=False)
    assert change["after"]["row"] == decision(newer=True)


@pytest.mark.parametrize("field", ["result", "kwargs", "args", "extra", "caller_line", "caller_file", "source", "decision_function"])
def test_prefix_rejects_any_undeclared_column(tmp_path: Path, short_interval: None, field: str) -> None:
    changed = decision(newer=True)
    changed[field] = "unapproved"
    with pytest.raises(ValueError, match="prefix"):
        compare(tmp_path, values() + [decision(newer=False)], values() + [changed])


@pytest.mark.parametrize("field", ["caller_line", "caller_function", "caller_file", "source"])
def test_old_caller_is_exact_too(tmp_path: Path, short_interval: None, field: str) -> None:
    old = decision(newer=False)
    old[field] = "unapproved_old"
    with pytest.raises(ValueError, match="prefix"):
        compare(tmp_path, values() + [old], values() + [decision(newer=True)])


@pytest.mark.parametrize("edit", ["missing", "added", "order", "frame_field"])
def test_prefix_requires_all_keys_order_and_other_values(tmp_path: Path, short_interval: None, edit: str) -> None:
    before, after = values(), values()
    if edit == "missing": after.pop(0)
    if edit == "added": after.insert(0, dict(after[0]))
    if edit == "order": after[0], after[1] = after[1], after[0]
    if edit == "frame_field": after[0]["value"] = 2
    after.append(marker())
    with pytest.raises(ValueError, match="prefix"):
        compare(tmp_path, before, after)


def test_post_intervention_all_diff_and_missing_are_retained(tmp_path: Path, short_interval: None) -> None:
    old = values() + [decision(newer=False), {"kind": "collector_snapshot", "frame_idx": 4, "time_sec": 4 / 60}]
    new = values() + [decision(newer=True), marker()]
    new[-3]["value"] = 2
    report = compare(tmp_path, old, new)
    assert report["mutation_count"] == 1 and len(report["missing_rows"]) == 1
    assert report["pre_mutation_prefix_bit_exact"] is False
    assert report["pre_mutation_prefix_exact_except_declared_caller_metadata"]
    assert len(report["complete_changed_rows"]) == 2
    assert report["quality_gate_clear"] is False


def test_whitespace_only_prefix_change_is_not_caller_allowance(tmp_path: Path, short_interval: None) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    write_rows(old, values())
    new.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in values()), encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        target.compare_rows(old, new, target.MODE)


def receipt() -> dict[str, Any]:
    """固定候補と明示依存を含むloader用receipt。"""
    return {"boundary_repair": {"mode": target.MODE, "module": str(target.MODULE),
             "module_sha256": target.MODULE_SHA}, "input_and_code_sha256": dict(target.FIXED_SHA)}


def test_alias_and_package_attribute_restore_even_on_exception() -> None:
    package = importlib.import_module("scripts")
    assert target.ALIAS not in sys.modules and not hasattr(package, target.ATTRIBUTE)
    with pytest.raises(RuntimeError, match="body"), contextlib.ExitStack() as stack:
        module = target.load_module(stack, receipt())
        assert module.base is sys.modules[target.ALIAS]
        assert getattr(package, target.ATTRIBUTE) is module.base
        raise RuntimeError("body")
    assert target.ALIAS not in sys.modules and not hasattr(package, target.ATTRIBUTE)
    assert "_video38_boundary_repair_chain_end" not in sys.modules


@pytest.mark.parametrize("location", ["module", "attribute"])
def test_unexpected_preload_is_rejected_without_touching_it(monkeypatch: pytest.MonkeyPatch, location: str) -> None:
    package, sentinel = importlib.import_module("scripts"), NS(unexpected=True)
    if location == "module": monkeypatch.setitem(sys.modules, target.ALIAS, sentinel)
    else: monkeypatch.setattr(package, target.ATTRIBUTE, sentinel, raising=False)
    with pytest.raises(RuntimeError, match="既load"), contextlib.ExitStack() as stack:
        target.load_module(stack, receipt())
    actual = sys.modules.get(target.ALIAS) if location == "module" else getattr(package, target.ATTRIBUTE)
    assert actual is sentinel


@pytest.mark.parametrize("path", [target.V1_SOURCE, target.V1_TEST, target.MODULE, target.MODULE_TEST])
def test_missing_fixed_dependency_guard_is_rejected(path: Path) -> None:
    value = receipt()
    value["input_and_code_sha256"].pop(str(path))
    with pytest.raises(RuntimeError, match="guard"), contextlib.ExitStack() as stack:
        target.load_module(stack, value)


def test_alias_is_removed_when_candidate_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any) -> Any:
        raise RuntimeError("candidate_import")
    monkeypatch.setattr(target, "ORIGINAL_LOAD", fail)
    with pytest.raises(RuntimeError, match="candidate_import"), contextlib.ExitStack() as stack:
        target.load_module(stack, receipt())
    assert target.ALIAS not in sys.modules
    assert not hasattr(importlib.import_module("scripts"), target.ATTRIBUTE)


def test_partial_alias_execution_failure_restores_both_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """alias自身のexec例外も候補import前に取り残さない。"""
    original = target.importlib.util.spec_from_file_location
    def broken(name: str, path: Path) -> Any:
        spec = original(name, path)
        def fail(module: Any) -> None:
            raise RuntimeError("alias_exec")
        monkeypatch.setattr(spec.loader, "exec_module", fail)
        return spec
    monkeypatch.setattr(target.importlib.util, "spec_from_file_location", broken)
    with pytest.raises(RuntimeError, match="alias_exec"), contextlib.ExitStack() as stack:
        target.load_module(stack, receipt())
    assert target.ALIAS not in sys.modules
    assert not hasattr(importlib.import_module("scripts"), target.ATTRIBUTE)


def test_prepare_records_non_bitexact_policy_and_explicit_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧commonの完全一致説明をそのまま新PLANへ残さない。"""
    monkeypatch.setattr(target, "ORIGINAL_PREPARE", lambda args: (receipt(), {}))
    args = NS(mode=target.MODE, module_sha256=target.MODULE_SHA,
              module_test_sha256=target.MODULE_TEST_SHA)
    value, _ = target.prepare(args)
    assert value["boundary_repair"]["comparison_contract"] == value["chain_end_v2_adapter"]["prefix_policy"]
    assert "except" in value["boundary_repair"]["comparison_contract"]
    assert all(value["input_and_code_sha256"][str(path)] == target.FIXED_SHA[str(path)]
               for path in (target.V1_SOURCE, target.V1_TEST))


def test_existing_output_is_rejected_without_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """既存common→baseの新規root必須契約をadapter越しに維持する。"""
    original = tmp_path / "keep.txt"
    original.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(target, "ORIGINAL_PREPARE", lambda args: (receipt(), {}))
    args = NS(mode=target.MODE, module_sha256=target.MODULE_SHA,
              module_test_sha256=target.MODULE_TEST_SHA, output_root=tmp_path,
              start_sec=560.0, end_sec=605.0, allow_native_runtime_mismatch=True)
    with pytest.raises(FileExistsError):
        target.run(args)
    assert original.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "PLAN.json").exists()


def test_common_hooks_restored_after_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ("MODULES", "prepare", "load_module", "compare_rows", "finish")
    original = {name: inspect.getattr_static(target.common, name) for name in names}
    def fail(args: Any) -> None:
        assert target.common.MODULES[target.MODE] == "chain_end_epoch_shadow_v2"
        raise RuntimeError("run_failure")
    monkeypatch.setattr(target.common, "run", fail)
    with pytest.raises(RuntimeError, match="run_failure"):
        target.run(NS(mode=target.MODE))
    assert all(inspect.getattr_static(target.common, name) is value for name, value in original.items())


def test_complete_is_last_and_writer_change_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = tmp_path / "input"
    guard.write_text("before", encoding="utf-8")
    value = {"input_and_code_sha256": {str(guard): target.base.sha256(guard)}}
    def finish(output: Path, *args: Any) -> dict[str, Any]:
        target.base.write_json(output / "SUMMARY.json", {})
        guard.write_text("changed", encoding="utf-8")
        target.base.write_json(output / "COMPLETE", {})
        return {}
    monkeypatch.setattr(target, "ORIGINAL_FINISH", finish)
    writer = target.base.write_json
    with pytest.raises(ValueError, match="変更"):
        target.finish(tmp_path, value, None, 1.0)
    assert not (tmp_path / "COMPLETE").exists() and target.base.write_json is writer


def test_function_limit_and_no_eager_v1_import() -> None:
    for path in (target.SOURCE, Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not [(node.name, node.end_lineno - node.lineno + 1) for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.end_lineno - node.lineno + 1 > 50]
    assert target.ALIAS not in sys.modules


FRESH_PROBE = r'''
import argparse, contextlib, inspect, json, os, sys, types
from pathlib import Path
from scripts import diagnose_video38_chain_end_shadow_v2 as adapter
base, common = adapter.base, adapter.common
output = Path(sys.argv[1]).resolve()
args = argparse.Namespace(mode=adapter.MODE,start_sec=560.,end_sec=605.,output_root=output,
    module_sha256=adapter.MODULE_SHA,module_test_sha256=adapter.MODULE_TEST_SHA,
    allow_native_runtime_mismatch=True)
hashes={str(p):base.sha256(p) for p in (adapter.SOURCE,adapter.TEST,adapter.LAUNCHER)}
hashes.update(adapter.FIXED_SHA)
original_video, original_load, original_write = base.instrument_video, base.load_collector, base.write_json
descriptors, writes = [], []
def loaded():
    collector=original_load()
    owners=[collector,collector.RecognitionPipeline]
    owners += [m for n,m in sys.modules.items() if n.startswith('src.') and isinstance(m,types.ModuleType)]
    for obj in list(owners):
        if isinstance(obj,types.ModuleType):
            owners += [v for v in vars(obj).values() if isinstance(v,type) and v.__module__==obj.__name__]
    for obj in owners:
        for name in vars(obj):
            old=inspect.getattr_static(obj,name)
            if callable(old) or isinstance(old,(staticmethod,classmethod)): descriptors.append((obj,name,old))
    return collector
def installed(stack,collector,rec):
    original_video(stack,collector,rec)
    def collect(video,out,**kwargs):
        assert kwargs['start_sec']==560. and kwargs['max_sec']==45.
        assert kwargs['normalize_fps_30'] is True and kwargs['sample_interval_sec']==0
        module=sys.modules['_video38_boundary_repair_chain_end']
        assert module.base is sys.modules[adapter.ALIAS]
        assert Path(module.base.__file__).resolve()==adapter.V1_SOURCE
        assert rec.generation_recorder is not None and rec.transform_receipt['matched_blocks']==1
        pipeline=sys.modules[collector.RecognitionPipeline.__module__]
        assert Path(pipeline.__file__).is_relative_to(base.SNAPSHOT)
        assert pipeline._is_game_event_chain_exit((1,2),(2,3)) is True
        explicit={'src.chain_commit_candidate_v1','src.chain_prediction_ledger_v1'}
        for name,value in list(sys.modules.items()):
            if name.startswith('src.') and getattr(value,'__file__',None) and name not in explicit:
                assert Path(value.__file__).is_relative_to(base.SNAPSHOT),(name,value.__file__)
        contract=adapter.caller_contract()
        reference=base.VERIFY/common.REFERENCE_ROOTS[adapter.MODE]/'frames.jsonl'
        count=metadata=snapshots=0
        for line in reference.open():
            row=json.loads(line)
            before={key:row.get(key) for key in adapter.CALLER_FIELDS}
            if row['kind']==contract['kind'] and row.get('decision_function')==contract['decision_function'] and before in contract['before']:
                row.update(contract['after']); line=json.dumps(row,ensure_ascii=False)+'\n'; metadata+=1
            if row['kind']=='frame_side' and row['frame_idx']==35776 and row.get('side')=='2P':
                row['cpu_storage_fixture_difference']=True; line=json.dumps(row,ensure_ascii=False)+'\n'
            rec.stream.write(line); count+=1; snapshots+=row['kind']=='collector_snapshot'
        marker={'kind':'boundary_repair_mutation','repair':'chain_end','side':'2P',
            'frame_idx':35776,'time_sec':35776/60,'before':{},'after':{},
            'evidence':{'synthetic_storage_fixture':True}}
        rec.stream.write(json.dumps(marker)+'\n')
        assert metadata==19
        rec.frames,rec.rows,rec.snapshots=1350,count+1,snapshots
        rec.pipeline_receipt={'device':'not_loaded_CPU_fixture','recognition_performed':False}
        rec.model_loads=[{'scope':'CPU stored-log replay; no checkpoint loaded'}]
        raise base.CollectionFinished()
    base.patch(stack,collector,'collect_lean',collect)
def writer(path,value):
    if path.parent==output:
        assert not (output/'COMPLETE').exists()
        writes.append(path.name)
    original_write(path,value)
cwd, search = Path.cwd(),list(sys.path)
with contextlib.ExitStack() as stack:
    for name,value in [('load_collector',loaded),('instrument_video',installed),('write_json',writer)]:
        base.patch(stack,base,name,value)
    report=adapter.run(args)
assert Path.cwd()==cwd and sys.path==search
assert all(inspect.getattr_static(obj,name) is old for obj,name,old in descriptors)
assert adapter.ALIAS not in sys.modules and not hasattr(sys.modules['scripts'],adapter.ATTRIBUTE)
assert '_video38_boundary_repair_chain_end' not in sys.modules
assert 'src.chain_commit_candidate_v1' not in sys.modules
assert 'src.chain_prediction_ledger_v1' not in sys.modules
plan=base.read_json(output/'PLAN.json'); complete=base.read_json(output/'COMPLETE')
assert writes[-1]=='COMPLETE' and len(complete['sha256'])==8
base.assert_unchanged({str(output/name):sha for name,sha in complete['sha256'].items()})
base.assert_unchanged(plan['input_and_code_sha256']); base.assert_unchanged(hashes)
comparison=report['comparison']
assert comparison['pre_mutation_prefix_bit_exact'] is False
assert comparison['pre_mutation_prefix_exact_except_declared_caller_metadata'] is True
assert len(comparison['changed_keys'])==20 and comparison['mutation_count']==1
assert comparison['reference_rows']==comparison['candidate_rows']==10341
assert comparison['quality_gate_clear'] is False
import torch
assert not torch.cuda.is_initialized()
print(json.dumps({'manufacturing_cpu_driver_only':True,'artifact_sha256':complete['sha256'],
    'guard_count':len(plan['input_and_code_sha256']),'descriptor_entries_restored':len(descriptors),
    'caller_differences':19,'synthetic_functional_difference':1,'cuda_initialized':False}))
'''


def test_fresh_frozen_common_prepare_finish_eight_artifacts(tmp_path: Path) -> None:
    """実frozen/import/prepare/finishを通し、認識だけ保存ログfixtureに置換する。"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", FRESH_PROBE, str(tmp_path / "run")],
                            cwd=target.base.ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    assert result.returncode == 0, result.stdout
    assert '"manufacturing_cpu_driver_only": true' in result.stdout
    print(result.stdout)
