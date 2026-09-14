"""固定chain-end v2を旧runnerへ接続し、宣言済みcaller差だけを分離する。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts import diagnose_video38_boundary_repair_shadow_v1 as common


base = common.base
MODE = "chain_end"
FORMAT = "video38-chain-end-shadow/v2"
SOURCE = Path(__file__).resolve()
TEST = base.ROOT / "tests/test_diagnose_video38_chain_end_shadow_v2.py"
LAUNCHER = base.ROOT / "scripts/launch_video38_chain_end_shadow_v2.sh"
MODULE = base.ROOT / "scripts/chain_end_epoch_shadow_v2.py"
MODULE_TEST = base.ROOT / "tests/test_chain_end_epoch_shadow_v2.py"
V1_SOURCE = base.ROOT / "scripts/chain_end_epoch_shadow_v1.py"
V1_TEST = base.ROOT / "tests/test_chain_end_epoch_shadow_v1.py"
PIPELINE = base.SNAPSHOT / "src/recognition_pipeline.py"
MODULE_SHA = "a5300425466017335258fb37b534dc3a5eb070c2ff9d8059946f2bafa5e3ccb0"
MODULE_TEST_SHA = "3268a0adf86e91aff523d872ad55015c5c1788adc9e9a1ae5a2806ca3e569820"
FIXED_SHA = {
    str(MODULE): MODULE_SHA, str(MODULE_TEST): MODULE_TEST_SHA,
    str(V1_SOURCE): "06dc6ce0e3bdf962def5420fd0312acf2ecd395ca099f96afcba1ef8bd6fa480",
    str(V1_TEST): "215c4c4303331daf082dd19f22cf6f7c5fc6e8291582748d8862880303d60da1",
    str(PIPELINE): "6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02",
    str(Path(common.__file__).resolve()): "e65d368be8a68074184450811a9a112f8517b945d424530f47ebab41217ff453",
    str(common.TEST): "dd978ceff4ec1eb5b0f67b8966eb0d45a1f0db0d708968f8ce6bd30da6715d35",
    str(common.LAUNCHER): "7c4851027202873d58b6d087cdcfcdba082a4a14a7995ebc431f9bb6f0665806",
}
CALLER_FIELDS = frozenset(("caller_function", "caller_file", "caller_line", "source"))
ALIAS = "scripts.chain_end_epoch_shadow_v1"
ATTRIBUTE = "chain_end_epoch_shadow_v1"
ORIGINAL_PREPARE = common.prepare
ORIGINAL_LOAD = common.load_module
ORIGINAL_FINISH = common.finish


def _caller_sites(path: Path, function: str, callee: str) -> list[dict[str, Any]]:
    """固定sourceの実Call行からcaller metadataを導出する。"""
    source = path.read_text(encoding="utf-8-sig")
    functions = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef) and node.name == function]
    if len(functions) != 1:
        raise ValueError("caller関数が一意ではありません")
    calls = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == callee]
    return [{"caller_function": function, "caller_file": str(path.resolve()),
             "caller_line": node.lineno, "source": source.splitlines()[node.lineno - 1].strip()}
            for node in sorted(calls, key=lambda value: value.lineno)]


def caller_contract() -> dict[str, Any]:
    """任意wrapperの許容ではなく固定2caller→固定1callerだけを宣言する。"""
    base.assert_unchanged(FIXED_SHA)
    before = _caller_sites(PIPELINE, "update", "_is_game_event_chain_exit")
    after = _caller_sites(MODULE, "exit_signal", "original_exit")
    if len(before) != 2 or len(after) != 1:
        raise ValueError("原本2分岐またはwrapper1箇所のcaller契約が変化しました")
    return {"fields": sorted(CALLER_FIELDS), "before": before, "after": after[0],
            "kind": "exit_decision_detail", "decision_function": "_is_game_event_chain_exit"}


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """旧全guardに新3fileと型注釈付き依存の固定SHAを明示追加する。"""
    if args.mode != MODE or (args.module_sha256, args.module_test_sha256) != (MODULE_SHA, MODULE_TEST_SHA):
        raise ValueError("検収済みchain_end v2以外は実行しません")
    contract = caller_contract()
    receipt, config = ORIGINAL_PREPARE(args)
    hashes = receipt["input_and_code_sha256"]
    hashes.update(FIXED_SHA)
    hashes.update({str(path): base.sha256(path) for path in (SOURCE, TEST, LAUNCHER)})
    base.assert_unchanged(hashes)
    receipt["chain_end_v2_adapter"] = {"format_version": FORMAT,
        "explicit_annotated_guard_dependencies": [str(V1_SOURCE), str(V1_TEST)],
        "caller_metadata_contract": contract, "legacy_raw_rows_preserved": True,
        "prefix_policy": "exact except only declared caller metadata; all other columns/order/absence exact",
        "production_adoption": False, "quality_gate_clear": False}
    receipt["boundary_repair"]["comparison_contract"] = receipt["chain_end_v2_adapter"]["prefix_policy"]
    return receipt, config


def _remove_alias(package: ModuleType, module: ModuleType) -> None:
    """追加したaliasとpackage属性だけを元の不在状態へ戻す。"""
    sys.modules.pop(ALIAS, None)
    if hasattr(package, ATTRIBUTE):
        delattr(package, ATTRIBUTE)


def _preload_v1(stack: contextlib.ExitStack, guards: dict[str, str]) -> ModuleType:
    """frozen namespaceでもv1の実体が変わらないよう絶対pathで先行loadする。"""
    package = importlib.import_module("scripts")
    if ALIAS in sys.modules or hasattr(package, ATTRIBUTE):
        raise RuntimeError("v1 aliasまたはpackage属性が予期せず既loadです")
    required = {str(path): FIXED_SHA[str(path)] for path in (V1_SOURCE, V1_TEST)}
    if any(guards.get(path) != digest for path, digest in required.items()):
        raise RuntimeError("v1 source/testの固定依存guardがありません")
    base.assert_unchanged(required)
    spec = importlib.util.spec_from_file_location(ALIAS, V1_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("v1 aliasのspecを生成できません")
    module = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = module
    setattr(package, ATTRIBUTE, module)
    stack.callback(_remove_alias, package, module)
    spec.loader.exec_module(module)
    base.assert_unchanged(required)
    return module


def load_module(stack: contextlib.ExitStack, receipt: dict[str, Any]) -> ModuleType:
    """aliasも候補本体も同じExitStackの成功・例外復元に束縛する。"""
    info = receipt["boundary_repair"]
    if info["mode"] != MODE or Path(info["module"]).resolve() != MODULE.resolve():
        raise RuntimeError("v2専用module path/modeが不一致です")
    guards = receipt["input_and_code_sha256"]
    if info["module_sha256"] != MODULE_SHA or any(guards.get(path) != digest for path, digest in FIXED_SHA.items()):
        raise RuntimeError("v2と再利用資産の固定guardが不一致です")
    base.assert_unchanged(FIXED_SHA)
    alias = _preload_v1(stack, guards)
    module = ORIGINAL_LOAD(stack, receipt)
    if module.base is not alias or module.GUARD_PATHS != ("scripts/chain_end_epoch_shadow_v1.py",):
        raise RuntimeError("v2の実v1依存が固定aliasと不一致です")
    return module


def declared_caller_only(before: dict[str, Any], after: dict[str, Any],
                         fields: dict[str, Any], contract: dict[str, Any]) -> bool:
    """派生監査の4列限定判定を再利用し、旧source値も厳密に照合する。"""
    if set(fields) != CALLER_FIELDS:
        return False
    if any(row.get(key) != contract[key] for row in (before, after)
           for key in ("kind", "decision_function")):
        return False
    old = {key: before.get(key) for key in CALLER_FIELDS}
    new = {key: after.get(key) for key in CALLER_FIELDS}
    return old in contract["before"] and new == contract["after"]


def _differences(old: dict[str, Any], new: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """全差分のrow/textと各列before/afterを残し、metadataを削除しない。"""
    changes, counts = {}, Counter()
    for key, value in new.items():
        if key not in old or old[key]["text"] == value["text"]:
            continue
        before, after = old[key]["row"], value["row"]
        fields = common.changed_fields(before, after)
        changes[key] = {"fields": fields, "before": old[key], "after": value,
                        "declared_caller_metadata_only": declared_caller_only(before, after, fields, contract)}
        counts.update(f"{after['kind']}.{field}" for field in fields)
    return {"changed_keys": list(changes), "changed_field_counts": dict(counts),
            "changed_fields": {key: value["fields"] for key, value in changes.items()},
            "complete_changed_rows": changes,
            "missing_keys": [key for key in old if key not in new],
            "added_keys": [key for key in new if key not in old],
            "missing_rows": {key: value for key, value in old.items() if key not in new},
            "added_rows": {key: value for key, value in new.items() if key not in old},
            "common_key_order_equal": [key for key in old if key in new] == [key for key in new if key in old]}


def _prefix(values: dict[str, Any], first: float) -> list[str]:
    """時計Noneも旧実装同様にprefixへ残し、除外しない。"""
    return [key for key, value in values.items()
            if (value["row"]["frame_idx"] if value["row"]["frame_idx"] is not None else -1) < first]


def _check_prefix(old: dict[str, Any], new: dict[str, Any], diff: dict[str, Any], first: float) -> bool:
    """非caller列、欠落、追加、同frame順序の介入前変化は必ず拒否する。"""
    keys = _prefix(old, first)
    if keys != _prefix(new, first):
        raise ValueError("初回介入前のprefixに欠落/追加/順序差があります")
    raw_exact = True
    for key in keys:
        if old[key]["text"] == new[key]["text"]:
            continue
        raw_exact = False
        if not diff["complete_changed_rows"][key]["declared_caller_metadata_only"]:
            raise ValueError("初回介入前のprefixに未宣言の列またはtext差があります")
    return raw_exact


def compare_rows(reference: Path, candidate: Path, mode: str) -> dict[str, Any]:
    """原文完全一致とcaller宣言差を除く一致を別々に保存する。"""
    if mode != MODE:
        raise ValueError("chain_end以外の比較は実行しません")
    inputs = {str(path): base.sha256(path) for path in (reference, candidate)}
    inputs.update(FIXED_SHA)
    contract = caller_contract()
    old, prior = common.read_rows(reference)
    new, repairs = common.read_rows(candidate)
    if prior:
        raise ValueError("対照に新修正行が既にあります")
    mutations = common.validate_repairs(repairs, mode)
    first = min(mutations) if mutations else common.INTERVALS[mode][1] * common.history.FPS
    diff = _differences(old, new, contract)
    raw_exact = _check_prefix(old, new, diff, first)
    result = {"reference_rows": len(old), "candidate_rows": len(new), "repair_rows": len(repairs),
        "mutation_count": len(mutations), "first_mutation_frame": min(mutations) if mutations else None,
        "pre_mutation_prefix_bit_exact": raw_exact,
        "pre_mutation_prefix_exact_except_declared_caller_metadata": True,
        "all_legacy_rows_bit_exact": list(old.items()) == list(new.items()),
        "declared_caller_contract": contract, **diff,
        "coverage": common.coverage(new, mode), "input_sha256": inputs, "quality_gate_clear": False,
        "status": "INTERVENTION_REQUIRES_CAUSAL_QA" if mutations else "NO_INTERVENTION_NOT_QUALITY_CLEAR"}
    base.assert_unchanged(inputs)
    return result


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
    """旧8artifact保存を維持し、最終COMPLETEの直前にも全入力guardを確認する。"""
    writer = base.write_json
    def guarded_write(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            base.assert_unchanged(receipt["input_and_code_sha256"])
        writer(path, value)
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", guarded_write)
        return ORIGINAL_FINISH(output, receipt, rec, elapsed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """既存commonの一時hookだけを差替え、他候補/既存moduleは変更しない。"""
    if args.mode != MODE:
        raise ValueError("chain_end modeだけを扱います")
    with contextlib.ExitStack() as stack:
        replacements = {"MODULES": {**common.MODULES, MODE: "chain_end_epoch_shadow_v2"},
                        "prepare": prepare, "load_module": load_module,
                        "compare_rows": compare_rows, "finish": finish}
        for name, value in replacements.items():
            base.patch(stack, common, name, value)
        return common.run(args)


def main() -> int:
    """検収済み2SHAと既存45秒区間を固定し、新規rootだけへ保存する。"""
    common.pending.completion._configure_torch_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    args.mode, args.module_sha256, args.module_test_sha256 = MODE, MODULE_SHA, MODULE_TEST_SHA
    args.start_sec, args.end_sec = common.INTERVALS[MODE]
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "comparison"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
