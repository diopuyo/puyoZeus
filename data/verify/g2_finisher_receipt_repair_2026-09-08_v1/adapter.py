"""診断receiptのtuple/list差だけを正規化し、旧保存契約を維持する。"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
LEGACY = HERE.parent / "g2_split_diagnostic_finisher_2026-09-08_v1/finisher.py"
LEGACY_SHA = "96541c1e7df8b555e26bd5cca742d16060ba46f06c925e675adaacb5decc0702"
OWN = ("adapter.py", "test_adapter.py", "run_cpu.py", "ASSET_PREFLIGHT.md")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy() -> Any:
    if sha(LEGACY) != LEGACY_SHA:
        raise RuntimeError("legacy_finisher_source_changed")
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location("_receipt_repair_legacy_finisher", LEGACY)
    if spec.name in sys.modules:
        raise RuntimeError("legacy_finisher_alias_in_use")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


F = load_legacy()
ORIGINAL_VALIDATE = F.validate_start


def normalize(value: Any) -> Any:
    """exact型だけを複製する。並びや数値型を変えずtupleをarrayへ写す。"""
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite_receipt_number")
        return value
    if kind in (tuple, list):
        return [normalize(item) for item in value]
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("nonstring_receipt_key")
        return {key: normalize(item) for key, item in value.items()}
    raise ValueError("unsupported_receipt_type:" + kind.__name__)


def canonical_bytes(value: Any) -> bytes:
    """既存canonical_json_bytes方式。PythonのFalse==0等に依存しない。"""
    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_receipt_json_key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("nonfinite_receipt_json_constant:" + value)


def strict_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    if type(value) is not dict:
        raise ValueError("receipt_root_must_be_object")
    return normalize(value)


def guards() -> dict[str, str]:
    values = dict(F.guards())
    for name in OWN:
        path = str(HERE / name)
        digest = sha(Path(path))
        if path in values and values[path] != digest:
            raise RuntimeError("receipt_repair_guard_conflict")
        values[path] = digest
    return values


def validate_start(output: Path, receipt: dict[str, Any], elapsed: float) -> None:
    """旧関数の実readだけstrict化。元receipt/PLANへ書いて戻す方式は使わない。"""
    if type(receipt) is not dict:
        raise ValueError("receipt_root_must_be_object")
    expected, detached = canonical_bytes(receipt), normalize(receipt)
    original, plan_path = F.base.read_json, (output / "PLAN.json").resolve()
    def read(path: Path) -> Any:
        if Path(path).resolve() != plan_path:
            return original(path)
        value = strict_plan(path)
        if canonical_bytes(value) != expected:
            raise ValueError("receipt_plan_mismatch")
        return value
    with contextlib.ExitStack() as stack:
        F.base.patch(stack, F.base, "read_json", read)
        ORIGINAL_VALIDATE(output, detached, elapsed)
    actual = receipt["input_and_code_sha256"]
    if any(actual.get(path) != digest for path, digest in guards().items()):
        raise ValueError("receipt_repair_prepare_guard_missing")


def install(stack: contextlib.ExitStack, runtime: Any) -> None:
    """既存captured finishを使い、途中失敗でも差替えを即座に復元する。"""
    guards()
    if F.validate_start is not ORIGINAL_VALIDATE:
        raise RuntimeError("receipt_repair_reentry")
    with contextlib.ExitStack() as pending:
        F.base.patch(pending, F, "validate_start", validate_start)
        F.install(pending, runtime)
        stack.enter_context(pending.pop_all())
