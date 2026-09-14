"""実TorchVersion値だけを文字列へ写し、旧strict receipt/保存を維持する。"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterator

import torch.torch_version as VERSION_MODULE

ROOT = Path(__file__).resolve().parent
OLD_PATH = ROOT.parent / "g2_finisher_receipt_repair_2026-09-08_v1/adapter.py"
OLD_SHA = "e82d1cac690f77bbc819cecad29e0de294bab8fa7b94d2167ceb3b40498692a6"
VERSION_PATH = Path(VERSION_MODULE.__file__).resolve()
VERSION_SHA = "46eb73ac402b16bdd8971a9736fbb063f0baf708ec2dd7dfba244f9e0fe00b33"
TORCH_VERSION_TYPE = VERSION_MODULE.TorchVersion
OWN = ("adapter.py", "test_adapter.py", "run_cpu.py", "ASSET_PREFLIGHT.md", "reproduce.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_old() -> Any:
    alias = "_torch_version_strict_receipt"
    if sha(OLD_PATH) != OLD_SHA or alias in sys.modules:
        raise RuntimeError("old_receipt_source_or_alias_changed")
    spec = importlib.util.spec_from_file_location(alias, OLD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


OLD = load_old()
ORIGINAL_NORMALIZE, ORIGINAL_GUARDS = OLD.normalize, OLD.guards


def version_values(value: Any) -> Any:
    """既知containerの値だけ複製。未知型やdict keyは旧拒否器へ渡す。"""
    kind = type(value)
    if kind is TORCH_VERSION_TYPE:
        return str.__str__(value)
    if kind is list:
        return [version_values(item) for item in value]
    if kind is tuple:
        return tuple(version_values(item) for item in value)
    if kind is dict:
        return {key: version_values(item) for key, item in value.items()}
    return value


def normalize(value: Any) -> Any:
    return ORIGINAL_NORMALIZE(version_values(value))


def guards() -> dict[str, str]:
    if sys.modules.get("torch.torch_version") is not VERSION_MODULE or VERSION_MODULE.TorchVersion is not TORCH_VERSION_TYPE:
        raise RuntimeError("torch_version_module_identity_changed")
    if sha(OLD_PATH) != OLD_SHA or sha(VERSION_PATH) != VERSION_SHA:
        raise RuntimeError("torch_version_fixed_source_changed")
    result = dict(ORIGINAL_GUARDS())
    extra = {str(OLD_PATH): OLD_SHA, str(VERSION_PATH): VERSION_SHA,
             **{str(ROOT / name): sha(ROOT / name) for name in OWN}}
    for path, digest in extra.items():
        if path in result and result[path] != digest:
            raise RuntimeError("torch_version_guard_conflict")
        result[path] = digest
    return result


@contextlib.contextmanager
def policy() -> Iterator[None]:
    """元の正規化とguardへ戻るまでを単一scopeに限定する。"""
    guards()
    if OLD.normalize is not ORIGINAL_NORMALIZE or OLD.guards is not ORIGINAL_GUARDS:
        raise RuntimeError("torch_version_policy_reentry")
    with contextlib.ExitStack() as stack:
        OLD.F.base.patch(stack, OLD, "normalize", normalize)
        OLD.F.base.patch(stack, OLD, "guards", guards)
        yield


def install(stack: contextlib.ExitStack, runtime: Any) -> None:
    # 新policyだけを前置し、元F/実historyの保存分岐は一切変更しない。
    with contextlib.ExitStack() as pending:
        pending.enter_context(policy())
        OLD.install(pending, runtime)
        stack.enter_context(pending.pop_all())
