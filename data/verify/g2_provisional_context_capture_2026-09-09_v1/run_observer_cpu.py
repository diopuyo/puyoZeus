"""限定CPUを排他保存し、source前後・実exit・索引を分離して記録する。"""
from __future__ import annotations
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> int:
    if not __debug__:
        raise RuntimeError("optimized_python_not_supported")
    output = ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    os.environ["MANUFACTURING_OUTPUT"] = str(output)
    runtime = "--runtime" in sys.argv[2:]
    os.environ["CONTEXT_ACTUAL_RUNTIME"] = "1" if runtime else "0"
    paths = [ROOT / name for name in ("observer.py", "test_observer.py", "run_observer_cpu.py", "OBSERVER_CONTRACT.md")]
    spec = importlib.util.spec_from_file_location("_context_cpu_guard_module", ROOT / "observer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before, started = module.guards(), time.perf_counter()
    write(output / "PLAN.json", {"pid": os.getpid(), "before": before, "actual_runtime": runtime,
        "artificial_reader": True, "gpu_permitted": False})
    import pytest
    args = [str(ROOT / "test_observer.py"), "-q", "--tb=short", "-p", "no:cacheprovider",
        "--confcutdir", str(ROOT), "--basetemp", str(output / "tmp")]
    args += ["-k", "frozen_actual" if runtime else "not frozen_actual"]
    with (output / "PYTEST.log").open("x", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main(args))
    after = {path: sha(Path(path)) for path in before}
    long = [path.name + ":" + node.name for path in paths[:3]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno + 1 > 50]
    torch = sys.modules.get("torch")
    cuda = bool(torch and torch.cuda.is_initialized())
    good = code == 0 and before == after and not long and not cuda
    result = {"pid": os.getpid(), "pytest_exit": code, "seconds": time.perf_counter() - started,
        "before": before, "after": after, "functions_over_50": long, "cuda_initialized": cuda,
        "passed": good, "quality_gate_clear": False, "actual_runtime": runtime}
    write(output / "RESULT.json", result)
    hashes = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}
    write(output / ("COMPLETE.json" if good else "FAILED.json"), {"sha256": hashes, "exit_code": 0 if good else 1})
    print(json.dumps({key: value for key, value in result.items() if key not in ("before", "after")}), flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
