"""親binderの限定CPU結果と前後sourceを保存する。"""
from __future__ import annotations
import ast
import contextlib
import hashlib
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
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def main() -> int:
    output = ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    selected = "test_parent_capture.py" if "--capture" in sys.argv[2:] else "test_binding.py"
    paths = [ROOT / name for name in ("binding.py", selected, "run_binding_cpu.py", "saved_context.py", "analyze_context.py")]
    if selected == "test_parent_capture.py":
        paths += [ROOT / name for name in ("observer.py", "test_observer.py", "run_observer_cpu.py", "OBSERVER_CONTRACT.md")]
    before = {str(path): sha(path) for path in paths}
    started = time.perf_counter()
    import pytest
    with (output / "PYTEST.log").open("x", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(ROOT / selected), "-q", "--tb=short", "--confcutdir", str(ROOT),
                "-p", "no:cacheprovider", "--basetemp", str(output / "tmp")]))
    after = {path: sha(Path(path)) for path in before}
    long = [path.name + ":" + node.name for path in paths if path.suffix == ".py" for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno + 1 > 50]
    import torch
    good = code == 0 and before == after and not long and not torch.cuda.is_initialized()
    result = {"pid": os.getpid(), "actual_pytest_exit": code, "seconds": time.perf_counter() - started,
        "before": before, "after": after, "functions_over_50": long, "passed": good,
        "cuda_initialized": torch.cuda.is_initialized(), "artificial_update_context": True, "quality_gate_clear": False}
    write(output / "RESULT.json", result)
    write(output / ("COMPLETE.json" if good else "FAILED.json"), {"sha256":
        {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}})
    print({key: value for key, value in result.items() if key not in ("before", "after")}, flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
