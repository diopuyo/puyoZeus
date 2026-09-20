"""実合成prepareの全guardとsrc早期混入を検査する。"""
from __future__ import annotations
import contextlib
import json
import sys
import time
import traceback
from typing import Any
import live_cli as L


def main() -> int:
    output = L.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result: dict[str, Any] = {"quality_gate_clear": False, "gpu_executed": False}
    code = 0
    try:
        prior, driver, addon, engine = L.bootstrap()
        result["premature_src_modules"] = [name for name in sys.modules if name == "src" or name.startswith("src.")]
        if result["premature_src_modules"]:
            raise RuntimeError("src_before_frozen_factory")
        path = L.ROOT.parent / "g2_split_runtime_evidence_adapter_2026-09-08_v1/prepare_preflight.py"
        prepare = driver.load_fixed("_context_split_prepare", path,
            "88722830ec8f34a07b4f4bc95bce187fff8c59e9287fb9be1fbc42525e908332")
        with L.L.L.configuration(prior, driver):
            runtime, finisher = driver.load_runtime()
            with contextlib.ExitStack() as stack:
                runtime.M.base.patch(stack, prepare, "bootstrap", lambda: runtime)
                with driver.configured(runtime, finisher, addon):
                    engine.install(stack, runtime)
                    prepare.execute(output, result)
        receipt = json.loads((output / "PREPARE_RECEIPT.json").read_text())
        required = addon.guards() | finisher.guards() | driver.own_guards()
        if any(receipt["input_and_code_sha256"].get(path) != digest for path, digest in required.items()):
            raise RuntimeError("context_prepare_guard_missing")
    except BaseException:
        result["error"], code = traceback.format_exc(), 1
    result.update(actual_exit=code, seconds=time.perf_counter() - started)
    L.O.write(output / "RESULT.json", result)
    if code == 0:
        L.O.write(output / "COMPLETE.json", {"status": "PREPARE_ONLY", "sha256":
            {str(path.relative_to(output)): L.O.sha(path) for path in output.rglob("*") if path.is_file()}})
    print({key: value for key, value in result.items() if key not in ("before", "after")}, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
