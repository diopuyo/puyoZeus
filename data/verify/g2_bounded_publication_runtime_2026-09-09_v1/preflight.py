"""合成後のprepare guardと遅延factoryをGPU前に検収する。"""
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
    started, code = time.perf_counter(), 0
    result: dict[str, Any] = {'quality_gate_clear': False, 'gpu_executed': False}
    try:
        prior, driver, addon, engine = L.bootstrap()
        result['premature_src_modules'] = [n for n in sys.modules if n == 'src' or n.startswith('src.')]
        L.F.require(not result['premature_src_modules'], 'src_before_frozen_factory')
        path = L.ROOT.parent / 'g2_split_runtime_evidence_adapter_2026-09-08_v1/prepare_preflight.py'
        prepare = driver.load_fixed('_metadata_combined_split_prepare', path,
            '88722830ec8f34a07b4f4bc95bce187fff8c59e9287fb9be1fbc42525e908332')
        with L.configuration(prior, driver):
            runtime, finisher = driver.load_runtime()
            with contextlib.ExitStack() as stack:
                runtime.M.base.patch(stack, prepare, 'bootstrap', lambda: runtime)
                with L.configured(runtime, finisher, driver, addon, engine):
                    prepare.execute(output, result)
        receipt = json.loads((output / 'PREPARE_RECEIPT.json').read_text())
        required = addon.guards() | finisher.guards() | driver.own_guards()
        L.F.require(all(receipt['input_and_code_sha256'].get(p) == h for p, h in required.items()), 'prepare_guard_missing')
    except BaseException:
        result['error'], code = traceback.format_exc(), 1
    result.update(actual_exit=code, seconds=time.perf_counter() - started)
    L.F.P.write(output / 'RESULT.json', result)
    L.F.P.write(output / ('COMPLETE.json' if code == 0 else 'FAILED.json'), {'sha256':
        {str(p.relative_to(output)): L.F.sha(p) for p in output.rglob('*') if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
