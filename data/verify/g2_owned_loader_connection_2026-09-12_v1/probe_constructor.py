"""原load_default/attachの実生成を再用し、新owned選択を返却後に照合する。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace as N
from typing import Any
import owned_adapter as R

ROOT = Path(__file__).resolve().parent


def execute(output: Path) -> dict:
    spec = importlib.util.spec_from_file_location('_owned_constructor_probe_base', R.PRIOR / 'probe_actual_constructor.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    seen: dict[str, Any] = dict(builds=0)
    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == 'call' and frame.f_code.co_name == 'build' and frame.f_globals.get('__name__') == R.OWNED_ALIAS:
            seen['builds'] += 1
    original = base.C.inspect_closed
    def inspect(main: Any, kept: dict, expected: Any, path: Path) -> dict:
        result = original(main, kept, expected, path)
        context = kept['state']['live_empty_reset_context']
        origins = [c.__dict__['perform'].__globals__['__file__'] for c in type(context).__mro__
                   if 'perform' in c.__dict__]
        assert str(R.SIDE / 'side_actual_connection.py') in origins, 'actual_constructor_new_side_missing'
        assert str(R.VERIFY / 'g2_hidden_basis_initialization_2026-09-11_v1/side_actual_connection.py') not in origins
        assert seen['builds'] == 1, 'actual_constructor_owned_builds'
        return result | dict(owned_builds=seen['builds'], context_perform_origins=origins, new_side_selected=True)
    proxy = N(configured=R.configured, FINISH_MODULES=R.A.A.FINISH_MODULES)
    function = FunctionType(base.execute.__code__, dict(vars(base), A=proxy))
    with ExitStack() as stack:
        R.replace_owned(stack, base.C, 'inspect_closed', inspect)
        sys.setprofile(profile)
        stack.callback(sys.setprofile, None)
        return function(output)


def main() -> int:
    output = ROOT / 'constructor_v1'
    assert not output.exists(), 'exclusive_constructor'
    started, code = time.perf_counter(), 0
    try:
        result = execute(output)
    except BaseException:
        result, code = dict(error=traceback.format_exc(), quality_gate_clear=False), 1
        output.mkdir(exist_ok=True)
    result.update(exit_code=code, seconds=time.perf_counter()-started)
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
