"""A23実configuredの私有loaderで新接続の到達/型不変を確認。動画更新は行わない。"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any
import session_runtime_binding as B

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_split_tail_runtime_2026-09-13_v23'


def main() -> None:
    sys.path.insert(0, str(RUNTIME))
    import owned_adapter as A
    states: list[dict] = []
    with ExitStack() as stack:
        selected = A.configured(stack)
        original_adapter = A.configured.__globals__['A'].A.A.V4
        replace = original_adapter.replace_owned
        owner = sys.modules[original_adapter.OWNED_ALIAS]
        original = owner.bootstrap
        def bootstrap() -> Any:
            value = original()
            if not states:
                states.append(dict(owner=value, state=B.install(stack, value, replace)))
            elif states[0]['owner'] is not value:
                raise ValueError('origin_probe_bootstrap_owner')
            return value
        replace(stack, owner, 'bootstrap', bootstrap)
        with selected.__globals__['S'].configured() as env:
            binding = None if not states else states[0]['state']['binding']
            result = dict(bootstrap_reached=bool(states), session_class_bound=binding is not None,
                          video_updates=0, live_hook_verified=False, quality_gate_clear=False)
            if binding is not None:
                result.update(session_module=binding['module'].__file__,
                    reader_W_is_original=binding['reader'].W is binding['module'].W,
                    class_identity_preserved=binding['module'].Session is binding['cls'])
    result['configured_context_exited'] = True
    with (ROOT / 'SESSION_BINDING_RUNTIME_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
