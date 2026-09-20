"""実configuredのREPEAT.verify参照鎖を読むだけ。factory/update/終了品質は検証しない。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType
from typing import Any
import live_adapter as A


def chain(function: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    def visit(value: Any, via: str) -> None:
        assert id(value) not in seen, 'outer_verify_cycle'
        seen.add(id(value))
        rows.append(dict(via=via, name=value.__name__, file=value.__code__.co_filename,
                         line=value.__code__.co_firstlineno))
        for name, member in inspect.getclosurevars(value).nonlocals.items():
            if name in ('original_verify', 'checked'):
                assert isinstance(member, FunctionType)
                visit(member, name)
    visit(function, 'REPEAT.verify')
    return rows


def main() -> None:
    previous = list(sys.path)
    with ExitStack() as stack:
        main = A.configured(stack)
        repeated = main.__globals__['REPEAT']
        actual = repeated.verify
        rows = chain(actual)
        expected = ('g2_conditional_live_adapter_2026-09-10_v3/adapter.py',
                    'g2_conditional_live_adapter_2026-09-10_v1/adapter.py',
                    'g2_history_publication_probe_runtime_2026-09-10_v13/repeated_connection.py')
        assert len(rows) == len(expected)
        assert all(Path(row['file']).as_posix().endswith(suffix) for row, suffix in zip(rows, expected))
    assert repeated.verify is not actual and sys.path == previous
    torch = sys.modules.get('torch')
    assert torch is None or not torch.cuda.is_initialized()
    report = dict(chain=rows, paths_restored=True, outer_wrapper_restored=True,
                  factory_calls=0, updates=0, actual_video=False, quality_gate_clear=False)
    with (A.ROOT / 'OUTER_CONFIGURATION_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
