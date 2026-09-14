"""所有する通信子の生成だけを1スレッドへ制限し、元関数を復元する。"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def creator(original: Any) -> Any:
    def build(load: Any, frames: tuple[int, ...]) -> Any:
        create = original(load, frames)
        transport = create.__globals__['CLIENT'].T.T

        def run(stack: ExitStack, context: dict) -> Any:
            evidence = install(stack, transport)
            context['state']['m1_child_thermal_evidence'] = evidence
            return create(stack, context)

        return run

    return build


def install(stack: ExitStack, transport: Any) -> list[dict]:
    original = transport.Client.__init__
    process = original.__globals__['subprocess']
    evidence: list[dict] = []

    def launch(*args: Any, **kwargs: Any) -> Any:
        environment = dict(kwargs['env'])
        environment.update({key: '1' for key in THREAD_KEYS})
        kwargs['env'] = environment
        child = process.Popen(*args, **kwargs)
        try:
            actual = dict(v.split('=', 1) for v in Path(f'/proc/{child.pid}/environ').read_text().split('\0') if '=' in v)
            assert all(actual[key] == '1' for key in THREAD_KEYS), 'child_thread_environment'
        except BaseException:
            child.kill()
            child.wait()
            raise
        evidence.append(dict(pid=child.pid, threads={key: actual[key] for key in THREAD_KEYS}))
        return child

    proxy = SimpleNamespace(**vars(process))
    proxy.Popen = launch
    selected = FunctionType(original.__code__, dict(original.__globals__, subprocess=proxy),
                            original.__name__, original.__defaults__, original.__closure__)
    selected.__kwdefaults__ = original.__kwdefaults__
    transport.Client.__init__ = selected

    def restore() -> None:
        assert transport.Client.__init__ is selected, 'thermal_child_owner'
        transport.Client.__init__ = original

    stack.callback(restore)
    return evidence
