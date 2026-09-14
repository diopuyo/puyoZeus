"""current全installer完了後にpostcommit hookだけを一度供給する。"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
CURRENT = VERIFY / 'g2_history_current_full_runtime_2026-09-09_v2'
PUBLICATION = VERIFY / 'g2_history_current_publication_2026-09-09_v2'
FIXED = {CURRENT / 'assembly_current.py': 'ca20b57bdd248b845337262bfe2004d1a5d713d91950c6342197e4991e3c7038',
    CURRENT / 'run_cpu.py': '21608c0e1023def836865423813faa48ff5301510e6d1560ccf7860e21cbcde4',
    PUBLICATION / 'ticket.py': 'c4b5e7ae1748764a8f75416d611329660b92a97e5985707bdcfd15e036dbb03c',
    PUBLICATION / 'runtime_hook.py': '0a49a1b286ebbbe5eafb1cbbbea6e1afefdbba68d7ca710e0656572671aea975'}
SIDECAR, STATUS = 'history_current_publication.jsonl', 'HISTORY_CURRENT_PUBLICATION_STATUS.json'
KEY = 'history_current_publication_sink'
OWN = ('assembly_publication.py', 'run_cpu.py', 'PLAN.md')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'publication_fixed_source_changed')
    return {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): sha(ROOT / n) for n in OWN}


def load(stack: Any, alias: str, path: Path, injection: Any = None) -> Any:
    require(alias not in sys.modules, 'publication_private_alias:' + alias)
    original, mapped = builtins.__import__, injection or {}
    def importing(name: str, globals: Any = None, locals: Any = None,
                  fromlist: Any = (), level: int = 0) -> Any:
        return mapped[name] if level == 0 and name in mapped else original(name, globals, locals, fromlist, level)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=importing)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


class Sink:
    def __init__(self, state: Any) -> None:
        self.output = Path(state['output'])
        self.stream = (self.output / SIDECAR).open('x', encoding='utf-8')
        self.rows, self.errors, self.closed = 0, [], False
        self.receiver: Any = None

    def emit(self, value: Any) -> None:
        try:
            self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            self.rows += 1
        except BaseException as error:
            self.errors.append(repr(error))
            raise

    def close(self) -> None:
        self.stream.close()
        self.closed = True
        rec = self.receiver
        value = dict(closed=True, rows=self.rows, errors=self.errors,
            receiver_closed=None if rec is None else rec.closed,
            receiver_errors=None if rec is None else rec.errors,
            active=None if rec is None else rec.active is not None,
            issued=None if rec is None else rec.issued, released=None if rec is None else rec.released,
            old_consumer_comparison_connected=False, quality_gate_clear=False, physical_certified=False)
        with (self.output / STATUS).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def install(stack: Any, collector: Any, state: Any) -> Any:
    guards()
    require(KEY not in state, 'publication_full_installed_twice')
    control, module = state['normal_completion_controller'], sys.modules.get('controller_v3')
    require(module is not None and type(control).prepared is module.prepared
        and type(control).hold_transition is module.hold_transition, 'publication_current_v3_required')
    ticket = load(stack, '_history_publication_v2_ticket', PUBLICATION / 'ticket.py')
    hook = load(stack, '_history_publication_v2_hook', PUBLICATION / 'runtime_hook.py', {'ticket': ticket})
    require(hook.T is ticket and hook.Receiver.complete.__globals__['T'] is ticket,
        'publication_ticket_module_binding')
    sink = Sink(state)
    stack.callback(sink.close)
    sink.receiver = hook.install(stack, collector, state, sink.emit)
    state[KEY] = sink
    return sink.receiver


@contextmanager
def session(a: Any, current: Any, c: Any, modules: Any, loaded: Any) -> Iterator[Any]:
    with current.session(a, c, modules, loaded) as (runtime, addon, factory):
        from contextlib import ExitStack
        with ExitStack() as stack:
            original = runtime.M.instrument
            def instrument(inner: Any, collector: Any, history: Any, receipt: Any, state: Any) -> None:
                original(inner, collector, history, receipt, state)
                install(inner, collector, state)
            c.G.patch(stack, runtime.M, 'instrument', instrument)
            yield runtime, addon, factory
