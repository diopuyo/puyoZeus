"""最新palette構成を保持し、FIFO色結合と導出済みjournal期待codeだけを合成する。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import codegen as G

ROOT = Path(__file__).resolve().parent
LATEST = ROOT.parent / 'g2_palette_finish_runtime_2026-09-09_v1/live_cli.py'
FIXED = {LATEST: 'c083ed152453d28166bdf6ea3ab2436a25d1d3a47a65f058a851b0ff3abe227f',
    ROOT.parent / 'g2_atomic_journal_runtime_2026-09-09_v1/live_cli.py': '3453b2f5cc062617b384b744185246b1762ba365a9362280e1020be7f1e0db8d',
    ROOT.parent / 'g2_t2_bounded_runtime_2026-09-09_v1/live_cli.py': 'e73fba08e960c9af37e684e84bc85e395f00592791056a6eed9ad1505ed21c1c',
    ROOT.parent / 'g2_palette_veto_runtime_2026-09-09_v1/live_cli.py': '6c5c5fa6784e39036b461e6b82bf328137428067415854c5d0e1861f472f4d2f',
    ROOT.parent / 'g2_palette_finish_proof_2026-09-09_v1/adapter.py': 'd963d4b832b8751e62cd295f726bb242ac1514396e4dcf01a186f24cc36d0ec9'}
OWN = ('adapter.py', 'codegen.py', 'test_composition.py', 'run_cpu.py', 'CONTRACT.md')
RECEIPT = 'FIFO_PALETTE_COMPOSITION.json'
REQUIRED = frozenset((RECEIPT,))


def guards() -> dict[str, str]:
    G.require(all(G.sha(p) == h for p, h in FIXED.items()), 'composition_cli_source_changed')
    return G.guards() | {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): G.sha(ROOT / n) for n in OWN}


@contextmanager
def loaded_latest() -> Any:
    """coldロードの新aliasと探索pathはscope終了時に戻す。srcは先行importしない。"""
    guards()
    names, paths = set(sys.modules), list(sys.path)
    name = '_fifo_palette_latest_cli'
    G.require(name not in sys.modules, 'composition_alias_collision')
    try:
        spec = importlib.util.spec_from_file_location(name, LATEST)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        sys.path.insert(0, str(LATEST.parent))
        spec.loader.exec_module(value)
        yield value
    finally:
        sys.path[:] = paths
        for key in set(sys.modules) - names:
            module = sys.modules[key]
            path = getattr(module, '__file__', None)
            if path and Path(path).resolve().is_relative_to(G.PROJECT):
                del sys.modules[key]


def journal_view(derivation: dict[str, Any]) -> Any:
    validate_derivation(derivation)
    value, grace = G.module(G.JOURNAL), G.module(G.GRACE)
    expected = {row['bytecode_sha256']: row for name, row in derivation['on'].items()
                if name in ('pending', 'resolved')}
    G.require(len(expected) == 2, 'composition_expected_two')
    value.CODE_HASHES = frozenset(expected)
    original = value.install
    @functools.wraps(original)
    def install(stack: Any, collector: Any, history: Any, state: Any, **kwargs: Any) -> Any:
        codes = {history.step_code, state['pending_code']}
        actual = {hashlib.sha256(c.co_code).hexdigest(): G.code_record(grace, c) for c in codes}
        G.require(actual == expected, 'composition_unknown_full_generated_code')
        return original(stack, collector, history, state, **kwargs)
    value.install = install
    return value


def validate_derivation(derivation: dict[str, Any]) -> None:
    """外側dict改変を新しい許可codeの自己申告として受け付けない。"""
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    G.require(encode(derivation) == encode(G.derive()), 'composition_unbound_derivation')


def compose(latest: Any, *, enabled: bool = False) -> tuple[Any, Any, Any, Any]:
    G.require(type(enabled) is bool, 'composition_enabled_type')
    if not enabled:
        return latest.bootstrap()
    guards()
    derivation = G.derive()
    atomic = latest.B.B
    G.require(Path(atomic.__file__).resolve() in FIXED, 'composition_atomic_cli')
    journal = journal_view(derivation)
    with ExitStack() as stack:
        G.patch(stack, atomic, 'observer', lambda: journal)
        prior, driver, old, engine = latest.bootstrap()
    def finish(state: Any) -> None:
        validate_derivation(derivation)
        path = Path(state['output']) / RECEIPT
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'derivation': derivation, 'guards': guards(), 'physical_certified': False,
                       'quality_gate_clear': False}, stream, sort_keys=True, allow_nan=False)
        old.finish(state)
    def verify(output: Path) -> None:
        validate_derivation(derivation)
        saved = json.loads((output / RECEIPT).read_text())
        G.require(saved == {'derivation': derivation, 'guards': guards(), 'physical_certified': False,
                           'quality_gate_clear': False}, 'composition_receipt_changed')
        old.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | REQUIRED, install=old.install,
        finish=finish, verify=verify, bind=old.bind, proof=old.proof,
        guards=lambda: old.guards() | guards(), derivation=derivation, journal=journal)
    return prior, driver, addon, engine


@contextmanager
def configured(latest: Any, runtime: Any, finisher: Any, driver: Any,
               addon: Any, engine: Any, *, enabled: bool = False) -> Any:
    G.require(type(enabled) is bool, 'composition_enabled_type')
    with latest.configured(runtime, finisher, driver, addon, engine):
        if not enabled:
            yield
            return
        G.require(hasattr(addon, 'derivation'), 'composition_enabled_addon_required')
        validate_derivation(addon.derivation)
        t2, fifo = latest.B.B.B.T, G.module(G.FIFO)
        old_install, old_load = t2.install, addon.proof.load
        def install(stack: Any, target: Any, *, enabled: bool = False) -> None:
            G.require(target is runtime.M and enabled is True, 'composition_t2_install_scope')
            old_install(stack, target, enabled=True)
            fifo.install(stack, target, t2, enabled=True)
        def load(path: Path, suffix: str) -> Any:
            value = old_load(path, suffix)
            return journal_view(addon.derivation) if path.resolve() == G.JOURNAL.resolve() else value
        with ExitStack() as stack:
            G.patch(stack, t2, 'install', install)
            G.patch(stack, addon.proof, 'load', load)
            yield
