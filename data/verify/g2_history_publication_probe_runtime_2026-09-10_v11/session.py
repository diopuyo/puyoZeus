"""凍結full installerを再利用し、診断scopeだけを原Jへ明示する。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator
import common as K
import assembly_publication_probe as A


@contextmanager
def configured() -> Iterator[Any]:
    names, paths = set(sys.modules), list(sys.path)
    try:
        with installed() as env:
            yield env
    finally:
        sys.path[:] = paths
        for name in set(sys.modules) - names:
            path = getattr(sys.modules[name], '__file__', None)
            if path and Path(path).resolve().is_relative_to(K.PROJECT):
                del sys.modules[name]


@contextmanager
def installed() -> Iterator[Any]:
    K.guards()
    paths = list(sys.path)
    with ExitStack() as stack:
        stack.callback(setattr, sys, 'path', paths)
        sys.path[:0] = [str(K.ROOT), str(K.FULL), str(K.NORMAL), str(K.PROJECT)]
        d = K.load('deps_full', K.FULL / 'deps_full.py', stack)
        a = K.load('_probe_full_assembly', K.FULL / 'assembly_history.py', stack)
        K.require('reporting' not in sys.modules, 'foreign_reporting')
        c = K.load('_probe_normal_assembly', K.NORMAL / 'assembly.py', stack)
        x, modules = d.load('connection', stack), d.history(stack)
        original = (c.install_transform, c.derive)
        with A.session(stack, a, c, modules) as (runtime, addon, factory):
            journal_install = addon.journal.install
            def install(inner: Any, collector: Any, history: Any, state: Any, **kwargs: Any) -> Any:
                K.require(not kwargs, 'unexpected_journal_scope_override')
                return journal_install(inner, collector, history, state,
                    expected_scopes=[(f, s) for f in K.FRAMES for s in K.SIDES])
            c.G.patch(stack, addon.journal, 'install', install)
            old = runtime.extra
            directional_guards = {str(p): h for p, h in x.FIXED.values()}
            c.G.patch(stack, runtime, 'extra', lambda args: old(args) | K.guards() | d.guards() | directional_guards)
            yield dict(a=a, c=c, d=d, x=x, runtime=runtime, addon=addon, factory=factory)
        K.require(original == (c.install_transform, c.derive), 'probe_builder_restore')


def prepare(env: Any, output: Path) -> tuple[Any, Any]:
    runtime = env['runtime']
    latest, base = runtime.M.A, runtime.M.base
    with ExitStack() as stack:
        p = K.load('_probe_original_preflight', K.PREFLIGHT, stack)
        args = p.arguments(runtime, output)
        with runtime.configured(), latest.configured(), latest.prior.configured():
            receipt, config = latest.entry.prepare(latest.entry.legacy_arguments(args), args)
    receipt['input_and_code_sha256'].update(K.guards() | env['d'].guards())
    base.assert_unchanged(receipt['input_and_code_sha256'])
    receipt.update(probe_schema='history-current-publication-prefix/v1', diagnostic_bounds=K.bounds(),
        source_id=K.SOURCE, run_id=output.resolve().as_posix(), requested_interval_sec=[K.FIRST / K.FPS, K.END / K.FPS],
        expected_frame_count=len(K.FRAMES), original_prepare_interval_sec=[484.2, 605.0],
        current_permission=False, quality_gate_clear=False, production_permission=False,
        old_full_finish_called=False, bounds_override_is_explicit=True)
    return receipt, config


@contextmanager
def live_scopes(env: Any) -> Iterator[None]:
    runtime = env['runtime']
    latest = runtime.M.A
    old = (latest.instrument, latest.prior.instrument, latest.entry.instrument)
    try:
        with runtime.configured(), latest.configured(), latest.prior.configured():
            K.require(latest.instrument is runtime.M.instrument and latest.prior.instrument is runtime.M.instrument
                and latest.entry.instrument is runtime.M.instrument, 'original_three_instrument_scopes')
            yield
    finally:
        K.require(old == (latest.instrument, latest.prior.instrument, latest.entry.instrument), 'three_scope_restore')


def prepare_output(env: Any, output: Path, owned: list[Path]) -> tuple[Any, Any]:
    """原run同様、原prepare成功後だけ新出力を排他作成する。"""
    receipt, config = prepare(env, output)
    K.require(not output.exists() and not output.is_symlink(), 'original_prepare_created_output')
    output.mkdir(exist_ok=False)
    owned.append(output)
    receipt['output_boundary'] = dict(path=output.resolve().as_posix(),
        original_prepare_returned_before_mkdir=True, shared_live_preflight=True)
    return receipt, config


def preflight(output: Path, owned: list[Path]) -> dict[str, Any]:
    with configured() as env, ExitStack() as stack:
        p = K.load('_probe_guard_preflight', K.PREFLIGHT, stack)
        calls, hashes = [], {}
        K.require(not any(n == 'src' or n.startswith('src.') for n in sys.modules), 'cold_src_required')
        with p.hash_cache(env['runtime'].M.base, hashes), p.no_recognition(calls):
            receipt, config = prepare_output(env, output, owned)
        K.write(output / 'PREPARE_RECEIPT.json', receipt)
        K.write(output / 'INPUT_HASHES.json', hashes)
        result = dict(bounds=K.bounds(), calls=calls, guard_count=len(receipt['input_and_code_sha256']),
            hash_equal=hashes['equal'], config_tokens_preserved=bool(config['collection_tokens']),
            factory_called=False, video_decode=False, cuda_initialized=False, restored=True,
            output_boundary=receipt['output_boundary'])
    return result
