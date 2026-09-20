"""原76更新の後に4人工不整合観測を追加し、完了した信号だけをresetへ渡す。"""
from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any

import continuation as C
import post_inputs as OLD_INPUT
import run_qualification as Q

ROOT = Path(__file__).resolve().parent
FIRST, STRIDE, FPS, COUNT = 34924, 2, 60, 4
FRAMES = tuple(range(FIRST, FIRST + COUNT * STRIDE, STRIDE))
RESET = FRAMES[-1] + STRIDE
POST = tuple(range(RESET, RESET + len(OLD_INPUT.POST) * STRIDE, STRIDE))
BB, YY = (2, 2), (4, 4)


def inputs() -> Any:
    values = dict(vars(OLD_INPUT), RESET=RESET, SHIFT=RESET-OLD_INPUT.ORIGINAL_RESET, POST=POST)
    values['shifted'] = FunctionType(OLD_INPUT.shifted.__code__, values)
    return N(**values)


def install(context: Any, stack: Any, grid: Any) -> None:
    q, pipe, pixels, clock = (context[k] for k in ('q', 'pipe', 'pixels', 'clock'))
    types = context['factory'].types.parts.O
    helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
    original, call, read = helper.actual_next, type(pixels).__call__, pipe._reader.read_both_boards
    def next_value(pair: Any, other: Any = None) -> Any:
        result = original(pair, other)
        return type(result)(type(result.p1)(*BB, *YY), result.p2)
    def observed(self: Any, invocation: Any, side: str) -> Any:
        result = call(self, invocation, side)
        if self is not pixels or side != '1P': return result
        frame, epoch = invocation.frame, invocation.runtime.histories[side].epoch
        event = types.MotionCandidate(5, frame, frame, 'artificial-desync:5') if frame == FIRST else None
        quiet = (frame-STRIDE, frame) if frame in FRAMES[-2:] else None
        return types.MotionObservation(invocation, side, epoch, f'artificial-hidden-segment:{epoch}',
            frame, invocation.time_sec, quiet, event)
    def boards(image: Any, **kwargs: Any) -> Any:
        first, second = read(image, **kwargs)
        return type(first).from_dict({'grid': [list(row) for row in grid]}), second
    for obj, name, old in ((helper, 'actual_next', original), (type(pixels), '__call__', call),
                           (pipe._reader, 'read_both_boards', read)):
        stack.callback(setattr, obj, name, old)
    helper.actual_next, type(pixels).__call__, pipe._reader.read_both_boards = next_value, observed, boards


def fresh_proof(context: Any) -> Any:
    factory, state = context['factory'], context['state']
    prior = Q.KEPT['prefix_evidence']
    read = lambda name: [json.loads(line) for line in (state['output']/name).read_text().splitlines()]
    module = Q.E.fixed('g2_empty_tail_archive_candidate_2026-09-10_v1/empty_archive.py')
    archive = module.Archive(factory.controller, factory.controller.history['1P'])
    assert state['repeat_scope_guard'].frame == FRAMES[-1]
    proof = Q.E.PrefixEvidence(factory, prior.parts, archive, read('atomic_journal.jsonl'),
        read('directional_history.jsonl'), prior.step, FRAMES[-1])
    Q.KEPT['prefix_evidence'] = proof
    return proof


def continuation() -> Any:
    # 元連続実装の76件だけを実80件へ拡張。他の許可条件は変えない。
    values = dict(vars(C), I=inputs())
    for name in ('verify', 'extend'):
        tree = ast.parse(inspect.getsource(getattr(C, name)))
        hits = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 76:
                node.value += COUNT
                hits += 1
        assert hits == (1 if name == 'verify' else 2), 'original_prefix_count_anchors'
        exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), values)
    return values['extend']


def extend(context: Any, result: Any) -> None:
    factory, state, pipe = (context[k] for k in ('factory', 'state', 'pipe'))
    binding = state['directional_next_runtime']
    observer_module = Q.module('_actual_desync_observer', ROOT/'observer.py')
    observer = observer_module.install(context['stack'], binding['adapter'])
    # 保存rawの最終フレームを人工fixtureへ保持。新しい動画観測と呼ばない。
    prior = Q.KEPT['prefix_evidence']
    row = next(r for r in prior.history if r['scope']['frame_idx'] == FIRST-STRIDE)
    grid = row['raw_capture']['raw']['grid']
    palette = {c for line in grid for c in line if 1 <= c <= 5}
    assert set(BB+YY) <= palette, 'fixture_introduces_foreign_palette'
    Q.KEPT['prefix_input_stack'].close()
    owned = context['stack'].enter_context(ExitStack())
    Q.KEPT['prefix_input_stack'] = owned
    install(context, owned, grid)
    trace, signal, proof = [], None, None
    try:
        for frame in FRAMES:
            context['clock']['frame'], context['cap'].position = frame, frame
            context['collector'].collect_lean(context['cap'], pipe, frame, 1, STRIDE, FPS)
            found = observer.take(pipe, frame)
            trace.append(dict(frame=frame, signal=found is not None, FIFO=list(pipe._pending_tsumo_1p)))
            if found is not None:
                assert signal is None and frame == FRAMES[-1]
                signal = found
        assert signal is not None and not signal.reset_permission and not signal.current_permission
        proof = fresh_proof(context)
        assert proof.frame == signal.facts[-1].frame and not pipe._pending_tsumo_1p
        result['original_76_J_steps'], result['J_steps'] = result['J_steps'], factory.provider.journal.steps
        continuation()(context, result)
    finally:
        row = dict(trace=trace, facts=None if signal is None else [asdict(f) for f in signal.facts],
            fresh_samecall=None if proof is None else proof.result, missing_count='UNCERTIFIED',
            artificial_raw_geometry=True, actual_adapter=True, quality_gate_clear=False)
        with (state['output']/'DESYNC_TRIGGER.json').open('x', encoding='utf-8') as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2)
