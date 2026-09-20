"""人工NEXT時計・原SM/consume/infer経路のCPU境界。実動画再生ではない。"""
from __future__ import annotations

import ast
from collections import Counter, deque
import copy
import importlib.util
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any
import pytest
import transaction as T

PROJECT = T.ROOT.parents[2]
SNAPSHOT = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
SOURCE = SNAPSHOT / 'src/recognition_pipeline.py'
SOURCE_SHA = '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02'
OLD, NEXT, DNEXT, FOLLOW = (4, 3), (2, 5), (5, 5), (5, 4)
FPS, STEP = 60, 2
sys.path.insert(0, str(SNAPSHOT))
from src.board import Board
from src.board_state_machine import BoardState, BoardStateMachine, DetectorSignals
from src.placement_inferrer import infer_placement


def private(alias: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


A = private('_completion_prior_fifo_binding', T.ROOT.parent / 'g2_landing_fifo_pair_binding_2026-09-09_v1/adapter.py')
D = private('_completion_original_legal', T.ROOT.parent / 'g2_inventory_producer_diagnosis_2026-09-09_v1/probe.py')


class PlannedDetector:
    """状態提案だけ人工。原SM内から一回呼び出す。"""
    def __init__(self) -> None:
        self.target = BoardState.STABLE

    def detect(self, ctx: Any, signals: Any) -> Any:
        return self.target


class FixtureProvider:
    """live権限を持たない人工ref/token provider。原dequeは一つだけ。"""
    def __init__(self, pipe: Any) -> None:
        self.pipe, self.next, self.dnext = pipe, NEXT, DNEXT
        self.tokens, self.added = ['artificial:old:slot:0'], ()
        self.scope = ('artificial_source', 'artificial_run', 2, id(pipe), '1P')
        self.quiet = True

    def selected(self, pipe: Any, side: str, frame: int, clock: float) -> bool:
        return True

    def view(self, pipe: Any, side: str, frame: int, clock: float, caller: Any) -> T.View:
        queue = pipe._pending_tsumo_1p
        return T.View(self.scope, frame, clock, queue, tuple(queue), tuple(self.tokens),
            self.next, self.dnext, self.quiet, self.added)

    def append_new(self) -> None:
        self.pipe._pending_tsumo_1p.append(NEXT)
        self.pipe._last_consumed_color_1p = NEXT
        self.tokens.append('artificial:new:slot:1')
        self.added = (self.tokens[-1],)
        self.next, self.dnext = DNEXT, FOLLOW

    def before_consume(self, ticket: Any, caller: Any) -> None:
        assert ticket['item'].pair is self.pipe._pending_tsumo_1p[0]

    def after_consume(self, ticket: Any, caller: Any) -> None:
        assert self.tokens.pop(0) == ticket['item'].token

    def after_step(self, ticket: Any, caller: Any) -> None:
        assert self.tokens == list(ticket['view'].tokens[1:])


def source_blocks() -> tuple[Any, Any, Any, Any]:
    assert A.sha(SOURCE) == SOURCE_SHA
    tree = ast.parse(SOURCE.read_bytes())
    step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_step_side')
    conditions = [n for n in step.body if isinstance(n, ast.If)]
    base = 'prev_state == BoardState.TSUMO_FALL and ctx.state == BoardState.STABLE'
    consume = next(n for n in conditions if ast.unparse(n.test) == base)
    infer_text = ast.unparse(ast.parse(base
        + ' and prev_confirmed is not None and not _skip_infer_by_ojama_guard', mode='eval').body)
    infer = next(n for n in conditions if ast.unparse(n.test) == infer_text)
    infer.body = [n for n in infer.body if n.lineno <= 7492]
    assignments = [n for n in step.body if isinstance(n, ast.Assign)]
    warm = [n for n in assignments if ast.unparse(n.targets[0]) in
        ('_ojama_warmup_remaining', '_skip_infer_by_ojama_guard')]
    return consume, infer, warm, step


def driver(controller: Any) -> Any:
    consume, infer, warm, original = source_blocks()
    tree = ast.parse('''def _step_side(self, side, frame_idx, time_sec, sm, signals):
    cnn_board = signals.cnn_board
    prev_state = sm.context.state
    prev_confirmed = sm.context.confirmed_board.copy()
    prev_next_queue = list(sm.context.next_queue)
    frame_bgr = None
    try:
        ctx: StateContext = sm.update(frame_idx, signals)
    finally:
        __normal_completion.finish(sys._getframe(), sys.exc_info()[1])
    return ctx
''')
    trial = tree.body[0].body[-2]
    trial.body.extend([consume, *warm, infer])
    # 色結合の一意grace入口はAST形を保持し、この縮小driverでは到達させない。
    trial.body.extend(ast.parse('if False:\n    _, falling_pair_for_grace = landing_pending').body)
    T.add_transaction(A.add_binding(tree))
    ast.fix_missing_locations(tree)
    namespace = {'__normal_completion': controller, 'sys': sys, 'BoardState': BoardState,
        'StateContext': Any, 'infer_placement': infer_placement,
        'DEFAULT_P1_REGION': None, 'DEFAULT_P2_REGION': None}
    exec(compile(tree, str(T.ROOT / 'artificial_driver.py'), 'exec'), namespace)
    return namespace['_step_side']


def setup() -> tuple[Any, ...]:
    detector = PlannedDetector()
    sm = BoardStateMachine(detectors=[detector])
    sm.context.state, sm.context.confirmed_board = BoardState.TSUMO_FALL, Board()
    sm.context.next_queue = [OLD, NEXT]
    pipe = SimpleNamespace(_pending_tsumo_1p=deque([OLD]), _tsumo_count_1p=Counter(),
        _first_move_sec_1p=None, _enable_ojama_infer_guard=True, _ojama_tier1_warmup_remaining_1p=0,
        _last_consumed_color_1p=OLD, _enable_landing_color_fix=True, _reader=SimpleNamespace(),
        _chain_sim=None, _enable_infer_empty_guard=False, _enable_hsv_classify_fallback=False,
        _enable_hsv_deferred_consensus=False)
    raw = Board()
    raw.set(12, 0, OLD[0])
    raw.set(11, 0, OLD[1])
    provider = FixtureProvider(pipe)
    controller = T.Controller(provider, D.placement_matches, enabled=True)
    return pipe, sm, raw, provider, controller, driver(controller), detector


def invoke(parts: tuple[Any, ...], frame: int, raw: Any = None) -> Any:
    pipe, sm, original, provider, controller, step, detector = parts
    signals = DetectorSignals(cnn_board=raw or original, time_sec=frame / FPS,
        is_match_active=True, next_pair=provider.next, placement_validated=True)
    return step(pipe, '1P', frame, frame / FPS, sm, signals)


def prime(parts: tuple[Any, ...]) -> None:
    for frame in (2, 4, 6):
        result = invoke(parts, frame)
        assert result.state == BoardState.TSUMO_FALL
        assert result.confirmed_board.count_puyos() == 0
        assert parts[0]._tsumo_count_1p == Counter()


def test_finite_completion_through_ojama() -> None:
    parts = setup()
    pipe, sm, raw, provider, controller, step, detector = parts
    prime(parts)
    detector.target = BoardState.OJAMA_FALL
    assert invoke(parts, 8).state == BoardState.OJAMA_FALL
    detector.target = BoardState.STABLE
    assert invoke(parts, 10).state == BoardState.OJAMA_FALL
    assert invoke(parts, 12).state == BoardState.OJAMA_FALL
    assert invoke(parts, 14).state == BoardState.OJAMA_FALL
    provider.append_new()
    result = invoke(parts, 16)
    assert result.state == BoardState.STABLE and result.confirmed_board == raw, controller.records[-1]
    assert pipe._tsumo_count_1p == Counter(OLD) and tuple(pipe._pending_tsumo_1p) == (NEXT,)
    assert controller.records[-1]['execution_complete'] and not controller.tickets
    provider.added = ()
    invoke(parts, 18)
    assert pipe._tsumo_count_1p == Counter(OLD) and tuple(pipe._pending_tsumo_1p) == (NEXT,)
    assert not controller.pending and controller.sticky_error is None


@pytest.mark.parametrize('mutation', ['no_next', 'warmup', 'new_piece', 'unknown', 'changed', 'same_next', 'unquiet'])
def test_incomplete_stays_nonstable_without_consumption(mutation: str) -> None:
    parts = setup()
    pipe, sm, raw, provider, controller, step, detector = parts
    prime(parts)
    provider.append_new()
    if mutation == 'no_next':
        provider.added = ()
    elif mutation == 'warmup':
        pipe._ojama_tier1_warmup_remaining_1p = 1
    elif mutation in ('new_piece', 'unknown', 'changed'):
        raw = raw.copy()
        if mutation == 'new_piece':
            raw.set(12, 1, 2)
            raw.set(11, 1, 5)
        elif mutation == 'unknown':
            raw.set(0, 1, 10)
        else:
            raw.set(11, 0, 0)
            raw.set(12, 1, OLD[1])
    elif mutation == 'same_next':
        provider.next, provider.dnext = NEXT, DNEXT
    else:
        provider.quiet = False
    result = invoke(parts, 8, raw)
    assert result.state == BoardState.TSUMO_FALL and result.confirmed_board.count_puyos() == 0
    assert pipe._tsumo_count_1p == Counter() and tuple(pipe._pending_tsumo_1p) == (OLD, NEXT)
    assert controller.records[-1]['transition_calls'] == 0 and not controller.tickets


def test_off_and_unrelated_transition_keep_original() -> None:
    parts = setup()
    parts[4].enabled = False
    assert invoke(parts, 2).state == BoardState.STABLE
    assert parts[0]._tsumo_count_1p == Counter(OLD)
    assert not parts[4].records
    parts = setup()
    prime(parts)
    parts[-1].target = BoardState.CHAIN
    assert invoke(parts, 8).state == BoardState.CHAIN


def test_initial_unknown_arms_hold_not_completion() -> None:
    parts = setup()
    unknown = parts[2].copy()
    unknown.set(0, 1, 10)
    result = invoke(parts, 2, unknown)
    assert result.state == BoardState.TSUMO_FALL and result.confirmed_board.count_puyos() == 0
    assert parts[4].pending['1P'].previous is None
    assert parts[4].records[-1]['reason'] == 'raw_not_exact_old_pair'
    assert parts[0]._tsumo_count_1p == Counter()


def test_original_merge_must_also_confirm_candidate() -> None:
    parts = setup()
    prime(parts)
    parts[-1].target = BoardState.OJAMA_FALL
    invoke(parts, 8)
    parts[-1].target = BoardState.STABLE
    invoke(parts, 10)
    parts[3].append_new()
    result = invoke(parts, 12)
    assert result.state == BoardState.OJAMA_FALL
    assert parts[4].records[-1]['reason'] == 'original_merge_mismatch'
    assert parts[0]._tsumo_count_1p == Counter()


def test_same_filename_line_modified_transition_rejected(monkeypatch: Any) -> None:
    parts = setup()
    original = BoardStateMachine._apply_transition
    code = original.__code__
    altered = code.replace(co_consts=tuple('modified_doc' if i == 0 else c for i, c in enumerate(code.co_consts)))
    changed = FunctionType(altered, original.__globals__, original.__name__, original.__defaults__)
    monkeypatch.setattr(BoardStateMachine, '_apply_transition', changed)
    with pytest.raises(RuntimeError, match='unknown_transition_body'):
        invoke(parts, 2)
    assert BoardStateMachine._apply_transition is changed


def test_exception_identity_and_restore(monkeypatch: Any) -> None:
    parts = setup()
    original = BoardStateMachine._apply_transition
    failure = RuntimeError('original_within_failure')
    def failed(self: Any, signals: Any) -> None:
        raise failure
    monkeypatch.setattr(BoardStateMachine, '_update_within_current_state', failed)
    with pytest.raises(RuntimeError) as caught:
        invoke(parts, 2)
    assert caught.value is failure and BoardStateMachine._apply_transition is original
    assert parts[4].sticky_error == repr(failure) and not parts[4].tickets


def test_ast_is_unique_and_original_consume_body_unchanged() -> None:
    consume, _, _, _ = source_blocks()
    old = ast.dump(consume)
    tree = ast.Module(body=[copy.deepcopy(consume)], type_ignores=[])
    with pytest.raises(RuntimeError, match='unique_transaction_sites'):
        T.add_transaction(tree)
    assert ast.dump(consume) == old


def test_default_off_needs_no_provider() -> None:
    with pytest.raises(RuntimeError, match='enabled_type'):
        T.Controller(None, None, enabled=1)
    assert not T.Controller(None, None).enabled
