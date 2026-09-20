"""通常候補のSM更新前保留と、原FIFO/infer入口の同call結合。私有・既定OFF。"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import CodeType
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
PREVIEW = ROOT.parent / 'g2_chigiri_completion_shadow_2026-09-08_v1/shadow.py'
PREVIEW_SHA = '0ed36d8026dd9ccae71c1c5e44a5c1705930bd9036e61e246ec7dff10cdc39e8'
COLORS, PAIR_SIZE = frozenset(range(1, 6)), 2
ACTIVE_STATES = frozenset(('tsumo_fall', 'ojama_fall'))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def load_preview() -> Any:
    require(hashlib.sha256(PREVIEW.read_bytes()).hexdigest() == PREVIEW_SHA, 'preview_source_changed')
    alias = '_normal_completion_existing_preview'
    require(alias not in sys.modules, 'preview_alias_collision')
    spec = importlib.util.spec_from_file_location(alias, PREVIEW)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


P = load_preview()


@lru_cache(maxsize=2)
def original_method(path: str, name: str) -> CodeType:
    source = Path(path)
    require(hashlib.sha256(source.read_bytes()).hexdigest() == P.SM_SHA, 'update_source_changed')
    module = compile(source.read_bytes(), path, 'exec', dont_inherit=True)
    cls = next(c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == 'BoardStateMachine')
    code = next(c for c in cls.co_consts if isinstance(c, CodeType) and c.co_name == name)
    return code


def original_code(actual: CodeType, name: str) -> bool:
    expected = original_method(actual.co_filename, name)
    return (actual == expected and actual.co_filename == expected.co_filename
        and actual.co_linetable == expected.co_linetable
        and repr(actual.co_consts) == repr(expected.co_consts))


def board_key(board: Any) -> Any:
    value = P.grid(board)
    return None if value is None else tuple(tuple(row) for row in value)


def valid_pair(value: Any) -> bool:
    return type(value) is tuple and len(value) == PAIR_SIZE and all(
        type(color) is int and color in COLORS for color in value)


@dataclass(frozen=True)
class View:
    """原観測への参照。新しいFIFO/Counter ownerでも物理着手証明でもない。"""
    scope: tuple[Any, ...]
    frame: int
    clock: float
    queue: Any
    refs: tuple[Any, ...]
    tokens: tuple[str | None, ...]
    next_pair: tuple[int, int] | None
    dnext_pair: tuple[int, int] | None
    quiet: bool
    added: tuple[str, ...]


@dataclass
class Candidate:
    scope: tuple[Any, ...]
    queue: Any
    token: str
    pair: Any
    baseline: Any
    next_pair: Any
    dnext_pair: Any
    started: int
    previous: Any = None
    previous_frame: int | None = None


class Controller:
    """実providerが束縛した一updateだけを許可する。未接続なら動作しない。"""
    def __init__(self, provider: Any, legal: Any, *, enabled: bool = False) -> None:
        require(type(enabled) is bool, 'enabled_type')
        self.provider, self.legal, self.enabled = provider, legal, enabled
        self.pending: dict[str, Candidate] = {}
        self.tickets: dict[int, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.sticky_error: str | None = None

    def exact(self, sm: Any, baseline: Any, raw: Any, pair: Any) -> bool:
        module = sys.modules[type(sm).__module__]
        evidence = P.exact_addition(baseline, raw, module._apply_gravity_filter)
        if not evidence['ok'] or not valid_pair(pair):
            return False
        if Counter(c[2] for c in evidence['cells']) != Counter(pair):
            return False
        return bool(self.legal(board_key(baseline), board_key(raw), list(pair)))

    def candidate(self, sm: Any, signals: Any, view: View) -> Candidate | None:
        side, old = view.scope[-1], self.pending.get(view.scope[-1])
        if old is not None and old.scope != view.scope:
            self.records.append({'frame': view.frame, 'side': side, 'reason': 'scope_invalidated'})
            self.pending.pop(side)
            old = None
        if old is not None:
            require(old.queue is view.queue and bool(view.refs) and view.refs[0] is old.pair
                and view.tokens[0] == old.token, 'pending_slot_lost')
            require(board_key(sm.context.confirmed_board) == board_key(old.baseline), 'baseline_changed')
            return old
        if (sm.context.state.value != 'tsumo_fall' or not view.refs or not view.tokens[0]
            or not view.quiet or not valid_pair(view.next_pair) or not valid_pair(view.dnext_pair)):
            return None
        baseline = sm.context.confirmed_board
        grid = board_key(baseline)
        if grid is None or not valid_pair(view.refs[0]) or any(
            color not in COLORS | {0, 9} for row in grid for color in row):
            return None
        old = Candidate(view.scope, view.queue, view.tokens[0], view.refs[0], baseline.copy(),
            view.next_pair, view.dnext_pair, view.frame)
        self.pending[side] = old
        return old

    def advance(self, item: Candidate, view: View) -> bool:
        if not view.quiet or not all(valid_pair(v) for v in (view.next_pair, view.dnext_pair)):
            return False
        if item.next_pair == view.next_pair or item.dnext_pair != view.next_pair:
            return False
        if len(view.added) != 1 or len(view.refs) != PAIR_SIZE or len(view.tokens) != PAIR_SIZE:
            return False
        return (view.tokens[-1] == view.added[0] and view.tokens[-1] != item.token
            and view.refs[-1] == item.next_pair and item.started < view.frame)

    def eligibility(self, pipe: Any, side: str, sm: Any, signals: Any,
                    item: Candidate, view: View) -> tuple[bool, str]:
        exact = self.exact(sm, item.baseline, signals.cnn_board, item.pair)
        previous = item.previous is not None and item.previous == board_key(signals.cnn_board)
        old_frame = item.previous_frame
        item.previous, item.previous_frame = (board_key(signals.cnn_board), view.frame) if exact else (None, None)
        if not exact:
            return False, 'raw_not_exact_old_pair'
        if not previous or old_frame is None or old_frame >= view.frame:
            return False, 'candidate_changed_or_first'
        if not self.advance(item, view):
            return False, 'completion_advance_unavailable'
        warmup = getattr(pipe, '_ojama_tier1_warmup_remaining_' + side.lower())
        if pipe._enable_ojama_infer_guard and warmup > 0:
            return False, 'original_infer_warmup'
        return True, 'prepared_observational_completion'

    def update(self, sm: Any, frame: int, signals: Any, pipe: Any, side: str) -> Any:
        caller = sys._getframe(1)
        if not self.enabled or not self.provider.selected(pipe, side, frame, signals.time_sec):
            return sm.update(frame, signals)
        try:
            view = self.provider.view(pipe, side, frame, signals.time_sec, caller)
            require(side == view.scope[-1] and view.frame == frame, 'view_side_frame')
            require(id(caller) not in self.tickets, 'duplicate_update')
            item = self.candidate(sm, signals, view)
            if item is None:
                return sm.update(frame, signals)
            if not signals.is_match_active or sm.context.state.value not in ACTIVE_STATES:
                self.pending.pop(side)
                self.records.append({'frame': frame, 'side': side, 'reason': 'unsupported_state_invalidated'})
                return sm.update(frame, signals)
            ready, reason = self.eligibility(pipe, side, sm, signals, item, view)
            with self.transition(sm, signals, pipe, side, caller, item, view, ready, reason):
                result = sm.update(frame, signals)
            return result
        except BaseException as exc:
            self.sticky_error = repr(exc)
            raise

    @contextmanager
    def transition(self, sm: Any, signals: Any, pipe: Any, side: str, caller: Any,
                   item: Candidate, view: View, ready: bool, reason: str) -> Iterator[None]:
        cls, original = type(sm), type(sm)._apply_transition
        require(hashlib.sha256(Path(sys.modules[cls.__module__].__file__).read_bytes()).hexdigest()
            == P.SM_SHA, 'sm_source_changed')
        require(original.__code__.co_firstlineno == P.TRANSITION_FIRST_LINE
            and Path(original.__code__.co_filename).resolve() == Path(sys.modules[cls.__module__].__file__).resolve(),
            'transition_not_original_in_selected_scope')
        require(original_code(original.__code__, '_apply_transition'), 'unknown_transition_body')
        preview = P.Controller(cls, original)

        def wrapped(current: Any, target: Any, observed: Any) -> None:
            parent = sys._getframe(1)
            require(current is sm and observed is signals and original_code(parent.f_code, 'update'), 'actual_sm_caller')
            require(parent.f_locals.get('self') is sm and parent.f_locals.get('new_state') is target, 'actual_transition')
            if target.value != 'stable':
                return original(current, target, observed)
            self.commit_or_hold(preview, sm, target, signals, pipe, side, caller, item, view, ready, reason)
        cls._apply_transition = wrapped
        try:
            yield
        finally:
            cls._apply_transition = original

    def commit_or_hold(self, preview: Any, sm: Any, target: Any, signals: Any,
                       pipe: Any, side: str, caller: Any, item: Candidate,
                       view: View, ready: bool, reason: str) -> None:
        row = {'frame': view.frame, 'side': side, 'reason': reason, 'transition_calls': 0,
            'within_calls': 0, 'preview_calls': 0, 'physical_certified': False, 'quality_gate_clear': False}
        clone = None
        if ready:
            clone, receipt = preview.preview(sm, target, signals)
            row['preview_calls'] = 1
            row['preview_merge'] = receipt['merge_check']
            ready = receipt['merge_check']['ok'] and board_key(clone.context.confirmed_board) == item.previous
            row['reason'] = 'completion_prepared' if ready else 'original_merge_mismatch'
        if ready:
            counter = Counter(getattr(pipe, '_tsumo_count_' + side.lower()))
            preview.original(sm, target, signals)
            require(P.scope_snapshot(sm) == P.scope_snapshot(clone), 'preview_actual_diverged')
            self.tickets[id(caller)] = {'frame': caller, 'item': item, 'view': view, 'counter': counter,
                'stage': 0, 'infer_calls': 0, 'infer_returned': False, 'row': row}
            row['transition_calls'] = 1
        else:
            preview.within(sm, signals)
            row['within_calls'] = 1
            require(sm.context.state.value in ACTIVE_STATES
                and board_key(sm.context.confirmed_board) == board_key(item.baseline), 'hold_mutated_confirmed')
        self.records.append(row)

    def gate(self, stage: str, original: bool) -> bool:
        caller = sys._getframe(1)
        ticket = self.tickets.get(id(caller))
        if ticket is None:
            return original
        require(ticket['frame'] is caller and stage in ('consume', 'infer'), 'ticket_caller')
        expected = 0 if stage == 'consume' else 1
        require(ticket['stage'] == expected, 'duplicate_or_unordered_gate')
        if stage == 'consume':
            self.provider.before_consume(ticket, caller)
        else:
            self.consumed(ticket, caller)
        ticket['stage'] += 1
        return True

    def consumed(self, ticket: dict[str, Any], caller: Any) -> None:
        item, view, values = ticket['item'], ticket['view'], caller.f_locals
        require(values.get('committed') is item.pair, 'committed_reference_mismatch')
        require(len(item.queue) + 1 == len(view.refs)
            and all(a is b for a, b in zip(item.queue, view.refs[1:])), 'remaining_fifo_changed')
        actual = Counter(getattr(values['self'], '_tsumo_count_' + values['side'].lower()))
        require(actual == ticket['counter'] + Counter(item.pair), 'counter_not_exact_pair')
        require(values.get('prev_confirmed') is not None and values.get('_skip_infer_by_ojama_guard') is False,
            'original_infer_precondition_changed')
        self.provider.after_consume(ticket, caller)

    def infer(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        caller = sys._getframe(1)
        ticket = self.tickets.get(id(caller))
        if ticket is None:
            return original(*args, **kwargs)
        require(ticket['stage'] == 2 and ticket['infer_calls'] == 0, 'infer_order')
        require(len(args) >= 3 and args[2] is ticket['item'].pair, 'infer_pair_not_committed')
        require(board_key(args[0]) == board_key(ticket['item'].baseline)
            and board_key(args[1]) == ticket['item'].previous, 'infer_board_scope')
        ticket['infer_calls'] += 1
        result = original(*args, **kwargs)
        ticket['infer_returned'] = True
        ticket['row']['infer_nonempty'] = result is not None
        require(board_key(result) == ticket['item'].previous, 'infer_result_not_prepared_candidate')
        return result

    def finish(self, caller: Any, error: BaseException | None = None) -> None:
        ticket = self.tickets.pop(id(caller), None)
        if ticket is None:
            return
        try:
            if error is not None:
                ticket['row']['error'] = repr(error)
                self.sticky_error = repr(error)
                return
            require(ticket['stage'] == 2 and ticket['infer_calls'] == 1
                and ticket['infer_returned'], 'completion_execution_incomplete')
            self.provider.after_step(ticket, caller)
            self.pending.pop(ticket['view'].scope[-1])
            ticket['row'].update(execution_complete=True, no_rollback_claim=True)
        except BaseException as exc:
            self.sticky_error = repr(exc)
            raise
        finally:
            ticket['frame'] = None


class Transform(ast.NodeTransformer):
    """原updateと元2条件、最初のinfer呼出だけを置換する。"""
    def __init__(self) -> None:
        self.hits = Counter()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if ast.unparse(node.target) == 'ctx' and ast.unparse(node.value) == 'sm.update(frame_idx, signals)':
            self.hits['update'] += 1
            node.value = ast.copy_location(ast.parse(
                '__normal_completion.update(sm, frame_idx, signals, self, side)', mode='eval').body, node.value)
        return node

    def visit_If(self, node: ast.If) -> Any:
        source = ast.unparse(node.test)
        base = 'prev_state == BoardState.TSUMO_FALL and ctx.state == BoardState.STABLE'
        targets = {ast.unparse(ast.parse(text, mode='eval').body): stage for text, stage in
            ((base, 'consume'), (base + ' and prev_confirmed is not None and not _skip_infer_by_ojama_guard', 'infer'))}
        if source in targets:
            stage = targets[source]
            self.hits[stage] += 1
            node.test = ast.copy_location(ast.Call(ast.Attribute(ast.Name('__normal_completion', ast.Load()),
                'gate', ast.Load()), [ast.Constant(stage), node.test], []), node.test)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        if [ast.unparse(n) for n in node.targets] == ['inferred_landing'] and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == 'infer_placement':
                self.hits['infer_call'] += 1
                node.value.args.insert(0, node.value.func)
                node.value.func = ast.copy_location(ast.parse('__normal_completion.infer', mode='eval').body, node.value.func)
        return node


def add_transaction(tree: ast.Module) -> ast.Module:
    change = Transform()
    tree = change.visit(tree)
    require(change.hits == Counter(update=1, consume=1, infer=1, infer_call=1), 'unique_transaction_sites')
    return tree
