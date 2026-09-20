"""原writerの同call因果だけを逐次保存する。会計/公開の認証は行わない。"""
from __future__ import annotations
import ast
from collections import Counter, deque
import dataclasses
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SNAPSHOT = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
SOURCE = SNAPSHOT / 'src/recognition_pipeline.py'
NEXT = PROJECT / 'scripts/next_enqueue_live_shadow_v1.py'
WITNESS = ROOT.parent / 'g2_hsv_correction_witness_2026-09-08_v1/observer.py'
FIXED = {SOURCE: '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02',
    NEXT: 'e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237',
    WITNESS: '7b3407ccbc46bd5ad48fdcc1fc15810ab3dce2c39866ab42b0d5da3d1084642d'}
CODE_HASHES = frozenset(('3083fa45586687a218f8dcf72f799e194fa9356dd7a624767f1ab4c4341e2d09',
    'b7e3c9457d647f34fd84b7acad2835f8a665a7f8108bf40832700307bd351aa4'))
OWN = ('observer.py', 'test_observer.py', 'run_cpu.py', 'CONTRACT.md')
SIDECAR, STATUS, RECEIPT = 'atomic_journal.jsonl', 'ATOMIC_JOURNAL_STATUS.json', 'ATOMIC_JOURNAL_RECEIPT.json'
REQUIRED = frozenset((SIDECAR, STATUS, RECEIPT))
SIDES, FIRST, LAST, STRIDE = ('1P', '2P'), 29052, 36298, 2
KEY, SCHEMA = 'atomic_journal_observer', 'atomic-mutation-journal-observation/v1'
PERMISSIONS = ('quality_gate_clear', 'accounting_permission', 'physical_identity_certified')
COMMON_STAGES = frozenset(('persistence', 'glow', 'answer_check', 'publication'))


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded(value))


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'journal_fixed_source')
    return {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): sha(ROOT / n) for n in OWN}


def board(value: Any) -> Any:
    if value is None:
        return None
    cells = value._grid.tolist()
    require(len(cells) == 13 and all(len(r) == 6 for r in cells), 'journal_grid_shape')
    require(all(type(c) is int for r in cells for c in r), 'journal_grid_type')
    return {'grid': cells, 'object_id': id(value), 'sha256': hashlib.sha256(encoded(cells).encode()).hexdigest()}


def scalar(value: Any) -> Any:
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        require(math.isfinite(value), 'journal_nonfinite')
        return value
    if type(value) in (list, tuple, deque):
        return [scalar(v) for v in value]
    if type(value) in (set, frozenset):
        return [scalar(v) for v in sorted(value)]
    if type(value) in (dict, Counter):
        return {str(k): scalar(v) for k, v in value.items()}
    return {'type': type(value).__name__, 'object_id': id(value), 'not_serialized': True}


def account(pipe: Any, side: str) -> dict[str, Any]:
    suffix = side.lower()
    return {name: scalar(getattr(pipe, '_' + name + '_' + suffix, None)) for name in
            ('tsumo_count', 'pending_tsumo', 'last_seen_next', 'last_consumed_color',
             'landing_pending', 'constraint_valid', 'first_move_sec')}


def event(value: Any) -> Any:
    if value is None:
        return None
    return {'object_id': id(value), 'before_board': board(value.before_board),
        **{n: scalar(getattr(value, n, None)) for n in ('trigger_sec', 'end_sec', 'mechanism',
            'chain_count', 'total_erased', 'total_score', 'base_score', 'score_estimated')}}


def anchors() -> dict[str, tuple[int, int]]:
    require(sha(SOURCE) == FIXED[SOURCE], 'journal_source')
    tree = ast.parse(SOURCE.read_bytes())
    calls = {'origin': 'self._start_chain_estimate',
             'next_validation': 'self._validate_next_history', 'persistence': '_apply_piece_persistence_guard',
             'answer_check': 'self._update_chain_estimate_verification', 'publication': 'self._classify_board_none_reason'}
    step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_step_side')
    result = {}
    for name, target in calls.items():
        hits = [n for n in ast.walk(step) if isinstance(n, ast.Call) and ast.unparse(n.func) == target]
        require(len(hits) == 1, 'journal_unique_anchor:' + name)
        result[name] = (hits[0].lineno, hits[0].end_lineno)
    resolves = [n for n in ast.walk(step) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'resolve_after_placement']
    require({ast.unparse(n.targets[0]) for n in resolves} == {'(final_board, chain_count)', '(final_b, _)'},
            'journal_resolve_assignments')
    for node in resolves:
        key = 'resolve' if ast.unparse(node.targets[0]) == '(final_board, chain_count)' else 'resolve_secondary'
        result[key] = node.value.lineno, node.value.end_lineno
    before = [n for n in ast.walk(step) if isinstance(n, ast.Assign)
              and ast.unparse(n) == 'committed = pending.popleft()']
    after = [n for n in ast.walk(step) if isinstance(n, ast.AugAssign)
             and ast.unparse(n) == 'tsumo_count_target[committed[1]] += 1']
    require(len(before) == len(after) == 1, 'journal_fifo_anchor')
    result['fifo'] = before[0].lineno, after[0].end_lineno
    glow = [n for n in ast.walk(step) if isinstance(n, ast.If)
            and ast.unparse(n.test) == 'self._enable_ojama_warning_glow_guard and frame_bgr is not None']
    require(len(glow) == 1, 'journal_glow_anchor')
    result['glow'] = glow[0].lineno, glow[0].end_lineno
    return result


class Fifo:
    """観測したappendだけにsoftware occurrenceを割当。pair同値で追跡しない。"""
    def __init__(self) -> None:
        self.entries: dict[tuple[int, str], dict[str, Any]] = {}

    def sync(self, pipe: Any, side: str, epoch: Any) -> dict[str, Any]:
        queue = getattr(pipe, '_pending_tsumo_' + side.lower())
        key, refs = (id(pipe), side), tuple(queue)
        old = self.entries.get(key)
        if old is None or old['epoch'] != epoch:
            old = {'queue': queue, 'refs': refs, 'tokens': [None] * len(refs), 'epoch': epoch,
                   'initial_reason': 'scope_initial_unknown' if old is None else 'software_reset_invalidated',
                   'discarded_tokens': [] if old is None else list(old['tokens'])}
            self.entries[key] = old
        require(old['queue'] is queue and len(old['refs']) == len(refs)
                and all(a is b for a, b in zip(old['refs'], refs)), 'journal_unexplained_fifo_mutation')
        return old

    def appended(self, old: dict[str, Any], token: str) -> list[str]:
        after, before = tuple(old['queue']), old['refs']
        require(len(after) in (len(before), len(before) + 1)
                and all(a is b for a, b in zip(before, after)), 'journal_enqueue_prefix')
        added = [token + ':slot:' + str(len(before))] if len(after) > len(before) else []
        old['refs'], old['tokens'] = after, old['tokens'] + added
        return added

    def consumed(self, old: dict[str, Any]) -> str | None:
        after, before = tuple(old['queue']), old['refs']
        require(bool(before) and len(after) + 1 == len(before)
                and all(a is b for a, b in zip(before[1:], after)), 'journal_consume_prefix')
        consumed = old['tokens'][0]
        old['refs'], old['tokens'] = after, old['tokens'][1:]
        return consumed

    def inactive_clear(self, pipe: Any, side: str, epoch: int, beginning: dict[str, Any]) -> None:
        old = self.entries.get((id(pipe), side))
        queue = getattr(pipe, '_pending_tsumo_' + side.lower())
        if old is None or old['epoch'] != epoch or not old['refs'] or queue:
            return
        require(beginning['queue'] is old['queue'] is queue and len(beginning['refs']) == len(old['refs'])
            and all(a is b for a, b in zip(beginning['refs'], old['refs']))
            and not getattr(pipe, '_tsumo_count_' + side.lower()), 'journal_inactive_clear_unbound')
        old.update(refs=(), tokens=[], discarded_tokens=list(old['tokens']),
                   initial_reason='inactive_clear_between_begin_and_enqueue_writer_not_traced')


class Recorder:
    """一stepのみRAMに保持。初期不明FIFO、未接続会計権を維持する。"""
    def __init__(self, output: Path, history: Any, tracker: Any, controller: Any,
                 source_id: str, run_id: str, expected: list[tuple[int, str]]) -> None:
        self.output, self.history, self.tracker, self.controller = output, history, tracker, controller
        self.source_id, self.run_id, self.expected = source_id, run_id, expected
        self.selected = set(expected)
        self.stream = (output / SIDECAR).open('x', encoding='utf-8')
        self.fifo, self.active, self.closed = Fifo(), None, False
        self.errors: list[str] = []
        self.count, self.steps, self.enqueues = 0, 0, 0
        self.pipe: Any = None
        self.targets = anchors()
        self.codes: set[Any] = set()
        self.update_before: dict[str, Any] = {}

    def scope(self, pipe: Any, side: str, frame: int, clock: float) -> dict[str, Any]:
        require(type(frame) is int and type(clock) in (int, float) and math.isfinite(clock)
                and abs(clock - frame / 60) < 1e-8 and side in SIDES, 'journal_clock')
        require((frame, side) in self.selected and (self.history.frame, self.history.time_sec) == (frame, clock),
                'journal_active_clock')
        require(self.tracker._pipeline is pipe and self.tracker._machines[side] is getattr(pipe, '_sm_' + side.lower()),
                'journal_pipe_sm_binding')
        require(self.pipe is None or self.pipe is pipe, 'journal_different_pipe')
        self.pipe = pipe
        generation = dataclasses.asdict(self.tracker.generation(side))
        return {'frame_idx': frame, 'time_sec': clock, 'side': side, 'generation': generation,
                'source_id': self.source_id, 'run_id': self.run_id, 'pipe_object_id': id(pipe)}

    def epoch(self, pipe: Any, side: str) -> int:
        require(id(pipe) in self.controller.instances, 'journal_next_begin_not_observed')
        return self.controller.instances[id(pipe)].histories[side].epoch

    def emit(self, value: dict[str, Any]) -> None:
        row = {'schema': SCHEMA, 'row_index': self.count, 'physical_identity_certified': False,
               'accounting_permission': False, 'quality_gate_clear': False} | value
        self.stream.write(encoded(row) + '\n')
        self.count += 1

    def snapshot(self, frame: Any, name: str) -> dict[str, Any]:
        values, item = frame.f_locals, self.active
        pipe, side = item['pipe'], item['scope']['side']
        ctx = values.get('ctx')
        return {'stage': name, 'line': frame.f_lineno, 'accounting': account(pipe, side),
            'state': getattr(getattr(ctx, 'state', None), 'name', None),
            'boards': {k: board(values.get(k)) for k in ('cnn_board', 'prev_confirmed', 'inferred_landing',
                'final_board', 'inferred_b', 'final_b', 'published_confirmed', 'correction_board')},
            'confirmed': board(getattr(ctx, 'confirmed_board', None)),
            'pending': board(getattr(ctx, 'pending_board', None)), 'next_queue': scalar(getattr(ctx, 'next_queue', None)),
            'ever_seen': scalar(values.get('ever_seen')), 'committed': scalar(values.get('committed')),
            'falling_pair': scalar(values.get('falling_pair')), 'falling_pair_b': scalar(values.get('falling_pair_b')),
            'chain_count': scalar(values.get('chain_count')), 'pseudo': event(values.get('pseudo')),
            'active_origin': event(getattr(pipe, '_active_chain_' + side.lower(), None)),
            'enable_starvation_fix': scalar(getattr(pipe, '_enable_next_history_starvation_fix', None)),
            'min_colors': scalar(getattr(pipe, 'NEXT_HISTORY_MIN_COLORS_FOR_VALIDATION', None)),
            'is_glow': scalar(values.get('is_glow')), 'answer_check_result': scalar(values.get('answer_check_result'))}

    def observe(self, frame: Any, kind: str, arg: Any) -> None:
        item = self.active
        if item is None or frame.f_code not in self.codes:
            return
        require(frame.f_locals.get('self') is item['pipe'], 'journal_actual_code_pipe')
        if kind == 'call':
            require(item['frame'] is None, 'journal_multiple_generated_steps')
            item['frame'] = frame
        if kind == 'return':
            item['return_line'] = frame.f_lineno
        if frame is not item['frame'] or kind != 'line':
            return
        for name, (start, end) in self.targets.items():
            if frame.f_lineno == start and name not in item['begun']:
                item['begun'].add(name)
                if name == 'fifo':
                    item['fifo'] = self.fifo.sync(item['pipe'], item['scope']['side'], item['epoch'])
                data = self.snapshot(frame, name + '_before')
                if name == 'fifo':
                    data['fifo_occurrence_tokens'] = list(item['fifo']['tokens'])
                    data['initial_reason'] = item['fifo']['initial_reason']
                item['events'].append(data)
            if name in item['begun'] and name not in item['ended'] and frame.f_lineno > end:
                item['ended'].add(name)
                data = self.snapshot(frame, name + '_after')
                if name == 'fifo':
                    data['enqueue_occurrence_token'] = self.fifo.consumed(item['fifo'])
                item['events'].append(data)

    def safe_observe(self, frame: Any, kind: str, arg: Any) -> None:
        """計装エラーは完了を拒否するが原writer/原例外を置換しない。"""
        try:
            self.observe(frame, kind, arg)
        except Exception as error:
            self.errors.append('observer:' + repr(error))

    def traced(self, previous: Any) -> Any:
        def global_trace(frame: Any, kind: str, arg: Any) -> Any:
            prior = previous(frame, kind, arg) if previous is not None else None
            if frame.f_code not in self.codes:
                return prior
            self.safe_observe(frame, kind, arg)
            def local(current: Any, event_kind: str, value: Any) -> Any:
                nonlocal prior
                if prior is not None:
                    prior = prior(current, event_kind, value)
                self.safe_observe(current, event_kind, value)
                return local
            return local
        return global_trace

    def wrap_step(self, original: Any) -> Any:
        @functools.wraps(original)
        def step(pipe: Any, side: str, frame: int, clock: float, *args: Any, **kwargs: Any) -> Any:
            if (frame, side) not in self.selected:
                return original(pipe, side, frame, clock, *args, **kwargs)
            require(self.active is None, 'journal_reentry')
            scope = self.scope(pipe, side, frame, clock)
            require(self.steps < len(self.expected) and self.expected[self.steps] == (frame, side), 'journal_step_order')
            previous, profile = sys.gettrace(), sys.getprofile()
            item = {'pipe': pipe, 'scope': scope, 'epoch': self.epoch(pipe, side), 'events': [],
                    'begun': set(), 'ended': set(), 'frame': None, 'return_line': None, 'token': 'step:' + str(self.steps)}
            self.active = item
            error, result = None, None
            try:
                sys.settrace(self.traced(previous))
                result = original(pipe, side, frame, clock, *args, **kwargs)
                require(item['frame'] is not None and item['begun'] == item['ended'], 'journal_stage_unclosed')
                return result
            except BaseException as caught:
                error = caught
                self.errors.append(repr(caught))
                raise
            finally:
                sys.settrace(previous)
                self.active = None
                self.steps += 1
                self.complete_step(item, result, error, profile)
        return step

    def complete_step(self, item: dict[str, Any], result: Any, error: Any, profile: Any) -> None:
        try:
            if sys.getprofile() is not profile:
                sys.setprofile(profile)
                raise ValueError('journal_profile_changed')
            self.emit(item['scope'] | {'kind': 'step', 'token': item['token'], 'events': item['events'],
                'software_reset': item['epoch'], 'return_line': item['return_line'],
                'generation_after': dataclasses.asdict(self.tracker.generation(item['scope']['side'])),
                'status': 'returned' if error is None else 'exception', 'exception': repr(error) if error else None,
                'returned': None if result is None else {'confirmed': board(result.confirmed_board),
                    'inferred': board(result.inferred_board), 'state': result.state.name,
                    'active_origin': event(getattr(item['pipe'], '_active_chain_' + item['scope']['side'].lower(), None))},
                'code_sha256': None if item['frame'] is None else hashlib.sha256(item['frame'].f_code.co_code).hexdigest()})
        except Exception as failure:
            self.errors.append('step_save:' + repr(failure))
            if error is None:
                raise
        finally:
            # 原frameのf_back/local画像を循環参照で次stepへ保持しない。
            item['frame'] = None

    def wrap_enqueue(self, original: Any) -> Any:
        def enqueue(pipe: Any, side: str, frame: int, clock: float, active: bool, pair: Any) -> Any:
            if (frame, side) not in self.selected:
                return original(pipe, side, frame, clock, active, pair)
            scope = self.scope(pipe, side, frame, clock)
            epoch = self.epoch(pipe, side)
            beginning = self.update_before.get(side)
            if active is False and beginning is not None:
                self.fifo.inactive_clear(pipe, side, epoch, beginning)
            saved = self.fifo.sync(pipe, side, epoch)
            token = self.source_id + ':' + self.run_id + ':reset:' + str(epoch) + ':' + side + ':enqueue:' + str(self.enqueues)
            self.enqueues += 1
            before, error = account(pipe, side), None
            try:
                result = original(pipe, side, frame, clock, active, pair)
                added = self.fifo.appended(saved, token)
                self.emit(scope | {'kind': 'enqueue', 'token': token, 'software_reset': epoch,
                    'pair': scalar(pair), 'active': active, 'before': before, 'after': account(pipe, side),
                    'added_occurrence_tokens': added, 'fifo_occurrence_tokens': saved['tokens'],
                    'initial_reason': saved['initial_reason'], 'discarded_tokens': saved['discarded_tokens'],
                    'update_begin_accounting': None if beginning is None else beginning['accounting'],
                    'status': 'returned', 'returned_none': result is None})
                return result
            except BaseException as caught:
                self.errors.append(repr(caught))
                try:
                    self.emit(scope | {'kind': 'enqueue', 'token': token, 'status': 'exception',
                        'exception': repr(caught), 'before': before, 'after': account(pipe, side)})
                except Exception as failure:
                    self.errors.append('enqueue_save:' + repr(failure))
                raise
        return enqueue

    def wrap_begin(self, original: Any) -> Any:
        def begin(pipe: Any, frame: int, clock: float) -> Any:
            self.update_before = {}
            if any((frame, side) in self.selected for side in SIDES):
                self.update_before = {side: {'queue': getattr(pipe, '_pending_tsumo_' + side.lower()),
                    'refs': tuple(getattr(pipe, '_pending_tsumo_' + side.lower())),
                    'accounting': account(pipe, side)} for side in SIDES}
            return original(pipe, frame, clock)
        return begin

    def close(self) -> None:
        self.stream.close()
        self.closed = True


def patch(stack: Any, owner: Any, name: str, value: Any) -> None:
    present, old = name in vars(owner), vars(owner).get(name)
    setattr(owner, name, value)
    stack.callback(setattr, owner, name, old) if present else stack.callback(delattr, owner, name)


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any], *,
            expected_scopes: list[tuple[int, str]] | None = None,
            source_id: str | None = None, run_id: str | None = None) -> Recorder:
    guards()
    require(KEY not in state, 'journal_already_installed')
    cls = collector.RecognitionPipeline
    namespace = vars(sys.modules[cls.__module__])
    controller = namespace.get('__next_live')
    require(controller is not None, 'journal_next_controller_missing')
    require(Path(inspect.getfile(type(controller))).resolve() == NEXT, 'journal_next_controller_source')
    modules = list({id(m): m for m in tuple(sys.modules.values()) if m is not None and getattr(m, '__file__', None)
               and Path(m.__file__).resolve() == WITNESS}.values())
    require(len(modules) == 1, 'journal_fixed_witness_module')
    targets = {history.step_code, state['pending_code']}
    functions = modules[0].step_functions(cls._step_side, targets)
    require(set(functions) == targets and len(targets) == 2, 'journal_two_generated_codes')
    require({hashlib.sha256(c.co_code).hexdigest() for c in targets} == CODE_HASHES
        and all(Path(c.co_filename).resolve() == SOURCE for c in targets), 'journal_unknown_generated_code')
    pb = state.get('hidden_probability_observer')
    source_id, run_id = source_id or getattr(pb, 'source_id', None), run_id or getattr(pb, 'run_id', None)
    require(type(source_id) is str and bool(source_id) and type(run_id) is str and bool(run_id), 'journal_source_run')
    scopes = expected_scopes if expected_scopes is not None else [(f, s) for f in range(FIRST, LAST + 1, STRIDE) for s in SIDES]
    require(bool(scopes) and len(set(scopes)) == len(scopes) and scopes == sorted(scopes)
        and all(type(f) is int and f >= 0 and s in SIDES for f, s in scopes), 'journal_expected_scope')
    rec = Recorder(Path(state['output']), history, state['tracker'], controller, source_id, run_id, scopes)
    rec.codes = targets
    state[KEY] = rec
    stack.callback(rec.close)
    patch(stack, controller, 'enqueue', rec.wrap_enqueue(controller.enqueue))
    patch(stack, controller, 'begin', rec.wrap_begin(controller.begin))
    patch(stack, cls, '_step_side', rec.wrap_step(cls._step_side))
    return rec


def verify_stages(row: dict[str, Any], targets: dict[str, Any]) -> set[str]:
    require(row['code_sha256'] in CODE_HASHES and type(row['return_line']) is int, 'journal_saved_code')
    require(type(row['events']) is list, 'journal_saved_events')
    opened, closed, order = set(), set(), []
    for value in row['events']:
        name, phase = value['stage'].rsplit('_', 1)
        require(name in targets and type(value['line']) is int, 'journal_saved_stage')
        if phase == 'before':
            require(name not in opened and value['line'] == targets[name][0], 'journal_stage_duplicate_before')
            opened.add(name)
            order.append(targets[name][0])
        else:
            require(phase == 'after' and name in opened and name not in closed
                    and value['line'] > targets[name][1], 'journal_stage_extra_after')
            closed.add(name)
        require(type(value['accounting']['tsumo_count']) is dict
                and type(value['accounting']['pending_tsumo']) is list, 'journal_account_missing')
    require(opened == closed and order == sorted(order), 'journal_stage_order_or_unclosed')
    return opened


def verify_enqueue(row: dict[str, Any]) -> None:
    before, after = row['before']['pending_tsumo'], row['after']['pending_tsumo']
    require(type(before) is list and type(after) is list and len(after) in (len(before), len(before) + 1)
            and after[:len(before)] == before, 'journal_saved_enqueue_prefix')
    added, tokens = row['added_occurrence_tokens'], row['fifo_occurrence_tokens']
    require(type(added) is list and len(added) == len(after) - len(before)
            and (not added or added == [row['token'] + ':slot:' + str(len(before))]), 'journal_saved_enqueue_occurrence')
    require(type(tokens) is list and len(tokens) == len(after)
            and (not added or tokens[-1:] == added), 'journal_saved_fifo_tokens')
    require(type(row['software_reset']) is int and type(row['active']) is bool, 'journal_saved_enqueue_scope')
    require(row['before']['tsumo_count'] == row['after']['tsumo_count'], 'journal_enqueue_counter_changed')


def verify_rows(output: Path, receipt: dict[str, Any], expected: list[Any]) -> dict[str, Any]:
    steps, seen, stages, enqueues, previous = [], set(), set(), 0, -1
    with (output / SIDECAR).open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            require(row['row_index'] == index and type(row['row_index']) is int and row['schema'] == SCHEMA, 'journal_saved_row')
            require(type(row['frame_idx']) is int and row['side'] in SIDES and type(row['time_sec']) in (int, float)
                and math.isfinite(row['time_sec']) and abs(row['time_sec'] - row['frame_idx'] / 60) < 1e-8, 'journal_saved_clock')
            require(row['frame_idx'] >= previous and [row['frame_idx'], row['side']] in expected, 'journal_saved_row_scope')
            previous = row['frame_idx']
            require(all(row[k] == receipt[k] for k in ('source_id', 'run_id')), 'journal_saved_source')
            require(row['status'] == 'returned' and row['token'] not in seen, 'journal_saved_failure_or_duplicate')
            seen.add(row['token'])
            require(all(row[k] is False for k in PERMISSIONS), 'journal_saved_permission')
            if row['kind'] == 'step':
                steps.append([row['frame_idx'], row['side']])
                stages.update(verify_stages(row, receipt['anchors']))
            else:
                require(row['kind'] == 'enqueue', 'journal_unknown_kind')
                verify_enqueue(row)
                enqueues += 1
    require(steps == expected and len(seen) == receipt['row_count'] and enqueues == receipt['enqueue_calls']
            and len(steps) == receipt['step_count'], 'journal_saved_coverage')
    return {'scope_count': len(steps), 'row_count': len(seen), 'enqueues': enqueues,
            'observed_stages': sorted(stages), 'quality_gate_clear': False}


def verify(output: Path, *, expected_scopes: list[tuple[int, str]] | None = None) -> dict[str, Any]:
    receipt = json.loads((output / RECEIPT).read_text())
    status = json.loads((output / STATUS).read_text())
    require(set(status) == {'closed', 'errors', 'active'} and status['closed'] is True
            and status['active'] is False and type(status['errors']) is list and not status['errors'],
            'journal_saved_status')
    require(set(receipt['sha256']) == {SIDECAR, STATUS}
        and all(sha(output / n) == h for n, h in receipt['sha256'].items()), 'journal_saved_sha')
    scopes = expected_scopes if expected_scopes is not None else [(f, s) for f in range(FIRST, LAST + 1, STRIDE) for s in SIDES]
    expected = [list(v) for v in scopes]
    require(receipt['expected_scopes'] == expected, 'journal_saved_expected')
    require(receipt['guards'] == guards() and receipt['code_sha256'] == sorted(CODE_HASHES), 'journal_saved_guards')
    require(receipt['anchors'] == {k: list(v) for k, v in anchors().items()}, 'journal_saved_anchors')
    require(all(receipt[k] is False for k in PERMISSIONS), 'journal_receipt_permission')
    result = verify_rows(output, receipt, expected)
    required = set(anchors()) - {'resolve_secondary'} if expected_scopes is None else COMMON_STAGES
    require(required <= set(result['observed_stages']), 'journal_zero_required_stage')
    require(expected_scopes is not None or result['enqueues'] > 0, 'journal_no_enqueue_calls')
    return result


def finish(state: dict[str, Any]) -> None:
    rec = state[KEY]
    write(rec.output / STATUS, {'closed': rec.closed, 'errors': rec.errors, 'active': rec.active is not None})
    require(rec.closed and not rec.errors and rec.active is None, 'journal_incomplete')
    write(rec.output / RECEIPT, {'source_id': rec.source_id, 'run_id': rec.run_id,
        'expected_scopes': rec.expected, 'row_count': rec.count, 'step_count': rec.steps,
        'enqueue_calls': rec.enqueues, 'anchors': rec.targets, 'guards': guards(), 'code_sha256': sorted(CODE_HASHES),
        'sha256': {n: sha(rec.output / n) for n in (SIDECAR, STATUS)},
        'quality_gate_clear': False, 'accounting_permission': False, 'physical_identity_certified': False})
    verify(rec.output, expected_scopes=rec.expected)
