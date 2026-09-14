"""私有loaderが返す最終Sessionへ装着し、元型identity検査を維持する。"""
from pathlib import Path
import json
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent / 'g2_async_projected_evaluation_2026-09-13_v1'
REPO = ROOT.parents[2]
FROZEN = REPO / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
PUBLICATION = ROOT.parent / 'g2_belief_live_publication_2026-09-11_v1'
NOTICE = ROOT.parent / 'g2_second_origin_postrun_2026-09-13_v1'
PATCH = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/runtime_patch.py'
ALIASES = ('_async_live_pair_reader', '_async_live_origin_capture',
           '_async_live_session_origin_binding', '_async_live_settled_notice',
           '_async_live_commit_types', '_async_live_prediction_ledger',
           '_async_live_projected_view', '_async_live_prefix_projection',
           '_async_live_prefix_reader', '_async_live_projected_input', '_async_live_fixed_origin_reference',
           '_async_live_fixed_root_probability')


def owned_loader(stack: Any, load: Any) -> Any:
    if any(alias in sys.modules for alias in ALIASES):
        raise ValueError('origin_runtime_foreign_alias')
    owned: dict = {}
    closed = False
    def release(kind: Any, body: Any, trace: Any) -> bool:
        nonlocal closed
        closed = True
        changed = []
        for alias, module in owned.items():
            if sys.modules.get(alias) is module: sys.modules.pop(alias)
            else: changed.append(alias)
        owned.clear()
        if changed and body is None: raise ValueError('origin_runtime_alias_changed:' + ','.join(changed))
        return False
    stack.push(release)
    def selected(alias: str, path: Path, injection: Any = None) -> Any:
        if closed: raise ValueError('origin_runtime_loader_closed')
        try: return load(alias, path, injection)
        finally:
            module = sys.modules.get(alias)
            if module is not None and Path(module.__file__).resolve() == path.resolve():
                owned[alias] = module
    return selected


def physics(session_module: Any) -> tuple:
    """元Beliefの私有凍結物理を再用し、原pipelineのsimulate計装へ流入させない。"""
    belief = session_module.B.B
    board, simulator = belief.Board, belief.ChainSimulator
    board_module = sys.modules[board.__module__]
    if Path(board_module.__file__).resolve() != FROZEN / 'src/board.py':
        raise ValueError('origin_runtime_nonfrozen_board')
    if Path(simulator.__init__.__code__.co_filename).resolve() != FROZEN / 'src/chain.py':
        raise ValueError('origin_runtime_nonfrozen_simulator')
    if simulator is sys.modules['src.chain'].ChainSimulator:
        raise ValueError('origin_runtime_pipeline_simulator_shared')
    return board, simulator


def projected_callback(load: Any, session: Any, lease: Any, reader: Any, notice: Any,
                       belief: Any) -> Any:
    live, adapter = lease.current()
    if live.module.B is not belief:
        raise ValueError('projected_input_belief_owner')
    view = load('_async_live_projected_view', ROOT / 'projected_view.py', {'belief': belief})
    projection = load('_async_live_prefix_projection', ROOT / 'prefix_projection.py',
        dict(identified_origin_candidate=adapter.O, lane_state=adapter.N, projected_view=view))
    source = load('_async_live_prefix_reader', ROOT / 'prefix_live_reader.py')
    packet = load('_async_live_projected_input', ROOT / 'prefix_projected_input.py')
    def read(frame: int) -> dict:
        return packet.read(session, lease, reader, source, projection, frame, settled=notice.is_settled)
    return read


def reference_hook(load: Any, reader: Any, enabled: bool) -> tuple:
    """計数は所有wrapperに置く。stateless読取moduleに共有状態を作らない。"""
    counts = dict(calls=0, REFERENCE_ONLY=0, HOLD=0, last_frame=-1)
    if not enabled: return None, counts
    module = load('_async_live_fixed_origin_reference', ROOT / 'fixed_origin_reference.py',
                  {'journal_pair_reader': reader})
    def completed(capture: Any, frame: int) -> None:
        if frame <= counts['last_frame']: raise ValueError('fixed_reference_duplicate_save')
        result = module.save(capture, frame)
        counts['calls'] += 1
        for side in result['sides']: counts[side['status']] += 1
        counts['last_frame'] = frame
    return completed, counts


def root_hook(load: Any, reader: Any, history_getter: Any, enabled: bool, previous: Any) -> tuple:
    counts = dict(calls=0, READY=0, HOLD=0, last_frame=-1)
    if not enabled: return previous, counts
    if previous is None or history_getter is None: raise ValueError('root_probability_dependencies')
    module = load('_async_live_fixed_root_probability', ROOT / 'fixed_root_probability.py',
                  {'fixed_origin_reference': sys.modules['_async_live_fixed_origin_reference']})
    def completed(capture: Any, frame: int) -> None:
        if frame <= counts['last_frame']: raise ValueError('root_probability_duplicate_save')
        previous(capture, frame)
        source = None
        try:
            source = reader.read_pair(capture.witness, capture.journal, capture.pipe, frame).rows_json
            result = module.read(history_getter(), capture, frame)
            capture.stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
            capture.stream.flush()
            counts['calls'] += 1
            counts[result['status']] += 1
            counts['last_frame'] = frame
        except BaseException as error:
            capture.error = error
            capture.record_failure(frame, source, error)
            raise
    return completed, counts


def bind(stack: Any, load: Any, session_module: Any, cls: type, *, prefix_lease: Any = None,
         history_getter: Any = None, fixed_reference_enabled: bool = False,
         root_probability_enabled: bool = False) -> dict:
    Board, ChainSimulator = physics(session_module)
    load = owned_loader(stack, load)
    commit = load('_async_live_commit_types', REPO / 'src/chain_commit_candidate_v1.py')
    ledger = load('_async_live_prediction_ledger', REPO / 'src/chain_prediction_ledger_v1.py',
                  {'src.chain_commit_candidate_v1': commit})
    reader = load('_async_live_pair_reader', ROOT / 'journal_pair_reader.py',
                  {'journal_witness': session_module.W})
    consumer = load('_async_live_origin_capture', Path(__file__).resolve().parent / 'journal_origin_capture_candidate.py',
                    {'journal_pair_reader': reader})
    lifecycle = load('_async_live_session_origin_binding', ROOT / 'session_origin_binding.py')
    notice = load('_async_live_settled_notice', NOTICE / 'settled_notice.py')
    created_frames: list[int] = []
    def factory(session: Any, stream: Any) -> Any:
        if type(session.witness) is not reader.W.Witness:
            raise ValueError('origin_runtime_witness_type')
        callback = None if prefix_lease is None else projected_callback(
            load, session, prefix_lease, reader, notice, session_module.B.B)
        capture = consumer.OriginCapture(session.witness, session.journal, session.pipe, ledger,
            ChainSimulator(exclude_hidden_row_from_pop=True), Board.from_dict, notice.is_settled, stream,
            projected_input=callback)
        if history_getter is not None:
            try: history_getter().prime(capture, session.journal.history.frame)
            except BaseException:
                capture.close()
                raise
        created_frames.append(session.journal.history.frame)
        return capture
    after, counts = reference_hook(load, reader, fixed_reference_enabled)
    after, root_counts = root_hook(load, reader, history_getter, root_probability_enabled, after)
    lifecycle.install_class(stack, cls, factory, after_completed=after)
    result = dict(module=session_module, cls=cls, initialize=cls.__init__, completed=cls.completed,
                reader=reader, consumer=consumer, created_frames=created_frames, live_hook_verified=False,
                projected_input_enabled=prefix_lease is not None, closed=False,
                fixed_reference_enabled=fixed_reference_enabled, reference_counts=counts,
                root_probability_enabled=root_probability_enabled, root_probability_counts=root_counts)
    stack.callback(result.__setitem__, 'closed', True)
    return result


def publication(parent: type) -> Any:
    matches = []
    for base in parent.__mro__:
        method = vars(base).get('__init__')
        namespace = getattr(method, '__globals__', {})
        if namespace.get('__file__') and Path(namespace['__file__']).resolve() == PUBLICATION / 'live_session.py':
            module = sys.modules[base.__module__]
            if namespace is not vars(module): raise ValueError('origin_runtime_parent_globals')
            matches.append(module)
    if len(matches) != 1: raise ValueError('origin_runtime_publication_parent')
    return matches[0]


def install(stack: Any, bootstrap: Any, replace: Any, *, prefix_lease: Any = None,
            binding_stack: Any = None, history_getter: Any = None,
            fixed_reference_enabled: bool = False, root_probability_enabled: bool = False) -> dict:
    original = bootstrap.load
    state: dict = dict(binding=None, patch=None)
    def install_factory(module: Any) -> None:
        prior = module.scheduled
        def scheduled(load: Any, parent: type) -> type:
            if state['binding'] is not None and state['binding']['closed']:
                raise ValueError('origin_runtime_binding_closed')
            cls = prior(load, parent)  # 原factoryのglobals参照が完了してから装着する。
            if state['binding'] is None:
                scope = stack if binding_stack is None else binding_stack()
                state['binding'] = bind(scope, load, publication(parent), cls,
                    prefix_lease=prefix_lease, history_getter=history_getter,
                    fixed_reference_enabled=fixed_reference_enabled, root_probability_enabled=root_probability_enabled)
            elif cls is not state['binding']['cls']:
                raise ValueError('origin_runtime_factory_replaced')
            return cls
        replace(stack, module, 'scheduled', scheduled)
        state['patch'] = (module, scheduled)
    def load(alias: str, path: Path, injection: Any = None) -> Any:
        selected = state['binding']
        if selected is not None and selected['closed'] and Path(path).resolve() == PUBLICATION / 'live_session.py':
            raise ValueError('origin_runtime_binding_closed')
        value = original(alias, path, injection)
        if Path(path).resolve() == PATCH:
            if state['patch'] is None: install_factory(value)
            elif state['patch'][0] is not value: raise ValueError('origin_runtime_patch_reloaded')
        if state['binding'] is not None and not state['binding']['closed']:
            selected = state['binding']
            cls = selected['cls']
            if cls.__init__ is not selected['initialize'] or cls.completed is not selected['completed']:
                raise ValueError('origin_runtime_session_changed')
        return value
    replace(stack, bootstrap, 'load', load)
    return state


def verify(state: dict, *, require_instance: bool = False, require_projected_input: bool = False,
           require_fixed_reference: bool = False, require_reference_sample: bool = False,
           require_root_probability: bool = False, require_root_probability_sample: bool = False) -> None:
    """未到達を成功扱いしない。生成/実instanceの必須境界でcallerが検査する。"""
    if state['patch'] is None: raise ValueError('origin_runtime_patch_not_hooked')
    binding = state['binding']
    if binding is None: raise ValueError('origin_runtime_session_not_bound')
    if binding['closed']: raise ValueError('origin_runtime_binding_closed')
    cls = binding['module'].Session
    if (cls is not binding['cls'] or cls.__init__ is not binding['initialize']
            or cls.completed is not binding['completed']):
        raise ValueError('origin_runtime_final_class_changed')
    if require_instance and not binding['created_frames']:
        raise ValueError('origin_runtime_instance_not_created')
    if require_projected_input and not binding['projected_input_enabled']:
        raise ValueError('origin_runtime_projected_input_not_enabled')
    if (require_fixed_reference or require_reference_sample) and not binding['fixed_reference_enabled']:
        raise ValueError('origin_runtime_fixed_reference_not_enabled')
    if require_reference_sample and binding['reference_counts']['REFERENCE_ONLY'] < 1:
        raise ValueError('origin_runtime_fixed_reference_sample_missing')
    if (require_root_probability or require_root_probability_sample) and not binding['root_probability_enabled']:
        raise ValueError('origin_runtime_root_probability_not_enabled')
    if require_root_probability_sample and binding['root_probability_counts']['READY'] < 1:
        raise ValueError('origin_runtime_root_probability_sample_missing')
