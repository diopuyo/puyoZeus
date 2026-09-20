"""実collectorの入力と採録後metadataを観測する。採否・値は変更しない。"""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from enum import Enum
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import CodeType
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCE = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/scripts/collect_boards_lean.py'
SOURCE_SHA = '672963055a9fffa66531411be0a51742ee57c35de481652a77d02158ce24bee8'
FIRST, LAST, STRIDE, FPS = 29052, 36298, 2, 60
SIDES = ('1P', '2P')
KEY = 'collector_metadata_sink'
ENTRIES, STATUS, RECEIPT = 'collector_metadata.jsonl', 'COLLECTOR_METADATA_STATUS.json', 'COLLECTOR_METADATA_RECEIPT.json'
REQUIRED = frozenset((ENTRIES, STATUS, RECEIPT))
OWN = ('observer.py', 'test_observer.py', 'run_cpu.py', 'CONTRACT.md')
OBJECT_ARGS = frozenset(('acc', 'state', 'board', 'bstate', 'estimated_board', 'chain_event', 'shared_game', 'physics_sim'))
ROW_KEYS = frozenset(('frame_idx', 'side', 'time_sec', 'arguments', 'state_before', 'shared_before',
    'appends', 'state_after', 'shared_after', 'returned'))
STORED_KEYS = frozenset(('grids', 'video_ids', 'sides', 't_secs', 'game_idxs', 'frame_idxs', 'scores',
    'next1_as', 'next1_bs', 'dnext_as', 'dnext_bs', 'chain_trigger_secs', 'chain_mechanisms', 'tsumo_counts',
    'all_clear_pendings', 'ojama_net_balances', 'ojama_forecasts', 'match_end_lockeds',
    'post_match_lockdown_actives', 'stable_persistence_confidences', 'board_provenances'))


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def serial(value: Any) -> Any:
    """欠測NaNを0にせず、型付き標識で保持する。"""
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float:
        return value if math.isfinite(value) else {'nonfinite_float': repr(value)}
    if isinstance(value, Enum):
        return {'enum_type': kind.__name__, 'value': serial(value.value)}
    if kind is bytes:
        return {'bytes_hex': value.hex()}
    if kind in (tuple, list):
        return {'sequence_type': kind.__name__, 'items': [serial(v) for v in value]}
    if isinstance(value, dict):
        return {'mapping': [[serial(k), serial(v)] for k, v in value.items()]}
    if isinstance(value, (set, frozenset)):
        return {'set_type': kind.__name__, 'items': [serial(v) for v in sorted(value, key=repr)]}
    if is_dataclass(value):
        return {'dataclass_type': kind.__name__, 'fields': {f.name: serial(getattr(value, f.name)) for f in fields(value)}}
    import numpy as np
    if isinstance(value, np.ndarray):
        return {'ndarray_dtype': str(value.dtype), 'shape': list(value.shape), 'values': serial(value.tolist())}
    if isinstance(value, np.generic):
        return serial(value.item())
    raise ValueError('metadata_unsupported_type:' + kind.__name__)


def board(value: Any, collector: Any) -> Any:
    require(value is None or type(value) is collector.Board, 'metadata_board_type')
    return None if value is None else serial(value._grid)


def arguments(values: dict[str, Any], collector: Any) -> dict[str, Any]:
    result = {k: serial(v) for k, v in values.items() if k not in OBJECT_ARGS}
    result.update(board=board(values['board'], collector), estimated_board=board(values['estimated_board'], collector),
        bstate=serial(values['bstate']))
    chain = values['chain_event']
    result['chain_event_used_fields'] = None if chain is None else {k: serial(getattr(chain, k, None))
        for k in ('trigger_sec', 'mechanism')}
    sim = values['physics_sim']
    result['physics_sim_type'] = None if sim is None else type(sim).__name__
    return result


def validate_collector(collector: Any) -> None:
    require(sha(SOURCE) == SOURCE_SHA and Path(collector.__file__).resolve() == SOURCE, 'metadata_collector_source')
    code = next(v for v in compile(SOURCE.read_bytes(), str(SOURCE), 'exec').co_consts
        if isinstance(v, CodeType) and v.co_name == '_process_side_lean')
    require(collector._process_side_lean.__code__ == code, 'metadata_collector_function')


def guards() -> dict[str, str]:
    require(sha(SOURCE) == SOURCE_SHA, 'metadata_source_changed')
    return {str(SOURCE): SOURCE_SHA} | {str(ROOT / n): sha(ROOT / n) for n in OWN}


class Sink:
    def __init__(self, collector: Any, output: Path) -> None:
        self.collector, self.output = collector, output
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.busy, self.closed = False, False
        self.before = guards()

    def append_capture(self, acc: Any, record: dict[str, Any], original: Any) -> Any:
        def wrapped(grid: Any, video: str, side: str, seconds: float, game: int, frame: int, **kwargs: Any) -> Any:
            before = len(acc.grids)
            inputs = serial({'grid': grid, 'video_id': video, 'side': side, 't_sec': seconds,
                'game_idx': game, 'frame_idx': frame, **kwargs})
            result = original(grid, video, side, seconds, game, frame, **kwargs)
            require(len(acc.grids) == before + 1, 'metadata_append_not_one')
            stored = {}
            for field in fields(acc):
                if field.name == 'wons':
                    continue
                values = getattr(acc, field.name)
                require(type(values) is list and len(values) == before + 1, 'metadata_column_length')
                stored[field.name] = serial(values[-1])
            record['appends'].append({'before_count': before, 'arguments': inputs, 'stored_nonlabel_row': stored})
            return result
        return wrapped

    def invoke(self, original: Any, values: dict[str, Any], args: Any, kwargs: Any) -> Any:
        acc, state, shared = values['acc'], values['state'], values['shared_game']
        require(type(acc) is self.collector._LeanNpzAccumulator and not self.busy, 'metadata_acc_or_reentry')
        require('append' not in vars(acc), 'metadata_instance_append_override')
        row = {'frame_idx': values['frame_idx'], 'side': values['side_label'], 'time_sec': values['t_sec'],
            'arguments': arguments(values, self.collector), 'state_before': serial(state),
            'shared_before': serial(shared), 'appends': []}
        self.busy = True
        acc.append = self.append_capture(acc, row, acc.append)
        try:
            result = original(*args, **kwargs)
            row.update(state_after=serial(state), shared_after=serial(shared), returned=serial(result))
            require(len(row['appends']) <= 1, 'metadata_multiple_appends')
            self.rows.append(row)
            return result
        finally:
            del acc.append
            self.busy = False

    def wrapper(self, original: Any) -> Any:
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            frame = values['frame_idx']
            if not (type(frame) is int and FIRST <= frame <= LAST):
                return original(*args, **kwargs)
            try:
                require(values['side_label'] in SIDES and (frame - FIRST) % STRIDE == 0
                    and values['t_sec'] == frame / FPS, 'metadata_clock')
                return self.invoke(original, values, args, kwargs)
            except BaseException as error:
                self.errors.append(repr(error))
                raise
        return wrapped

    def close(self) -> None:
        require(not self.closed and not self.busy, 'metadata_close_state')
        with (self.output / ENTRIES).open('x') as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        self.closed = True
        write(self.output / STATUS, {'closed': True, 'errors': self.errors, 'rows': len(self.rows),
            'label_columns_read': False, 'quality_gate_clear': False})


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any], *, enabled: bool = False) -> None:
    if not enabled:
        return
    require(type(enabled) is bool and KEY not in state, 'metadata_install_reentry')
    validate_collector(collector)
    sink, original = Sink(collector, state['output']), collector._process_side_lean
    stack.callback(sink.close)
    stack.callback(setattr, collector, '_process_side_lean', original)
    collector._process_side_lean = sink.wrapper(original)
    state[KEY] = sink


def validate_row(row: dict[str, Any], frame: int, side: str, append_count: int) -> int:
    require(set(row) == ROW_KEYS and type(row['frame_idx']) is int and row['frame_idx'] == frame and row['side'] == side
        and row['time_sec'] == frame / FPS, 'metadata_saved_row_clock_or_schema')
    args = row['arguments']
    require(type(args['frame_idx']) is int and args['frame_idx'] == frame and args['side_label'] == side and args['t_sec'] == frame / FPS,
        'metadata_saved_argument_clock')
    require(type(row['appends']) is list and len(row['appends']) <= 1, 'metadata_saved_append_count')
    for item in row['appends']:
        require(set(item) == {'before_count', 'arguments', 'stored_nonlabel_row'}
            and type(item['before_count']) is int and item['before_count'] == append_count, 'metadata_saved_append_sequence')
        saved = item['stored_nonlabel_row']
        require(set(saved) == STORED_KEYS, 'metadata_saved_column_schema')
        game = row['state_after']['fields']['game_idx']
        require(type(saved['frame_idxs']) is int and saved['frame_idxs'] == frame and saved['sides'] == side
            and saved['t_secs'] == round(frame / FPS, 3) and type(game) is int and type(saved['game_idxs']) is int
            and saved['game_idxs'] == game, 'metadata_saved_stored_clock_game')
        tag = args['stable_persistence_confidence']
        require(tag is None or type(tag) is bool, 'metadata_saved_image_tag_type')
        require(saved['stable_persistence_confidences'] == (-1 if tag is None else int(tag)), 'metadata_saved_image_tag')
        captured = item['arguments']
        require(set(captured) == {'mapping'}, 'metadata_saved_append_arguments')
        pairs = captured['mapping']
        require(all(type(p) is list and len(p) == 2 and type(p[0]) is str for p in pairs)
            and len({p[0] for p in pairs}) == len(pairs), 'metadata_saved_append_argument_keys')
        actual = dict(pairs)
        require(type(actual['frame_idx']) is int and actual['frame_idx'] == frame and actual['side'] == side
            and actual['t_sec'] == saved['t_secs'] and type(actual['game_idx']) is int
            and actual['game_idx'] == saved['game_idxs'] and actual['grid'] == saved['grids'],
            'metadata_saved_append_payload')
        append_count += 1
    return append_count


def saved_counts(output: Path) -> tuple[int, int]:
    """保存した実行順と非ラベル行を再読し、receiptの件数だけに依存しない。"""
    expected = ((frame, side) for frame in range(FIRST, LAST + STRIDE, STRIDE) for side in SIDES)
    count, appends = 0, 0
    with (output / ENTRIES).open() as stream:
        for line in stream:
            clock = next(expected, None)
            require(clock is not None, 'metadata_saved_excess_rows')
            appends = validate_row(json.loads(line), *clock, appends)
            count += 1
    require(next(expected, None) is None, 'metadata_saved_missing_rows')
    return count, appends


def finish(state: dict[str, Any]) -> None:
    sink = state[KEY]
    require(sink.closed and not sink.errors and sink.before == guards(), 'metadata_failed_or_changed')
    clocks = [(r['frame_idx'], r['side']) for r in sink.rows]
    expected = [(frame, side) for frame in range(FIRST, LAST + STRIDE, STRIDE) for side in SIDES]
    require(clocks == expected, 'metadata_full_coverage')
    count, appends = saved_counts(sink.output)
    require(count == len(clocks) and appends == sum(len(r['appends']) for r in sink.rows), 'metadata_saved_memory_counts')
    write(sink.output / RECEIPT, {'guards': sink.before, 'rows': len(clocks),
        'sha256': {n: sha(sink.output / n) for n in (ENTRIES, STATUS)},
        'actual_append_count': appends, 'quality_gate_clear': False,
        'source_values_changed': False, 'label_columns_read': False})


def verify(output: Path) -> None:
    receipt = json.loads((output / RECEIPT).read_text())
    status = json.loads((output / STATUS).read_text())
    require(receipt['guards'] == guards() and set(receipt['sha256']) == {ENTRIES, STATUS}, 'metadata_saved_guards')
    require(all(sha(output / n) == h for n, h in receipt['sha256'].items()), 'metadata_saved_sha')
    require(status['closed'] is True and status['errors'] == [] and status['rows'] == receipt['rows'], 'metadata_saved_failure')
    require(receipt['rows'] == (LAST - FIRST + STRIDE) // STRIDE * len(SIDES), 'metadata_saved_coverage')
    require(saved_counts(output) == (receipt['rows'], receipt['actual_append_count']), 'metadata_saved_counts')
    require(all(receipt[k] is False for k in ('quality_gate_clear', 'source_values_changed', 'label_columns_read')), 'metadata_permissions')
