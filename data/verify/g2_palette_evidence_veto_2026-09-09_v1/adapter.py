"""同callのCNNと全色HSVが元色を支持する時だけ履歴置換を留保する。"""
from __future__ import annotations
from collections import Counter
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
SOURCE = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py'
SOURCE_SHA = '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02'
DEPENDENCIES = {SOURCE: SOURCE_SHA,
    SOURCE.parent / 'placement_inferrer.py': '412a15120db0d6e9c10f38ad8d5fa8dff1dfa05f05f3d1a14b71f4ce10610b82',
    SOURCE.parent / 'board.py': 'a314348cc0132f56f6f1ddf37ba12c2c97c6f7dffdb87ab81eab27eb4a9ef01c'}
OWN = ('adapter.py', 'test_adapter.py', 'run_cpu.py', 'CONTRACT.md')
KEY = 'palette_evidence_veto'
SIDECAR, RECEIPT = 'palette_evidence_veto.jsonl', 'PALETTE_EVIDENCE_VETO.json'
REQUIRED = frozenset((SIDECAR, RECEIPT))
VISIBLE_FIRST_ROW = 1
COLOR_CODES = frozenset(range(1, 6))
# 固定依存の既存定数を保存検証でも使い、cold verify時にsrcをimportしない。
MINIMUM_SATURATION = 60
MAXIMUM_COLOR_DISTANCE = 60.0


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guards() -> dict[str, str]:
    require(all(sha(path) == value for path, value in DEPENDENCIES.items()), 'palette_fixed_source')
    return {str(path): value for path, value in DEPENDENCIES.items()} | {str(ROOT / name): sha(ROOT / name) for name in OWN}


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded(value))


def candidates(board: Any, result: Any) -> list[tuple[int, int, int, int]]:
    return [(r, c, int(board.get(r, c)), int(result.get(r, c)))
        for r in range(VISIBLE_FIRST_ROW, len(board._grid)) for c in range(len(board._grid[r]))
        if int(board.get(r, c)) in COLOR_CODES and int(result.get(r, c)) in COLOR_CODES
        and int(board.get(r, c)) != int(result.get(r, c))]


def support(original: int, median: tuple[int, int, int], distances: dict[int, float],
            minimum_saturation: int, maximum_distance: float) -> str:
    require(set(distances) == COLOR_CODES and all(math.isfinite(x) for x in distances.values()), 'palette_distance_shape')
    if median[1] < minimum_saturation:
        return 'low_saturation'
    if distances[original] > maximum_distance:
        return 'distant_original'
    if any(distances[other] <= distances[original] for other in COLOR_CODES - {original}):
        return 'original_not_unique_nearest'
    return 'cnn_and_full_hsv_support_original'


def measure(frame: Any, region: Any, row: int, col: int) -> Any:
    import cv2
    import numpy as np
    from src import placement_inferrer as I
    require(Path(I.__file__).resolve() == SOURCE.parent / 'placement_inferrer.py', 'palette_color_module')
    patch = I._extract_cell_patch_from_frame(frame, region, row, col)
    if patch is None or not patch.size:
        return None
    values = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    median = tuple(int(np.median(values[:, :, n])) for n in range(3))
    distances = {color: I._hsv_distance(*median, center) for color, center in I.COLOR_HSV_CENTERS.items()}
    return median, distances, I.HSV_MIN_SATURATION_FOR_CLASSIFY, I.HSV_CLASSIFY_MAX_DISTANCE


def apply_veto(board: Any, result: Any, cnn: Any, frame: Any, region: Any) -> tuple[Any, list[Any]]:
    """占有maskを一切変えず、既存gravityが除いたセルを復元しない。"""
    records, out = [], result
    for r, c, original, replacement in candidates(board, result):
        row: dict[str, Any] = {'row': r, 'col': c, 'original': original, 'replacement': replacement,
                              'cnn_color': None if cnn is None else int(cnn.get(r, c))}
        if cnn is None or int(cnn.get(r, c)) != original:
            row['reason'] = 'cnn_missing_or_disagrees'
        elif frame is None or region is None:
            row['reason'] = 'image_or_region_missing'
        else:
            try:
                measured = measure(frame, region, r, c)
                if measured is None:
                    row['reason'] = 'patch_missing'
                else:
                    median, distances, saturation, distance = measured
                    row.update(hsv_medians=median, distances=distances,
                               minimum_saturation=saturation, maximum_distance=distance)
                    row['reason'] = support(original, median, distances, saturation, distance)
            except Exception as error:
                row.update(reason='measurement_failed', exception=repr(error))
        row['vetoed'] = row['reason'] == 'cnn_and_full_hsv_support_original'
        if row['vetoed']:
            if out is result:
                out = result.copy()
            out.set(r, c, original)
        records.append(row)
    return out, records


def bound_cnn(caller: Any, rec: Any, board: Any, frame: Any, region: Any) -> tuple[Any, dict[str, Any]]:
    require(caller.f_code in rec.journal.codes, 'palette_unknown_generated_caller')
    values = caller.f_locals
    pipe, side, ctx = values['self'], values['side'], values['ctx']
    require(pipe is rec.journal.tracker._pipeline and side in ('1P', '2P'), 'palette_pipe_side')
    require(ctx is getattr(pipe, '_sm_' + side.lower()).context and ctx.state.name == 'STABLE', 'palette_not_stable_ctx')
    require(board is ctx.confirmed_board and frame is values['frame_bgr']
            and region is values['region_for_validate'], 'palette_argument_identity')
    clock = values['frame_idx'], values['time_sec']
    require(clock == (rec.history.frame, rec.history.time_sec), 'palette_same_clock')
    require(rec.journal.active is not None and rec.journal.active['pipe'] is pipe
        and rec.journal.active['scope']['side'] == side, 'palette_active_scope')
    return values['cnn_board'], {'frame_idx': clock[0], 'time_sec': clock[1], 'side': side,
        'source_id': rec.journal.source_id, 'run_id': rec.journal.run_id}


class Recorder:
    """色変更候補だけ逐次記録し、画像/原frameは保持しない。"""
    def __init__(self, output: Path, history: Any, journal: Any) -> None:
        self.output, self.history, self.journal = output, history, journal
        self.stream = (output / SIDECAR).open('x', encoding='utf-8')
        self.calls, self.rows, self.vetoed = 0, 0, 0
        self.reasons: Counter[str] = Counter()
        self.closed = False

    def save(self, scope: dict[str, Any], cells: list[Any]) -> None:
        self.calls += 1
        if not cells:
            return
        value = scope | {'row_index': self.rows, 'cells': cells, 'call_index': self.calls - 1,
            'quality_gate_clear': False, 'physical_palette_certified': False}
        self.stream.write(encoded(value) + '\n')
        self.rows += 1
        self.vetoed += sum(c['vetoed'] for c in cells)
        self.reasons.update(c['reason'] for c in cells)

    def close(self) -> None:
        self.stream.close()
        self.closed = True


def wrapper(original: Any, rec: Recorder) -> Any:
    @functools.wraps(original)
    def validate(board: Any, next_queue: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(board, next_queue, *args, **kwargs)
        caller = sys._getframe(1)
        try:
            frame, region = kwargs.get('frame_bgr'), kwargs.get('region')
            cnn, scope = bound_cnn(caller, rec, board, frame, region)
            out, cells = apply_veto(board, result, cnn, frame, region)
            rec.save(scope, cells)
            return out
        finally:
            del caller
    return validate


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any], *, enabled: bool = False) -> Any:
    if not enabled:
        return None
    guards()
    require(KEY not in state, 'palette_already_installed')
    journal = state['atomic_journal_observer']
    cls = collector.RecognitionPipeline
    descriptor = inspect.getattr_static(cls, '_validate_next_history')
    require(isinstance(descriptor, staticmethod) and Path(inspect.getfile(descriptor.__func__)).resolve() == SOURCE,
            'palette_unknown_validator')
    rec = Recorder(Path(state['output']), history, journal)
    state[KEY] = rec
    stack.callback(rec.close)
    cls._validate_next_history = staticmethod(wrapper(descriptor.__func__, rec))
    stack.callback(setattr, cls, '_validate_next_history', descriptor)
    return rec


def finish(state: dict[str, Any]) -> None:
    rec = state[KEY]
    require(rec.closed and rec.calls > 0, 'palette_incomplete_or_zero_calls')
    write(rec.output / RECEIPT, {'closed': True, 'calls': rec.calls, 'rows': rec.rows,
        'vetoed_cells': rec.vetoed, 'reasons': dict(rec.reasons), 'guards': guards(),
        'sha256': {SIDECAR: sha(rec.output / SIDECAR)}, 'quality_gate_clear': False,
        'physical_palette_certified': False})
    verify(rec.output)


def verify_cell(cell: dict[str, Any]) -> None:
    require(type(cell['row']) is int and VISIBLE_FIRST_ROW <= cell['row'] < 13
        and type(cell['col']) is int and 0 <= cell['col'] < 6, 'palette_saved_coordinate')
    require(type(cell['vetoed']) is bool and type(cell['original']) is int and type(cell['replacement']) is int
        and cell['original'] in COLOR_CODES and cell['replacement'] in COLOR_CODES
        and cell['original'] != cell['replacement'], 'palette_saved_cell')
    require(cell['vetoed'] == (cell['reason'] == 'cnn_and_full_hsv_support_original'), 'palette_saved_decision')
    if cell['vetoed']:
        require(cell['cnn_color'] == cell['original']
            and cell['minimum_saturation'] == MINIMUM_SATURATION
            and cell['maximum_distance'] == MAXIMUM_COLOR_DISTANCE, 'palette_saved_policy')
        distances = {int(k): v for k, v in cell['distances'].items()}
        require(support(cell['original'], tuple(cell['hsv_medians']), distances,
            cell['minimum_saturation'], cell['maximum_distance']) == cell['reason'], 'palette_saved_numeric_support')


def verify(output: Path) -> dict[str, Any]:
    receipt = json.loads((output / RECEIPT).read_bytes())
    require(receipt['closed'] is True and type(receipt['calls']) is int and receipt['calls'] > 0, 'palette_saved_status')
    require(receipt['guards'] == guards() and receipt['sha256'] == {SIDECAR: sha(output / SIDECAR)}, 'palette_saved_sha')
    require(receipt['quality_gate_clear'] is False and receipt['physical_palette_certified'] is False, 'palette_permission')
    count, vetoed, reasons, previous = 0, 0, Counter(), -1
    with (output / SIDECAR).open() as stream:
        for line in stream:
            row = json.loads(line)
            require(type(row['row_index']) is int and row['row_index'] == count and type(row['call_index']) is int
                and previous < row['call_index'] < receipt['calls'], 'palette_saved_order')
            require(row['quality_gate_clear'] is False and row['physical_palette_certified'] is False, 'palette_row_permission')
            previous = row['call_index']
            for cell in row['cells']:
                verify_cell(cell)
                vetoed += cell['vetoed']
                reasons[cell['reason']] += 1
            count += 1
    require(count == receipt['rows'] and vetoed == receipt['vetoed_cells'] and dict(reasons) == receipt['reasons'], 'palette_saved_counts')
    return {'calls': receipt['calls'], 'rows': count, 'vetoed_cells': vetoed, 'quality_gate_clear': False}
