"""実保存32798の原補正を、同原画像・固定関数だけでCPU再現する。"""
from __future__ import annotations
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
LIVE = ROOT.parent / 'video38_atomic_journal_live_2026-09-09_v1'
SNAPSHOT = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
IMAGE = ROOT.parent / 'g2_first_inventory_break_2026-09-09_v1/physical_v1/source_32798.png'
FIXED_IMAGE = 'f24b1b79dfb01a91cf7ad3146715502eced0d55d0274bd57f4746b6404e2aaec'
TARGET_FRAME, TARGET_SIDE = 32798, '2P'
EXPECTED_SHAPE = (720, 1280, 3)
RUNTIME_SIZE = (1920, 1080)
FIXED_PIPELINE = '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def selected() -> tuple[Any, dict[str, str]]:
    complete = json.loads((LIVE / 'COMPLETE').read_bytes())
    require(type(complete['child_exit_code']) is int and complete['child_exit_code'] == 0, 'actual_failed')
    paths = ('atomic_journal.jsonl', 'ATOMIC_JOURNAL_STATUS.json', 'ATOMIC_JOURNAL_RECEIPT.json')
    guards = {str(LIVE / name): complete['sha256'][name] for name in paths}
    guards[str(LIVE / 'COMPLETE')] = sha(LIVE / 'COMPLETE')
    require(all(sha(Path(p)) == h for p, h in guards.items()), 'actual_changed')
    receipt = json.loads((LIVE / 'ATOMIC_JOURNAL_RECEIPT.json').read_bytes())
    status = json.loads((LIVE / 'ATOMIC_JOURNAL_STATUS.json').read_bytes())
    require(status == {'closed': True, 'errors': [], 'active': False}, 'journal_incomplete')
    source = SNAPSHOT / 'src/recognition_pipeline.py'
    require(sha(source) == receipt['guards'][str(source)], 'captured_source_changed')
    with (LIVE / 'atomic_journal.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row['kind'], row['frame_idx'], row['side']) == ('step', TARGET_FRAME, TARGET_SIDE):
                return row, guards
    raise ValueError('target_missing')


def partial_selected() -> tuple[Any, dict[str, str]]:
    """完了を待たず一行だけを読む。全run認証は発行しない。"""
    source = SNAPSHOT / 'src/recognition_pipeline.py'
    require(sha(source) == FIXED_PIPELINE, 'partial_source_changed')
    with (LIVE / 'atomic_journal.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row['kind'], row['frame_idx'], row['side']) == ('step', TARGET_FRAME, TARGET_SIDE):
                row['partial_line_sha256_without_newline'] = hashlib.sha256(line.rstrip('\n').encode()).hexdigest()
                return row, {str(source): FIXED_PIPELINE}
    raise ValueError('partial_target_not_reached')


def reproduce(row: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import numpy as np
    from src.board import Board
    from src.image_reader import DEFAULT_P2_REGION
    from src.recognition_pipeline import RecognitionPipeline as Pipeline
    from src import placement_inferrer as I
    require(Path(inspect.getfile(Pipeline)).resolve() == SNAPSHOT / 'src/recognition_pipeline.py', 'wrong_runtime')
    events = {e['stage']: e for e in row['events']}
    before, after = events['next_validation_before'], events['next_validation_after']
    frame = cv2.imread(str(IMAGE))
    require(frame is not None and frame.shape == EXPECTED_SHAPE, 'source_image_shape')
    frame = cv2.resize(frame, RUNTIME_SIZE)
    grid = before['confirmed']['grid']
    result = Pipeline._validate_next_history(Board.from_list(grid), before['next_queue'],
        ever_seen=set(before['ever_seen']), frame_bgr=frame, region=DEFAULT_P2_REGION,
        enable_starvation_fix=before['enable_starvation_fix'], min_colors_for_validation=before['min_colors'])
    expected = after['boards']['published_confirmed']['grid']
    require(result._grid.tolist() == expected, 'original_branch_does_not_reproduce_saved_writer')
    cells = []
    for r, values in enumerate(grid):
        for c, original in enumerate(values):
            if original == expected[r][c]:
                continue
            patch = I._extract_cell_patch_from_frame(frame, DEFAULT_P2_REGION, r, c)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            medians = tuple(int(np.median(hsv[:, :, n])) for n in range(3))
            distances = {str(k): I._hsv_distance(*medians, v) for k, v in I.COLOR_HSV_CENTERS.items()}
            cells.append({'row': r, 'col': c, 'before': original, 'after': expected[r][c],
                'hsv_medians': medians, 'all_color_distances': distances,
                'cnn_matches_input': before['boards']['cnn_board']['grid'][r][c] == original})
    return {'frame': TARGET_FRAME, 'side': TARGET_SIDE, 'queue': before['next_queue'],
        'ever_seen': before['ever_seen'], 'starvation_fix': before['enable_starvation_fix'],
        'min_colors': before['min_colors'], 'changed_cells': cells, 'original_result_matches_saved': True,
        'source_id': row['source_id'], 'run_id': row['run_id'], 'quality_gate_clear': False}


def main() -> int:
    output = ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    partial = '--partial' in sys.argv[2:]
    started = time.perf_counter()
    row, before = partial_selected() if partial else selected()
    before |= {str(IMAGE): FIXED_IMAGE, str(Path(__file__)): sha(Path(__file__))}
    before |= {str(SNAPSHOT / 'src' / name): sha(SNAPSHOT / 'src' / name)
               for name in ('recognition_pipeline.py', 'placement_inferrer.py', 'image_reader.py', 'board.py')}
    require(all(sha(Path(p)) == h for p, h in before.items()), 'probe_inputs_changed')
    require(not any(n == 'src' or n.startswith('src.') for n in sys.modules), 'src_loaded_early')
    sys.path.insert(0, str(SNAPSHOT))
    report = reproduce(row)
    report['actual_run_complete'] = not partial
    after = {p: sha(Path(p)) for p in before}
    require(before == after, 'source_changed_during_probe')
    write(output / 'REPRODUCTION.json', report)
    write(output / 'CAPTURED_ROW.json', row)
    result = {'pid': os.getpid(), 'actual_exit': 0, 'seconds': time.perf_counter() - started,
              'before': before, 'after': after, 'gpu_executed': False, 'quality_gate_clear': False,
              'actual_run_complete': not partial}
    write(output / 'RESULT.json', result)
    name = 'PROVISIONAL_RECEIPT.json' if partial else 'COMPLETE.json'
    write(output / name, {'sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
