"""SM更新直後ではなく、原Jの公開直前・直後のSMと資格票を厳格突合する。"""
from __future__ import annotations

import old_stable as OLD

BASE, L = OLD.BASE, OLD.L
capture, validate, grid = OLD.capture, OLD.validate, OLD.grid
# 凍結recognition_pipeline.pyの_classify_board_none_reason前後。原J anchorsと対応。
PUBLICATION_STAGES = ('publication_before', 'publication_after')
PUBLICATION_LINES = (8632, 8637)


def publication(step: dict) -> dict:
    events = [e for e in step['events'] if e['stage'] in PUBLICATION_STAGES]
    L.require(tuple(e['stage'] for e in events) == PUBLICATION_STAGES, 'stable_publication_coverage')
    L.require(tuple(e['line'] for e in events) == PUBLICATION_LINES, 'stable_publication_lines')
    L.require(all(e['state'] == 'STABLE' and e.get('confirmed') is not None for e in events),
              'stable_publication_state')
    before, after = (e['confirmed'] for e in events)
    L.require(before == after, 'stable_publication_mutation')
    return after


def verify(row: dict, step: dict, context: dict) -> bool:
    value = row['stable_qualification']
    validate(value)
    L.require(value['source_call_token'] == step['token'] == row['journal_token'], 'stable_saved_call')
    for key in ('frame_idx', 'time_sec', 'side', 'source_id', 'run_id', 'pipe_object_id', 'generation'):
        L.require(value['scope'][key] == step[key], 'stable_saved_scope')
    L.require(value['software_reset'] == step['software_reset'], 'stable_saved_epoch')
    L.require(step['status'] == 'returned' and step['exception'] is None
              and step['returned']['state'] == 'STABLE'
              and value['returned'] == step['returned']['confirmed']['grid'], 'stable_saved_returned')
    L.require(all(context[k] == step[k] for k in ('source_id', 'run_id', 'frame_idx', 'time_sec')),
              'stable_context_scope')
    side = context['sides'][step['side']]
    pairs = (('raw', side['pb']['raw']), ('cnn', side['sm']['input_cnn']),
             ('sm', publication(step)), ('returned', side['before_hold']['confirmed']))
    L.require(all(value[k] == saved['grid'] for k, saved in pairs), 'stable_context_channels')
    return True
