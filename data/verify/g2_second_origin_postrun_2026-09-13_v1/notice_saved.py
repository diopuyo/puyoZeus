"""終了通知別票を原Jの全対象行から独立再導出する。物理対応を認証しない。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterator

FILENAME = 'SECOND_SETTLED_NOTICES.jsonl'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('second_notice_saved:' + reason)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def expected(step: dict[str, Any], initial: dict[str, Any]) -> dict | None:
    notices = [dict(stage=e['stage'], origin=e['active_origin']) for e in step['events']
               if e.get('active_origin') is not None and e['active_origin'].get('mechanism') == 'baseline']
    if not notices:
        return None
    return dict(kind='second_settled_notice/v1', scope=initial['state']['scope'],
        initial_call_token=initial['source_call_token'], source_call_token=step['token'],
        frame=step['frame_idx'], notices=notices, classification='CHAIN_SETTLED',
        physical_identity_certified=False, operation_token_assigned=False,
        next_consumed=False, probability_updated=False, quality_gate_clear=False)


def match_scope(step: dict[str, Any], wanted: list) -> bool:
    # 原JのgenerationはSM idを持たない版もある。identityは元source/pipe/side/epochで照合。
    return (step['source_id'], step['run_id'], step['software_reset'], step['pipe_object_id'],
            step['generation_after']['reset_epoch'], step['side']) == tuple(wanted[i] for i in (0,1,2,3,5,6))


def verify_rows(saved: list[dict], journal: Any, modes: list[dict], end: int) -> dict:
    wanted, identities = [], {}
    for step in journal:
        if step.get('kind') != 'step' or step['side'] != '2P':
            continue
        eligible = [m for m in modes if match_scope(step, m['initial']['state']['scope'])
            and m['initial']['state']['frame'] < step['frame_idx']
            and step['frame_idx'] < (m['retired']['frame'] if m.get('retired') else end + 1)]
        require(len(eligible) <= 1, 'ambiguous_owner')
        if not eligible:
            continue
        require(step['status'] == 'returned' and step['exception'] is None, 'original_step_error')
        record = expected(step, eligible[0]['initial'])
        if record is not None:
            for notice in record['notices']:
                origin = notice['origin']
                key = (record['initial_call_token'], origin['object_id'], origin['trigger_sec'])
                grid = origin['before_board']['grid']
                require(key not in identities or identities[key] == grid, 'origin_mutated')
                identities[key] = grid
            wanted.append(record)
    require(saved == wanted, 'full_source_notice_mismatch')
    return dict(notice_steps=len(wanted), original_J_matched=True, physical_identity_certified=False,
                operation_token_assigned=False, quality_gate_clear=False)


def verify(output: Path, end: int) -> dict:
    session = json.loads((output / 'BELIEF_M1_SESSION.json').read_bytes())
    require(session['error'] is None and session['session_error'] is None, 'session_error')
    modes = session['modes']
    require(all(m['closed'] and m['error'] is None for m in modes), 'mode_error_or_open')
    return verify_rows(list(rows(output / FILENAME)), rows(output / 'atomic_journal.jsonl'), modes, end)
