"""保存Jの同世代終了付近を元consumerへ再生し、参考rootと現在供給資格を区別する。"""
import io
import json
from pathlib import Path
import time
from src import chain_prediction_ledger_v1 as L
from journal_origin_capture import OriginCapture
from settled_notice import is_settled
from probe_opponent_projection import rows
import projected_view as P

ROOT = Path(__file__).resolve().parent
FIRST, LAST, AVAILABLE = 34702, 35812, 35160
SELECTED = (34702, 35160, 35672, 35786, 35808, 35810, 35812)


def main() -> None:
    started, results = time.perf_counter(), []
    capture = OriginCapture(None, None, None, L, P.B.ChainSimulator(exclude_hidden_row_from_pop=True),
                            P.B.Board.from_dict, is_settled, io.StringIO())
    for row in rows('atomic_journal.jsonl'):
        if row.get('kind') != 'step' or row['side'] != '2P' or not FIRST <= row['frame_idx'] <= LAST:
            continue
        result = capture.consume(row, available_frame=AVAILABLE if row['frame_idx'] <= AVAILABLE else None)
        if row['frame_idx'] not in SELECTED:
            continue
        returned = row['returned'] or {}
        raw = returned.get('active_origin')
        handle = capture.handles.get('2P')
        snapshot = None if handle is None else capture.ledger.snapshot(handle)
        results.append(dict(frame=row['frame_idx'], state=returned.get('state'),
            raw_mechanism=None if raw is None else raw['mechanism'], settled=raw is not None and is_settled(raw),
            instance_id=result.get('instance_id'), root_count=None if result['origin'] is None else result['origin']['chain_count'],
            generation=row['generation_after'], episodes=0 if snapshot is None else len(snapshot.episodes),
            landing_signatures=[] if snapshot is None else sorted(set((e.trigger_sec, e.before_sha256)
                for e in snapshot.episodes if e.mechanism == 'landing'))))
    indexed = {row['frame']: row for row in results}
    assert indexed[35808]['settled'] and indexed[35808]['root_count'] == 13
    assert indexed[35808]['episodes'] == indexed[35786]['episodes']
    assert indexed[35672]['root_count'] == 13 and indexed[35786]['raw_mechanism'] is None
    capture.close()
    result = dict(boundaries=results, settled_not_added_as_episode=True, root_presence_is_not_supply_permission=True,
        original_J_live_hook_verified=False, physical_identity_verified=False, quality_gate_clear=False,
        seconds=time.perf_counter()-started)
    with (ROOT / 'ROOT_SUPPLY_BOUNDARIES_v1.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
