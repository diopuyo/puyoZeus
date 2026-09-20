"""保存原票で参考資格の内容条件だけ再生。read入口のlive J認証は検証しない。"""
import io
import json
from pathlib import Path
import time
from src import chain_prediction_ledger_v1 as L
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
from journal_origin_capture import OriginCapture
from settled_notice import is_settled
from probe_opponent_projection import rows
import projected_view as P
import fixed_origin_reference as F

ROOT = Path(__file__).resolve().parent
FIRST, LAST, AVAILABLE = 34702, 35812, 35160
SELECTED = (35160, 35672, 35786, 35808, 35810, 35812)


def main() -> None:
    started, packets, counts = time.perf_counter(), [], {}
    capture = OriginCapture(None, None, None, L, P.B.ChainSimulator(exclude_hidden_row_from_pop=True),
                            P.B.Board.from_dict, is_settled, io.StringIO())
    for row in rows('atomic_journal.jsonl'):
        frame = row.get('frame_idx', -1)
        if row.get('kind') != 'step' or row['side'] != '2P' or not FIRST <= frame <= LAST:
            continue
        capture.consume(row, available_frame=AVAILABLE if frame <= AVAILABLE else None)
        if frame < AVAILABLE: continue
        snapshot = F.snapshot_for(capture, row, '2P')
        assert snapshot is not None
        reason = F.qualification(capture, row, snapshot, '2P')
        before = capture.ledger.snapshot(capture.handles['2P'])
        value = F.packet('2P', frame, reason or 'software_episode_reference_not_physical_binding',
                         None if reason else _origin_prediction(snapshot), snapshot.handle.instance_id)
        assert capture.ledger.snapshot(capture.handles['2P']) == before
        counts[value['status']] = counts.get(value['status'], 0) + 1
        if frame in SELECTED:
            packets.append(dict(frame=frame, status=value['status'], reason=value['reason'],
                root_count=None if value['origin'] is None else value['origin']['chain_count']))
    assert packets[0]['root_count'] == packets[1]['root_count'] == 13
    assert all(value['status'] == 'HOLD' for value in packets[2:])
    capture.close()
    result = dict(counts=counts, selected=packets, original_J_live_gate_exercised=False,
        original_consumer_and_content_conditions=True, quality_gate_clear=False,
        seconds=time.perf_counter()-started)
    with (ROOT / 'SAVED_REFERENCE_CUTOFFS_v1.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
