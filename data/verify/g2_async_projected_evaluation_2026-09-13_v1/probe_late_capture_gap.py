"""実Session開始後だけの原票で固定起点が取れるか検査。過去票をliveへ補完しない。"""
import io
import json
from pathlib import Path
import time
from typing import Any, Iterator
from src import chain_prediction_ledger_v1 as L
from journal_origin_capture import OriginCapture
from settled_notice import is_settled
import projected_view as P

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'video38_split_tail_candidate_v22'
FIRST, LAST, SIDE = 35162, 35672, '2P'


def rows() -> Iterator[dict]:
    with (SOURCE / 'atomic_journal.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') != 'step': continue
            if row['frame_idx'] > LAST: break
            if row['side'] == SIDE and row['frame_idx'] >= FIRST: yield row


def main() -> None:
    started = time.perf_counter()
    value = OriginCapture(None, None, None, L,
        P.B.ChainSimulator(exclude_hidden_row_from_pop=True), P.B.Board.from_dict, is_settled, io.StringIO())
    count, origins, reasons, first, last = 0, set(), set(), None, None
    for row in rows():
        packet = value.consume(row)
        origin = packet.get('origin')
        if origin is not None:
            origins.add((origin['chain_count'], origin['calculated_total_score']))
        reasons.update(packet.get('reasons', []))
        raw = (row.get('returned') or {}).get('active_origin')
        brief = dict(frame=row['frame_idx'], returned=None if raw is None else {
            key: raw[key] for key in ('object_id', 'trigger_sec', 'mechanism', 'chain_count')},
            instance=packet.get('instance_id'), supplied=origin is not None)
        if first is None: first = brief
        last, count = brief, count + 1
    assert count == (LAST - FIRST) // 2 + 1
    value.close()
    result: dict[str, Any] = dict(rows=count, first=first, last=last, origins=sorted(origins),
        reasons=sorted(reasons), original_root=(13, 79080), original_root_available=(13, 79080) in origins,
        source='same_saved_A22_rows_after_actual_Session_start', live_owner_verified=False,
        quality_gate_clear=False, seconds=time.perf_counter() - started)
    with (ROOT / 'LATE_CAPTURE_GAP_v1.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
