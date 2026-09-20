"""保存原票の前史→live相当後続をCPU比較。新runの所有や物理終了の証明ではない。"""
import io
import json
from pathlib import Path
import time
from src import chain_prediction_ledger_v1 as L
from probe_existing_origin_ledger import selected
from journal_origin_capture import OriginCapture
from settled_notice import is_settled
import projected_view as P

ROOT = Path(__file__).resolve().parent
AVAILABLE_FRAME = 35160


def main() -> None:
    started, first, historical, live = time.perf_counter(), None, 0, 0
    value = OriginCapture(None, None, None, L, P.B.ChainSimulator(exclude_hidden_row_from_pop=True),
        P.B.Board.from_dict, is_settled, io.StringIO())
    for row in selected():
        past = row['frame_idx'] <= AVAILABLE_FRAME
        result = value.consume(row, available_frame=AVAILABLE_FRAME if past else None)
        historical, live = historical + int(past), live + int(not past)
        origin = result['origin']
        if first is None: first = origin
        assert origin == first
        assert origin['available_at']['frame_idx'] == AVAILABLE_FRAME
    assert (first['chain_count'], first['calculated_total_score']) == (13, 79080)
    snapshot = value.ledger.snapshot(value.handles['2P'])
    result = dict(historical_rows=historical, following_rows=live, origin_count=first['chain_count'],
        origin_score=first['calculated_total_score'], available_frame=AVAILABLE_FRAME,
        recorded_origin_frame=34702, episodes=len(snapshot.episodes),
        first_provenance=snapshot.episodes[0].capture_source, root_fixed=True,
        same_run_live_history_capture_verified=False, physical_identity_verified=False,
        quality_gate_clear=False, seconds=time.perf_counter() - started)
    value.close()
    with (ROOT / 'SAVED_HISTORY_AVAILABILITY_v1.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
