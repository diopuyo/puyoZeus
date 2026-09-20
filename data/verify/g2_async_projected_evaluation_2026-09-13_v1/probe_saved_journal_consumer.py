"""新consumerの台帳経路をA22原票へ適用する。live所有検証の代用ではない。"""
import io
import json
from pathlib import Path
from src import chain_prediction_ledger_v1 as L
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
from probe_existing_origin_ledger import selected
from journal_origin_capture import OriginCapture
from settled_notice import is_settled
import projected_view as P

ROOT = Path(__file__).resolve().parent


def main() -> None:
    records = selected()
    value = OriginCapture(None, None, None, L, P.B.ChainSimulator(exclude_hidden_row_from_pop=True),
                          P.B.Board.from_dict, is_settled, io.StringIO())
    first, final = None, None
    for row in records:
        final = value.consume(row)
        if first is None:
            first = final['origin']
        assert final['origin'] == first
    snapshot = value.ledger.snapshot(value.handles['2P'])
    origin = _origin_prediction(snapshot)
    result = dict(source='video38_split_tail_candidate_v22', rows=len(records),
        episodes=len(snapshot.episodes), predictions=len(snapshot.predictions),
        origin_count=origin.chain_count, origin_score=origin.calculated_total_score,
        origin_fixed_for_all_rows=True, live_witness_exercised=False, model_evaluated=False,
        physical_identity_verified=False, quality_gate_clear=False)
    value.close()
    assert value.ledger is None and value.pipe is None and value.witness is None
    result['owned_references_released'] = True
    with (ROOT / 'SAVED_JOURNAL_CONSUMER_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
