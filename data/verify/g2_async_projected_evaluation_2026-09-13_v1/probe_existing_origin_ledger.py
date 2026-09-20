"""A22原Jを既存予測台帳へ再生し、先頭originを後続episodeから保護する。"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from src.chain_prediction_ledger_v1 import ChainPredictionLedger, ChainGeneration
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
import projected_view as P
from probe_opponent_projection import rows

ROOT = Path(__file__).resolve().parent
FIRST, LAST, SIDE = 34702, 35672, '2P'


def selected() -> list[dict]:
    result = [r for r in rows('atomic_journal.jsonl') if r.get('kind') == 'step'
              and r['side'] == SIDE and FIRST <= r['frame_idx'] <= LAST]
    assert [r['frame_idx'] for r in result] == list(range(FIRST, LAST + 1, 2))
    identity = ('source_id', 'run_id', 'software_reset', 'pipe_object_id')
    first = tuple(result[0][key] for key in identity)
    generation = result[0]['generation_after']
    assert all(tuple(r[key] for key in identity) == first and r['generation_after'] == generation
               and r['exception'] is None and r['status'] == 'returned' for r in result)
    return result


def event(raw: dict) -> SimpleNamespace:
    names = ('trigger_sec', 'end_sec', 'chain_count', 'total_score', 'mechanism', 'score_estimated')
    return SimpleNamespace(**{key: raw[key] for key in names},
                           before_board=P.B.Board.from_dict(raw['before_board']))


def replay(records: list[dict]) -> tuple:
    ledger, seen, handle, origin_reference = ChainPredictionLedger(), set(), None, None
    source = records[0]['generation_after']
    generation = ChainGeneration(SIDE, source['reset_epoch'], source['action_revision'])
    for row in records:
        for item in row['events']:
            raw = item.get('active_origin')
            if raw is None or raw['object_id'] in seen:
                continue
            seen.add(raw['object_id'])
            value, point = event(raw), dict(frame_idx=row['frame_idx'], time_sec=row['time_sec'])
            if handle is None:
                assert row['frame_idx'] == FIRST and value.mechanism == 'landing'
                handle = ledger.open_landing_provisional(generation=generation, **point,
                    origin_before_board=value.before_board, landing_event=value,
                    capture_source='saved_A22_J_offline_not_live_owner')
                episode = ledger.snapshot(handle).episodes[0]
            else:
                episode = ledger.add_episode(handle, generation=generation, **point, event=value,
                    capture_source='saved_A22_J_offline_not_live_owner')
            prediction = P.B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(value.before_board)
            ledger.add_prediction(handle, generation=generation, **point,
                episode_revision=episode.episode_revision, input_board=value.before_board, result=prediction)
            fixed = _origin_prediction(ledger.snapshot(handle))
            if origin_reference is None:
                origin_reference = fixed
            assert fixed == origin_reference
    return ledger.snapshot(handle), origin_reference


def main() -> None:
    records = selected()
    snapshot, origin = replay(records)
    result = dict(journal_rows=len(records), origin_frame=FIRST, cutoff_frame=LAST,
        origin_reference=asdict(origin), episodes=len(snapshot.episodes),
        prediction_scopes=[p.scope.value for p in snapshot.predictions],
        origin_never_replaced=True, original_selector_reused=True, original_ledger_reused=True,
        software_generation_only=True, physical_identity_verified=False, live_owner_verified=False,
        accounting_permission=False, quality_gate_clear=False)
    with (ROOT / 'EXISTING_ORIGIN_LEDGER_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(rows=len(records), episodes=len(snapshot.episodes),
        origin_score=origin.calculated_total_score, origin_count=origin.chain_count,
        origin_never_replaced=True, quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
