"""2Pの先行起点と後続起点を物理比較する。root同一性や火力を確定しない。"""
from dataclasses import replace
import json
from pathlib import Path
import projected_view as P
from probe_opponent_projection import rows

ROOT = Path(__file__).resolve().parent
TOKENS = ('step:5815', 'step:6037', 'step:6621')


def main() -> None:
    selected = [row for row in rows('atomic_journal.jsonl') if row.get('token') in TOKENS]
    assert len(selected) == len(TOKENS)
    results, simulations = [], []
    for row in selected:
        origins = [e['active_origin'] for e in row['events'] if e.get('active_origin')]
        origin = origins[0]
        assert all(o['before_board'] == origin['before_board'] for o in origins)
        board = P.B.Board.from_dict(origin['before_board'])
        assert all(c in P.B.PROB_COLORS for line in P.B.grid(board) for c in line)
        sim = P.B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(board)
        simulations.append(sim)
        results.append(dict(token=row['token'], frame=row['frame_idx'], origin_id=origin['object_id'],
            trigger_sec=origin['trigger_sec'], detector_count=origin['chain_count'],
            detector_score=origin['total_score'], simulated_count=sim.chain_count,
            simulated_raw_score=P.calculate_chain_score(sim).total_score,
            source_grid=P.B.grid(board), final_grid=P.B.grid(sim.final_board)))
    first, later = results[:2]
    value = dict(origins=results, final_boards_equal=first['final_grid'] == later['final_grid'],
        root_identity_verified=False, temporal_probability_propagated=False,
        detector_or_simulator_certified=False, quality_gate_clear=False)
    with (ROOT / 'OPPONENT_ROOT_RELATION_v1.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(dict(origins=[{k: v for k, v in r.items() if 'grid' not in k} for r in results],
        final_boards_equal=value['final_boards_equal'], quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
