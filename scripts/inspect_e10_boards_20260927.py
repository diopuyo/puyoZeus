"""固定が出た確定盤面の消去可能性を物理シミュレータで確認する。"""
from pathlib import Path
import json

from src.chain import ChainSimulator
from src.exchange_event_record import read_records
from src.indicators_v2 import near_future_fire_power
from src.production_config import GHOST_CHAIN_RULE_ENABLED

TIMES = ((2693.1833, 1), (2807.6166, 0), (3087.0833, 0))


def main() -> None:
    """非STABLE画像を使わず、各時刻より前の確定盤面を比較する。"""
    boards, done = [None, None], set()
    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for row in read_records(Path("logs/review_zenchi_part3/on_e9/inputs.jsonl.gz")):
        if row["kind"] != "update":
            continue
        result, _, _, t, *_ = row["args"]
        for i, side in enumerate((result.p1, result.p2)):
            if side.state.name == "STABLE" and side.confirmed_board is not None:
                boards[i] = (side.confirmed_board, side.next_pair, side.dnext_pair)
        for target, index in TIMES:
            if t < target or target in done:
                continue
            done.add(target)
            board, first, second = boards[index]
            resolved = sim.simulate(board)
            future = near_future_fire_power(resolved.final_board, first, second, k_levels=(1, 4))
            print(json.dumps(dict(t=t, side=index+1, dead=board.is_dead(),
                grid=board._grid.tolist(), resolved_dead=resolved.final_board.is_dead(),
                chains=resolved.chain_count, future={k: v.raw for k, v in future.values.items()})), flush=True)


if __name__ == "__main__":
    main()
