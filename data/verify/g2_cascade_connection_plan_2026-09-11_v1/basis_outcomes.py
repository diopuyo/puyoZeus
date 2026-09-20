"""保存された基準jointの原凍結cascade結果を集約し、人工時系列の設計入力にする。"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'))
import serialization as S

SOURCE = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v35/PROBABILISTIC_BASIS.jsonl'


def main() -> None:
    start = time.monotonic()
    raw = SOURCE.read_bytes()
    value = S.decode(json.loads(raw)['state'])
    groups = defaultdict(list)
    simulator = S.B.ChainSimulator(exclude_hidden_row_from_pop=True)
    for world in value.worlds:
        result = simulator.simulate(S.B.Board.from_dict({'grid': world.grid}))
        visible = S.B.grid(result.final_board)[S.B.HIDDEN_ROWS:]
        groups[(visible, result.chain_count)].append(world.weight)
    rows = [dict(visible=visible, chain_count=count, mass=math.fsum(weights), worlds=len(weights))
            for (visible, count), weights in groups.items()]
    result = dict(source=str(SOURCE), source_sha=hashlib.sha256(raw).hexdigest(),
        basis_frame=value.frame, worlds=len(value.worlds), outcomes=rows,
        seconds=time.monotonic() - start, actual_video=False, quality_gate_clear=False)
    with (ROOT / 'BASIS_OUTCOMES_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(worlds=len(value.worlds), visible_chain_groups=len(rows),
        chain_counts=sorted({row['chain_count'] for row in rows}), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
