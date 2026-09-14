"""2P実台帳＋人工tap/stepで全終端保存を検査。旧数学consumerはstubと明示。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as N
import pytest
import second_inactive_saved as C
import test_second_inactive_candidate as F


def fixture(path: Path) -> tuple:
    with ExitStack() as stack:
        end, item, sources = F.setup(path, stack)
        stack.push(end.close)
        for tick in range(end.ledger.clock+2, end.ledger.deadline+2, 2):
            if tick != sources[0]['frame_idx']:
                row = deepcopy(sources[0])
                row.update(frame_idx=tick, time_sec=tick/60, token=f'synthetic-enqueue:{tick}',
                    update_begin_accounting={'pending_tsumo':[]})
                sources.append(row)
            item['scope']['frame_idx'] = item['frame'].f_locals['frame_idx'] = tick
            item['frame'].f_locals['time_sec'] = tick/60
            item['token'] = f'synthetic-step:{tick}'
            end.observe(item, None)
    packets = [json.loads(line) for line in (path/'SECOND_INACTIVE_END.jsonl').read_text().splitlines()]
    status = json.loads((path/'SECOND_INACTIVE_END_STATUS.json').read_bytes())
    steps = {end.ledger.clock:dict(frame_idx=end.ledger.clock, row_index=-1,
        kind='step', side='2P', code_sha256='synthetic-code')}
    for index, (row, packet) in enumerate(zip(sources, packets, strict=True)):
        row['row_index'] = index*2
        steps[row['frame_idx']] = deepcopy(row) | dict(kind='step', row_index=index*2+1,
            token=packet['completed_call_token'], code_sha256='synthetic-code', exception=None,
            generation_after=packet['generation_after'], events=[])
    old = N(steps=steps, native=N(extract=lambda _:None), pending=[],
        contexts={end.ledger.clock:F.F.actual_end_context()})
    return end, sources, packets, status, old


@pytest.mark.parametrize('mutation', ['none', 'missing', 'status_error', 'old_failure'])
def test_original_prefix_then_complete_tail(tmp_path: Path, mutation: str) -> None:
    end, sources, packets, status, old = fixture(tmp_path)
    calls = []
    class Previous:
        def run(self, records: list, cutoff: int) -> dict:
            calls.append(cutoff)
            if mutation == 'old_failure':
                raise ValueError('old-prefix-failure')
            return dict(end=cutoff, old_math_stub=True)
    if mutation == 'missing': packets.pop()
    elif mutation == 'status_error': status['error'] = 'original-close-failure'
    selected = C.derive(N(Replay=Previous), packets, status)
    replay = selected()
    replay.terminal = [{'kind':'finish', 'ledger':{'clock':end.ledger.clock}}]
    replay.ledger, replay.L, replay.old = end.ledger, end.value.L, old
    replay.source = {row['frame_idx']:row for row in sources}
    replay.pending_timeline = {end.ledger.clock:()}
    if mutation != 'none':
        with pytest.raises(ValueError): replay.run([], end.ledger.deadline)
        return
    report = replay.run([], end.ledger.deadline)
    assert calls == [36884] and report['end'] == 36900 and report['old_math_stub']
    assert report['inactive_terminal']['original_terminal_updates'] == 8
    assert len(replay.ledger.acknowledgements) == 10 and len(replay.pending_timeline) == 9
