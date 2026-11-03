"""実測690.866秒の連鎖前番号35と、再開後の履歴36以降を再現する。"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.ama_evaluation_v7_tape import CausalTape
from src.ama_tsumo_seed_v5 import Identification, merged_known
from src.exchange_tsumo_tape import _Side
from src.phase_j.live_ama_inputs import AmaInputs

FIXTURE = Path(__file__).parent/'fixtures/ama_missing_pinned_history.json'


def recorded_tape() -> CausalTape:
    record = json.loads(FIXTURE.read_text(encoding='utf-8'))
    tape = CausalTape(None, inherit_counter=True)
    tape.tape.sides = [_Side(**s) for s in record['sides']]
    tape.tape.aligned = True
    tape.pinned = record['pinned']
    tape.identify = lambda *_: Identification('too_short')
    return tape


def test_recorded_missing_number_becomes_explicit_missing() -> None:
    tape = recorded_tape()
    known = merged_known(dict(counters=[s.base for s in tape.tape.sides],
                              tapes=[s.own for s in tape.tape.sides]))
    assert min(known) == 36 and tape.pinned[0] == 35
    with pytest.raises(KeyError, match='35'):
        [known[i] for i in range(tape.pinned[0], max(known)+1)]
    with pytest.raises(ValueError, match='連鎖前ツモ番号の観測履歴不足'):
        tape.bundle([1, 3, 4, 5])


def test_unpinned_observations_recover_without_changing_history() -> None:
    tape = recorded_tape()
    tape.pinned = [None, None]
    # 種未特定で3組不足の別の欠測も、そのまま維持する。
    with pytest.raises(ValueError, match='開始3組不足'):
        tape.bundle([1, 3, 4, 5])
    tape.pinned = [37, 37]
    result = tape.bundle([1, 3, 4, 5])
    assert result['counters'] == [37, 37]
    assert result['input_known'] == [t[:3] for t in result['tapes']]


def test_request_keeps_worker_alive_with_missing_reason() -> None:
    tape = recorded_tape()
    tape.last = 690.8666666666667
    snap = dict(evidence_times=[tape.last], boards=[[0]*78]*2, pending=[0, 0],
                carry=[0, 0], all_clear=[False, False], kinds=['chain', 'stable'])
    inputs = SimpleNamespace(tape=tape, index=0, colors={1, 3, 4, 5},
        projection=SimpleNamespace(states=['CHAIN', 'STABLE'], snapshot=lambda _: snap))
    item = AmaInputs.request(inputs, 1, tape.last, dict(origin=638.5333333333333))
    assert item['missing'] == '連鎖前ツモ番号の観測履歴不足'
    assert item['request']['known'] == [] and 'bundle' not in item
