"""事前登録された構成だけを選び、旧モードを変えないことを確認する。"""
import json
import sys
from pathlib import Path

import pytest

from scripts.falsedeath_replay_20261001 import enabled_flags, input_record, V2_FLAGS


def test_v2b_excludes_both_d2_flags() -> None:
    assert enabled_flags('v2b') == ['--death-input-integrity', '--score-attribution-v2',
                                   '--post-counter-current-puyo-integrity']
    assert input_record('v2b','review') == input_record('all','review')


def test_v2_configuration_is_unchanged() -> None:
    assert enabled_flags('v2') == list(V2_FLAGS)


def test_old_configurations_are_unchanged() -> None:
    flags = ['--death-input-integrity','--post-counter-input-integrity','--score-attribution']
    assert enabled_flags('prod') == []
    assert enabled_flags('all') == flags
    assert enabled_flags('no_d4') == flags
    for index in range(1,4):
        assert enabled_flags(f'no_d{index}') == [f for i,f in enumerate(flags,1) if i!=index]


@pytest.mark.parametrize('action', ['replay','score'])
@pytest.mark.parametrize('mode', ['v2b', 'v2c'])
def test_v2b_failed_ladder_prevents_full_run(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path, action: str, mode: str) -> None:
    from scripts import falsedeath_replay_20261001 as replay
    from scripts import falsedeath_v2_score_20261002 as score
    (tmp_path/mode).mkdir()
    (tmp_path/mode/'LADDER.json').write_text(json.dumps(dict(passed=False)))
    monkeypatch.setattr(replay,'OUT',tmp_path)
    monkeypatch.setattr(score.prior,'OUT',tmp_path)
    monkeypatch.setattr(score,'MODE',score.MODE)
    monkeypatch.setattr(score,'DEST',score.DEST)
    monkeypatch.setattr(sys,'argv',['replay',mode,'review'])
    with pytest.raises(AssertionError,match='短区間の停止条件'):
        replay.main() if action=='replay' else score.main(mode)
    assert not (tmp_path/'replay').exists()
    assert not (tmp_path/mode/f'RESULT_{mode}.json').exists()
