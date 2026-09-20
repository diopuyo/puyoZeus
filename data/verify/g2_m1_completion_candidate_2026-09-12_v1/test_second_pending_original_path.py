"""元2P物理/Registryの人工正常手と退役から出る実原票をreplayへ接続する。"""
from __future__ import annotations

import json
from typing import Any

import pytest
from test_second_basis import saved, live, context, policy, observed
from test_second_binding_v2_path import physical
import test_second_physical as NORMAL
import test_second_retirement as RETIRE
import second_pending_replay as R


@pytest.mark.parametrize('case,flag', [('normal', False), ('normal', True), ('retire', False), ('retire', True)])
def test_original_mode_pending_matches_saved_replay(observed: Any, policy: Any, physical: Any,
                                                    monkeypatch: Any, case: str, flag: bool) -> None:
    owner = NORMAL if case == 'normal' else RETIRE
    original, modes = owner.setup, []

    def setup(*args: Any) -> Any:
        value = original(*args)
        modes.append(value)
        return value

    monkeypatch.setattr(owner, 'setup', setup)
    if case == 'normal': NORMAL.test_normal_hand_once(observed, policy, physical, flag)
    else: RETIRE.test_retire_preserve_and_rebind(observed, policy, physical, flag)
    assert len(modes) == 1
    mode = modes[0]
    rows = [json.loads(line) for line in observed.witness.journal.stream.getvalue().splitlines()]
    rows = [r for r in rows if r.get('kind') == 'step' and r['side'] == '2P']
    steps = {r['frame_idx']: r for r in rows}
    assert len(steps) == len(rows)
    packet = dict(initial=mode.initial_receipt, retired=mode.retired_receipt, applied=mode.applied)
    packet = json.loads(json.dumps(packet))  # 実ファイルと同じtuple→list変換を通す。
    timeline = R.timeline(packet, steps, observed.pipe._sm_2p.context.frame_idx)
    actual = tuple(v.occurrence_token for v in mode.native.pending)
    assert timeline[max(timeline)] == actual
    assert any(timeline.values()) is (flag is True)
