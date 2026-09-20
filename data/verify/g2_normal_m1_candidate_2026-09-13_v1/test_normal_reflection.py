"""左右共通通常型から元M1反映検査への接続。原J入力は人工で評価実走ではない。"""
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_normal_tracking as F
parts, modules = F.parts, F.modules
PUB = Path(__file__).resolve().parent.parent / 'g2_belief_live_publication_2026-09-11_v1'


def test_original_reflection_both_sides(parts: Any, modules: Any) -> None:
    imports = dict(journal_context=parts.context, serialization=parts.original.binding.S)
    original = parts.loader('_normal_reflection_original', PUB / 'reflection.py', imports)
    reflection = parts.loader('_normal_reflection_current', PUB / 'reflection_v3.py', imports |
                              dict(reflection=original, second_physical=modules.physical))
    f = F.setup(parts, modules)
    for side, mode in zip(('1P', '2P'), f.modes):
        binding = mode.connection.binding
        value = f.registry.current(binding)
        reflection.verify(mode, value, binding.initial_call_token, 10)
        F.step(f, mode, side, 12, [])
        reflection.verify(mode, value, 'step:' + side + ':12', 12)
        with pytest.raises(ValueError, match='initial_reflection_J'):
            reflection.verify(mode, value, 'foreign_call', 12)
        foreign = N(**vars(mode))
        with pytest.raises(ValueError, match='initial_reflection_mode'):
            reflection.verify(foreign, value, 'step:' + side + ':12', 12)
        board = getattr(f.pipe, '_sm_' + side.lower()).context.confirmed_board
        board.set(12, 3, 1)
        board.set(11, 3, 2)
        token = 'normal:fixture:reset:0:' + side + ':enqueue:hand1'
        events = [dict(stage='fifo_before', accounting=dict(pending_tsumo=[[1, 2]]),
                       fifo_occurrence_tokens=[token]),
                  dict(stage='fifo_after', accounting=dict(pending_tsumo=[]),
                       committed=[1, 2], enqueue_occurrence_token=token)]
        F.step(f, mode, side, 14, events)
        reflection.verify(mode, f.registry.current(binding), 'step:' + side + ':14', 14)
        assert len(mode.applied) == 1
