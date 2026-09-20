"""閉じた実処理の検収用registryのcopyだけを差し替え、原盤面/stateは変えない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from typing import Any
import completion as C


def verify(factory: Any, E: Any, journal: Any, history: Any, step: Any) -> dict[str, Any]:
    control = factory.controller
    rows, refs = control.private_suffix_completion_rows, control.private_suffix_completion_refs
    before = C.verify(factory, E, journal, history, step)
    rejected = []
    for case in ('missing_receipt', 'payload', 'foreign_factory', 'proof', 'policy', 'missing_start'):
        copied, copied_refs, saved = list(rows), dict(refs), history
        receipt = rows[0]
        values = list(copied_refs[receipt.sha256])
        if case == 'missing_receipt':
            copied = []
        elif case == 'payload':
            copied[0] = replace(receipt, payload_json='{}')
        elif case == 'foreign_factory':
            values[0] = object()
        elif case == 'proof':
            values[4] = deepcopy(values[4])
            values[4]['previous_private_placement'] = 'foreign'
        elif case == 'policy':
            values[5] = deepcopy(values[5])
            values[5][0]['purpose'] = 'foreign'
        else:
            saved = [r for r in history if not (r['scope']['frame_idx'] == values[2].started[0]
                and r['scope']['side'] == values[2].scope[-1])]
        copied_refs[receipt.sha256] = tuple(values)
        control.private_suffix_completion_rows, control.private_suffix_completion_refs = copied, copied_refs
        try:
            C.verify(factory, E, journal, saved, step)
        except AssertionError:
            rejected.append(case)
        else:
            raise AssertionError('completion_negative_accepted:'+case)
        finally:
            control.private_suffix_completion_rows, control.private_suffix_completion_refs = rows, refs
    assert len(rejected) == 6 and C.verify(factory, E, journal, history, step) == before
    return dict(positive=before, rejected=rejected, actual_board_state_unchanged=True)
