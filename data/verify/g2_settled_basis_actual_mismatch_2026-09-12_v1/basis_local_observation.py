"""原step完了時の未保存値を読む診断部品。確率やSMを変更しない。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable

REQUIRED = ('signals', 'sm', 'side_prob_board', 'publish_prob_board', '_hsv_board_for_signals')
FLAGS = ('_enable_stable_recovery_gate', '_recovery_min_frames', '_enable_stable_resume_gate',
         '_empty_to_color_min_votes', '_enable_asymmetric_recovery_min_frames',
         '_enable_next_corroborated_confirm', '_enable_warmup_guard', '_enable_cnn_flicker_hsv_fallback')


def capture(local: dict[str, Any], result: Any, typed: dict[str, Any],
            snapshot: Callable[[Any], Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED if key not in local]
    if missing:
        raise ValueError('basis_local_missing:' + ','.join(missing))
    signals, sm = local['signals'], local['sm']
    published = local['publish_prob_board']
    if result.prob_board is not published or signals.hsv_board is not local['_hsv_board_for_signals']:
        raise ValueError('basis_local_return_or_signal_changed')
    flags = {key: getattr(sm, key) for key in FLAGS}
    branch = ('none' if published is None else
              'confirmed_fallback' if local['side_prob_board'] is None else 'side_probability')
    return dict(probability_branch=branch, branch_basis='original_step_locals_after_return',
        published_is_original_result=True, fallback_input_captured=False,
        confirmed_after_is_fallback_input_proof=False,
        original_returned_distribution=deepcopy(typed['probability']), flags=flags,
        hsv_board=snapshot(signals.hsv_board), cnn_board=snapshot(signals.cnn_board),
        confirmed_after=snapshot(sm.context.confirmed_board),
        non_stable_history_after=[snapshot(value) for value in sm.context.non_stable_cnn_history],
        warmup_remaining=sm.context.stable_warmup_remaining,
        scope_qualified=False, physical_certified=False, quality_gate_clear=False)
