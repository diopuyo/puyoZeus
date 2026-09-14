"""原式発火入口の実callerを読取り、元infer/resolveを私有コピーで検査する。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import sys
from typing import Any

SIDE, STRIDE, FPS = '1P', 2, 60


def key(board: Any) -> Any:
    return None if board is None else tuple(tuple(r) for r in board.to_dict()['grid'])


def infer(values: Any, pipe: Any, env: Any, before: Any, observed: Any, pair: Any) -> dict[str, Any]:
    deferred: list[Any] = []
    frame = values['frame']
    inferred = env['infer_placement'](before.copy(), observed.copy(), tuple(pair),
        chain_sim=pipe._chain_sim, score_delta_observed=0, frame_bgr=frame,
        region=env['DEFAULT_P1_REGION'], bg_fp=getattr(pipe._reader, '_bg_fp_p1', None),
        guard_empty_hallucination=pipe._enable_infer_empty_guard,
        enable_hsv_classify_fallback=pipe._enable_hsv_classify_fallback,
        enable_hsv_deferred_consensus=pipe._enable_hsv_deferred_consensus,
        deferred_out=deferred if pipe._enable_hsv_deferred_consensus else None)
    result = dict(inferred=key(inferred), deferred=len(deferred), observed=key(observed))
    if inferred is not None:
        # None対照は予測専用の仮説。元score 0ガードの採用変更ではない。
        for name, delta in (('actual', values['score_d_1p']), ('missing_control', None)):
            final, chains = env['resolve_after_placement'](inferred.copy(), pipe._chain_sim,
                prev_confirmed=before.copy(), score_delta_observed=delta)
            result[name] = dict(delta=delta, chains=chains, final=key(final))
    return result


def capture(frame: Any, factory: Any) -> dict[str, Any]:
    caller, values = frame.f_back, frame.f_back.f_locals
    pipe, control = values['self'], factory.controller
    binding, provider = control.history[SIDE], factory.provider
    before, raw, observed = frame.f_locals['prev_confirmed'], values['cnn_1p_raw'], values['cnn_1p']
    clock = values['frame_idx']
    owner = provider.journal.fifo.entries[(id(pipe), SIDE)]
    queue = pipe._pending_tsumo_1p
    assert caller.f_code.co_name == 'update' and frame.f_locals['self'] is pipe
    assert frame.f_locals['time_sec'] == values['time_sec'] == clock/FPS
    assert owner['queue'] is queue and len(queue) == len(owner['tokens']) == 1
    assert owner['refs'][0] is queue[0] and owner['tokens'][0] == binding.next_token
    assert binding.candidate.pair is queue[0] and binding.next_token not in binding.consumed_tokens
    assert key(before) == binding.grid == binding.current and key(raw) == key(observed)
    assert control.legal(binding.grid, key(raw), list(queue[0]))
    assert binding.clear_count >= 1 and binding.clear_last[0]+STRIDE == clock
    assert binding.clear_grid == key(raw)
    state, old_counter, old_queue = binding.owner.state, deepcopy(pipe._tsumo_count_1p), tuple(queue)
    result = infer(values, pipe, frame.f_globals, before, observed, queue[0])
    assert binding.owner.state is state and pipe._tsumo_count_1p == old_counter
    assert all(a is b for a,b in zip(queue, old_queue)) and len(queue) == len(old_queue)
    assert key(before) == binding.current and key(raw) == result['observed']
    return dict(frame=clock, caller_file=caller.f_code.co_filename, caller_line=caller.f_lineno,
        head_token=owner['tokens'][0], source_enqueue_frame=provider.enqueues[SIDE]['frame_idx'],
        latest_O_frame=provider.link.latest[SIDE]['frame'], clear_last=binding.clear_last,
        clear_count=binding.clear_count, raw=key(raw), current=binding.current,
        score_value=values['_score_ocr_val_1p'], formula_valid=pipe._formula_last_read_1p.valid,
        next_same_frame_unavailable=True, infer=result, no_writer=True, artificial_input=True)


def installed(stack: ExitStack, observations: Any, rows: list[Any]) -> None:
    original = observations.profiling
    def profiling(inner: Any, pipe: Any, factory: Any, clock: Any) -> Any:
        prior_rows = original(inner, pipe, factory, clock)
        previous, code = sys.getprofile(), pipe._apply_chain_formula_early_fire.__func__.__code__
        def profile(frame: Any, event: str, returned: Any) -> None:
            if previous is not None:
                previous(frame, event, returned)
            if frame.f_code is code and event == 'call' and frame.f_locals['side'] == SIDE:
                rows.append(capture(frame, factory))
        sys.setprofile(profile)
        inner.callback(sys.setprofile, previous)
        return prior_rows
    observations.profiling = profiling
    stack.callback(setattr, observations, 'profiling', original)
