"""凍結の計算を学習と同じ順序で呼び、値とは別に内訳を記録する。"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from ._live_ex_frozen.src.dl1k_engine import (
    MS, ExEngine, add_retr, classify_error, proof_of, retr_enabled, search_of,
)
from ._live_ex_frozen.src.dl1k_xc import blank, compute_xc, xc_enabled

STAGES = ('projection', 'retr', 'xc', 'proof', 'search')
RETR_DIAG_KEYS = ('I', 'I_pred', 'fire_back', 'fc_id', 'fc_total_o', 'fc_total_s',
                  'res_s', 'res_o', 'end_o', 't0_o', 'n_obs_o', 'n_pred_o', 'h')


def timed(times: dict, name: str, fn: Callable, *args: Any) -> Any:
    started = perf_counter()
    try:
        return fn(*args)
    finally:
        times[name] = (perf_counter() - started) * MS


def add_xc(engine: ExEngine, result: dict) -> None:
    """xc の失敗は xc の列だけを欠測にする。"""
    engine.xc_diag = None
    try:
        result['xc'], result['xc_why'] = compute_xc(engine, None)
    except Exception as exc:
        result['xc'], result['xc_why'] = blank(), [f'error:{exc!r}'] * 2
    result['xc_diag'] = engine.xc_diag


def retr_details(engine: ExEngine, value: dict, result: dict) -> None:
    """攻め側の射影状態は死亡証明の前に読み、盤配列は保存しない。"""
    result['retr_diag'] = [None if d is None else {k: d.get(k) for k in RETR_DIAG_KEYS}
                           for d in getattr(engine, 'retr_diag', [None, None])]
    result['retr_attack'] = [dict(
        board_prediction_active=bool(value['board_prediction_active'][1-s]),
        reconciliation_is_none=value['reconciliation'][1-s] is None) for s in (0, 1)]
    ama = getattr(engine, 'retr_ama', None)
    result['retr_ama'] = dict(calls=getattr(ama, 'calls', 0), hits=getattr(ama, 'hits', 0))


def compute(engine: ExEngine) -> dict:
    """射影→retR→xc→死亡証明→探索。OFF時も凍結と同じ関数・順序で呼ぶ。"""
    started = perf_counter()
    times = dict.fromkeys(STAGES, 0.0)
    result = dict(output=None, reason=None, observed_scores=None, proof=None, failure=None)
    try:
        if xc_enabled() and not retr_enabled():
            raise ValueError('live_24h_xcにはlive_24h_retrが必要です')
        value = timed(times, 'projection', engine.projection)
        if getattr(engine, 'ex_speed', False) and retr_enabled():
            from .live_ex_speed import parallel
            if parallel(engine, value, result, times):
                result['compute_ms'] = (perf_counter() - started) * MS
                result['compute_ms_by_stage'] = times
                return result
        if retr_enabled():
            engine.retr_diag = [None, None]
            timed(times, 'retr', add_retr, engine, value, result)
            retr_details(engine, value, result)
        if xc_enabled():
            timed(times, 'xc', add_xc, engine, result)
        result['proof'] = timed(times, 'proof', proof_of, engine, value)
        result['output'], result['reason'], result['observed_scores'] = timed(
            times, 'search', search_of, engine, value)
    except Exception as exc:
        result.update(failure=classify_error(exc), error=repr(exc))
    result['compute_ms'] = (perf_counter() - started) * MS
    result['compute_ms_by_stage'] = times
    return result
