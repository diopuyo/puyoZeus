"""観測直後の写しから xc だけを作り、行の時刻に結び付ける。"""
from __future__ import annotations

import math
from time import perf_counter
from typing import Any

from ._live_ex_frozen.src.dl1k_xc import COLUMNS, blank
from .live_ex_compute import add_xc, timed
from .live_ex_speed import snapshot

MS = 1000.0


def ex_signature(signature: Any) -> Any:
    """xcだけの到着でEX・retRを失効させないため、従来の署名を取り出す。"""
    if isinstance(signature, tuple) and len(signature) == 3 and signature[1] == 'xc':
        return signature[0]
    return signature


def observe_xc(engine: Any) -> dict:
    """射影の連鎖更新と xc_diag を写しに閉じ込める。失敗は xc だけ欠測。"""
    started, times, result = perf_counter(), {}, {}
    try:
        from src.light_consumption import enabled
        from .live_light_ex import forecast_memo
        options = (forecast_memo(engine),) if enabled() else ()
        clone, _ = timed(times, 'snapshot', snapshot, engine, {}, *options)
        timed(times, 'projection', clone.projection)
        timed(times, 'xc', add_xc, clone, result)
    except Exception as exc:
        result.update(xc=blank(), xc_why=[f'error:{exc!r}'] * 2, xc_diag=None)
    result.update(xc_ms=(perf_counter() - started) * MS, xc_ms_by_stage=times)
    from .live_light_ex import calculate_xc
    candidates = timed(times, 'light_candidates', calculate_xc, engine) if enabled() else calculate_xc(engine)
    if candidates:
        result['light_xc'] = candidates
        result.update(xc_ms=(perf_counter() - started) * MS, xc_ms_by_stage=times)
    return result


def select_xc(values: dict, key: dict, extra: dict | None) -> dict:
    """key_at と同じ過去向きの選択。撃ち合い外・未到着なら既存値を使う。"""
    source = 'none'
    if any(math.isfinite(values.get(name, math.nan)) for name in COLUMNS):
        source = 'hold' if extra and extra.get('held') else 'result'
    hold = (extra or {}).get('hold_audit', {})
    stamps = {side: hold.get('ex_hold_source_input_t_' + side, (extra or {}).get('input_t'))
              for side in ('p1', 'p2')}
    audit = dict(xc_source=source, xc_input_t_by_side=stamps)
    if key.get('fight') and 'xc' in key:
        values.update({name: key['xc'].get(name, math.nan) for name in COLUMNS if name in values})
        audit.update(xc_source='same_time', xc_input_t=key['input_t'], xc_serial=key['serial'])
        audit['xc_input_t_by_side'] = dict.fromkeys(('p1', 'p2'), key['input_t'])
    else:
        audit['xc_input_t'] = stamps['p1'] if stamps['p1'] == stamps['p2'] else None
    audit['xc'] = {name: values.get(name, math.nan) for name in COLUMNS}
    return audit
