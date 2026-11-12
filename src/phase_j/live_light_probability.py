"""候補に整合するB3行を再構成し、較正済み勝率を確率平均する。"""
from __future__ import annotations

import base64
from copy import copy, deepcopy
from dataclasses import replace
import math
import pickle
from typing import Any
import zlib

import numpy as np

from src.light_consumption import enabled
from src.light_hidden_posterior import scenarios
from .live_light_hold import identity

CONTEXT = '_light_context'
SCENARIOS = '_light_scenarios'
CANDIDATES = '_light_candidates'
SNAPSHOT_CHOICES = '_light_snapshot_choices'


def feature_state(state: Any) -> Any:
    """特徴用の配送には現在の予測と選択候補だけを固定する。観測側は変えない。"""
    if not enabled() or hasattr(state, SNAPSHOT_CHOICES):
        return state
    frozen = copy(state)
    setattr(frozen, SNAPSHOT_CHOICES, scenarios([side.forecast for side in state.sides]))
    frozen.sides = [copy(side) for side in state.sides]
    for side in frozen.sides:
        if side.forecast is not None:
            side.forecast = copy(side.forecast)
            side.forecast.fc = deepcopy(side.forecast.current())
            side.forecast._light_physics = None
    return frozen


def attach_context(rows: Any, row: dict) -> None:
    if not enabled():
        return
    from src.b3_async import snapshot
    frozen = copy(rows)
    frozen.state = feature_state(rows.state)
    row[CANDIDATES] = deepcopy(getattr(frozen.state, SNAPSHOT_CHOICES))
    if not row[CANDIDATES]:
        return
    payload = (snapshot(frozen), rows.table, rows.video_id)
    row[CONTEXT] = base64.b85encode(zlib.compress(pickle.dumps(payload))).decode('ascii')


def candidate_row(context: tuple, scenario: dict) -> dict | None:
    from src.b3_rows import _make_row
    payload, table, video = context
    state, trigger = pickle.loads(payload)
    for i, choice in enumerate(scenario['sides']):
        if choice is None:
            continue
        side = state.sides[i]
        owner, episode = side.forecast, side.episode
        original = owner.current()
        if (original is None or episode is None or not episode.active
                or original.chain_id != choice['chain_id']
                or tuple(choice['steps'][:len(original.observed)]) != original.observed):
            return None
        owner._light_physics = None
        owner.fc = replace(original, steps=tuple(choice['steps']),
                           final_board=np.asarray(choice['final'], dtype=np.int8), lower_bound=False)
        side._apply_forecast(state.t)
        episode.update_prediction()
    return _make_row(state, table, video, trigger, None)


def selected_values(owner: Any, extra: dict | None, t: float, row: dict | None = None) -> list[dict]:
    if not enabled():
        return []
    received = (extra or {}).get('light_scenarios', [])
    choices = (row or {}).get(CANDIDATES, received)
    available = {identity(item): item for item in received}
    history = getattr(owner.worker, 'xc_history', None)
    key = history.key_at(t) if owner.settings.get('xc_same_time', False) and history else {}
    xc = {identity(row): row for row in key.get('light_xc', [])}
    result = []
    for choice in choices:
        source = available.get(identity(choice), {})
        values = {name: source.get('extra', {}).get(name, math.nan) for name in owner.ex_columns}
        if key.get('fight') and 'xc' in key:
            from .live_24h_model import XC_COLUMNS
            current = xc.get(identity(choice), {}).get('xc', {})
            values.update({name: current.get(name, math.nan) for name in XC_COLUMNS if name in values})
        result.append(dict(choice, values=values, ex_available=bool(source)))
    return result


def prepared_rows(owner: Any, row: dict, choices: list[dict]) -> list[dict | None]:
    """同じ固定文脈の候補行だけ再利用し、EX・XCと重みは保持しない。"""
    if CONTEXT not in row:
        return [None] * len(choices)
    encoded = row[CONTEXT]
    previous, cached = getattr(owner, '_light_row_cache', (None, {}))
    if previous != encoded:
        cached = {}
    keys = [identity(choice) for choice in choices]
    context = None
    for key, choice in zip(keys, choices):
        if key not in cached:
            if context is None:
                context = pickle.loads(zlib.decompress(base64.b85decode(encoded)))
            cached[key] = candidate_row(context, choice)
    # その行で選んだ候補だけを残す。前の文脈・終了済みの状態を引き継がない。
    owner._light_row_cache = (encoded, {key: cached[key] for key in keys})
    return [None if cached[key] is None else dict(cached[key]) for key in keys]


def weighted_infer(owner: Any, row: dict, values: dict) -> float:
    scenarios = values[SCENARIOS]
    rows = prepared_rows(owner, row, scenarios)
    results = []
    pairs = [(scenario, derived) for scenario, derived in zip(scenarios, rows) if derived is not None]
    if pairs:
        infer_many = getattr(owner, 'infer_candidates', None)
        probabilities = (infer_many([derived for _, derived in pairs],
                                    [dict(scenario['values']) for scenario, _ in pairs])
                         if infer_many else [owner.infer_single(derived, dict(scenario['values']))
                                             for scenario, derived in pairs])
        results = [dict(weight=scenario['weight'], probability=value,
                        totals=[s['total'] if s else None for s in scenario['sides']])
                   for (scenario, _), value in zip(pairs, probabilities)]
    if not results:
        # 後の式で全候補が否定された行に、過去の候補EXを組み合わせない。
        missing = {name: math.nan for name in owner.ex_columns}
        owner.light_audit = dict(reason='candidate_row_mismatch', candidates=[])
        return owner.infer_single(row, missing)
    mass = sum(item['weight'] for item in results)
    probability = sum(item['weight']*item['probability'] for item in results)/mass
    owner.light_audit = dict(reason='posterior_mean', candidates=results,
        retained_mass=scenarios[0]['retained_mass'], compatible_mass=mass)
    return probability
