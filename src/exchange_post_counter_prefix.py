"""E35‴: 実際の完走候補の途中得点を、同じ連鎖の独立観測と照合する。"""
from __future__ import annotations

from typing import Any
from itertools import accumulate

from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ


def remember_prediction(projection: Any, chain: Any, result: Any) -> None:
    """元予測の全段得点を保持し、復元器の候補棄却による消失と区別する。"""
    engine = getattr(projection, 'post_counter_bound', None)
    if engine is not None and (engine.completion_prefix_integrity or engine.completion_step_integrity or engine.completion_recent_integrity):
        from src.scoring import calculate_chain_score
        engine.base_predictions[chain.chain_id] = dict(
            prefix=tuple(accumulate(s.score for s in calculate_chain_score(result).steps)),
            board=result.final_board._grid.tolist())


def observe_formula(overlay: Any, result: Any, observations: dict) -> None:
    """推定式通知は実読に昇格させず、当該連鎖の式だけを保存する。"""
    for idx, side in enumerate((result.p1, result.p2)):
        chain = overlay.tracker.latest_chain(f'{idx+1}P')
        event = side.chain_event
        if (chain is not None and event is not None
                and event.mechanism == CHAIN_MECHANISM_FORMULA_READ
                and event.trigger_sec >= chain.trigger_sec):
            observations[(overlay._game, chain.chain_id)] = (event.chain_count, event.total_score)


def candidates(projection: Any, chain: Any, hidden: dict | None) -> list[dict]:
    """隠し段候補を優先し、通常予測は終端盤面・総得点・段数まで同定する。"""
    if hidden is not None:
        return hidden['options']
    options = []
    engine = getattr(projection, 'post_counter_bound', None)
    if engine is not None and chain.chain_id in engine.base_predictions:
        options.append(engine.base_predictions[chain.chain_id])
    safety = getattr(projection, 'safety', None)
    if safety is not None:
        options.extend(safety.recovery.entries.get(chain.chain_id, {}).get('options', []))
    midchain = getattr(projection, 'midchain', None)
    if midchain is not None:
        for entry in midchain.entries.values():
            if entry['chain'] is chain and entry.get('active') is not None:
                options.append(entry['active'])
    return [v for v in options if v['board'] == chain.predicted_final_board
            and prefix_map(v) and max(prefix_map(v)) == chain.predicted_chain_count
            and prefix_map(v)[max(prefix_map(v))] == chain.predicted_final_score]


def prefix_map(option: dict) -> dict[int, float]:
    """途中起点の絶対段番号を保ち、通常起点だけ1始まりへ変換する。"""
    prefix = option.get('prefix', ())
    return dict(prefix) if isinstance(prefix, dict) else dict(enumerate(prefix, 1))


def check_prefix(projection: Any, chain: Any, hidden: dict | None,
                 observed: tuple | None) -> dict:
    """式・表示増分の欠測や矛盾は許容せず、全証明候補の一致を要求する。"""
    count, formula = observed if observed is not None else (None, None)
    before, display = chain.score_before, chain.display_score
    delta = None if before is None or display is None else display-before-chain.drop_bonus_score
    totals = [prefix_map(v) for v in candidates(projection, chain, hidden)]
    predicted = [dict(counts=list(p), cumulative=p.get(count)) for p in totals]
    values = (count, formula, chain.formula_total, delta)
    missing = any(v is None for v in values) or not totals
    matched = (not missing and count >= 1 and all(p.get(count) == formula for p in totals)
               and formula == chain.formula_total == delta)
    return dict(accepted=matched,
        reason='prefix_consistent' if matched else ('prefix_unobserved' if missing else 'prefix_mismatch'),
        predicted=predicted, observed_count=count, observed_formula_cumulative=formula,
        formula_total=chain.formula_total, score_before=before, display_score=display,
        score_display_increment=delta, score_delta=chain.score_delta,
        drop_bonus_score=chain.drop_bonus_score, predicted_final_score=chain.predicted_final_score,
        predicted_chain_count=chain.predicted_chain_count)
