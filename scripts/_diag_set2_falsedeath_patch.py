"""既定OFF修正案のランタイム差し替え。モデル・srcファイルは不変。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

UNSAFE_INPUT_REASONS = frozenset(('color_score_mismatch', 'board_unsettled', 'invalid_start'))


def install_input_guard() -> None:
    """D5bとE35に同じ入力不整合拒否を適用する。"""
    from src import exchange_single_death_proof as single
    from src.exchange_post_counter_bound import PostCounterDeathBound
    resolve, prove = single.resolve_negative_only, PostCounterDeathBound.prove

    def guarded_resolve(result: dict) -> dict:
        return result if result['reason'] in UNSAFE_INPUT_REASONS else resolve(result)

    def guarded_prove(self: Any, projection: Any, overlay: Any, idx: int, latest: tuple,
                      incoming: int, hands: int, context: dict, stamp: float) -> dict:
        reason = projection.safety.blocker(projection, overlay.tracker, idx, stamp)
        if context['hidden'][idx] is None and reason in UNSAFE_INPUT_REASONS:
            return dict(dead=False, reason='diag_input_guard', input_reason=reason)
        return prove(self, projection, overlay, idx, latest, incoming, hands, context, stamp)

    single.resolve_negative_only = guarded_resolve
    PostCounterDeathBound.prove = guarded_prove


def receiver_ledger(projection: Any, idx: int, cap_attack: bool = False) -> Any:
    """数値表示でも裏付けられた自側の式累積を相殺下限として補う。"""
    from src.scoring import score_to_ojama
    safety = projection.safety
    ledger = safety.ledger
    totals = dict(ledger.totals)
    changed = False
    for key, chain in safety.chains.items():
        formula = chain.formula_total
        if formula is None or chain.display_score is None or chain.score_before is None:
            continue
        # score_beforeは_score_changedで落下加点ぶん補正済み。二重控除しない。
        observed = max(0, chain.display_score - chain.score_before)
        if cap_attack and key[0] != idx:
            maximum = int(score_to_ojama(observed, elapsed_sec=safety.elapsed[key]).ojama_count)
            if key in totals and maximum < totals[key][0]:
                totals[key] = (maximum, True)
                changed = True
            continue
        if key[0] != idx or chain.end_signal_sec is None or chain.end_confirmed is False:
            continue
        floor = min(formula, observed)
        amount = int(score_to_ojama(floor, elapsed_sec=safety.elapsed[key]).ojama_count)
        if key in totals and amount > totals[key][0]:
            totals[key] = (amount, True)
            changed = True
    if changed:
        ledger = deepcopy(ledger)
        ledger.observe(ledger.game, totals, ledger.dropped)
    return ledger


def install_score_guard(cap_attack: bool = False, corroborate: bool = False) -> None:
    """死亡入力だけの会計を補正し、予測確率の会計は変更しない。"""
    from src import exchange_death_inputs as inputs
    original = inputs.death_inputs

    def corrected(projection: Any, overlay: Any, latest: tuple, incoming: list) -> dict:
        context = original(projection, overlay, latest, incoming)
        for idx in range(2):
            ledger = receiver_ledger(projection, idx, cap_attack)
            if ledger.pending[idx] < context['incoming'][idx]:
                context['incoming'][idx] = ledger.pending[idx]
                context['verified'][idx] = ledger.verified(idx)
            # 二つの会計が食い違う量を死亡の確定根拠には採らない。
            if corroborate:
                context['incoming'][idx] = min(context['incoming'][idx], int(max(0, incoming[idx])))
        return context

    inputs.death_inputs = corrected


def install(mode: str) -> None:
    """明示指定した差し替えだけを有効にする。"""
    if mode in ('guard', 'combined', 'attributed', 'corroborated'):
        install_input_guard()
    if mode in ('score', 'combined', 'attributed', 'corroborated'):
        install_score_guard(cap_attack=mode in ('attributed', 'corroborated'),
                            corroborate=mode == 'corroborated')
