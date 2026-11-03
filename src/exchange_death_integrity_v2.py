"""D-v2: 得点の独立証拠と、実際に使う完走候補の整合を検査する。"""
from __future__ import annotations

from typing import Any

UNKNOWN_CURRENT_PAIR = (0, 0)


def independently_verified(chain: Any) -> bool:
    """表示差そのものを二度数えず、式又は確定通知の範囲内だけを信用する。"""
    if chain.display_score is None or chain.score_before is None:
        return False
    if chain.formula_total is None and not hasattr(chain, 'finalized_evidence_score'):
        return False
    independent = max(chain.formula_total or 0., getattr(chain, 'finalized_evidence_score', 0.))
    return (chain.provisional_score <= max(0., chain.display_score-chain.score_before)
            and chain.provisional_score <= independent)


def completion_blocker(projection: Any, tracker: Any, idx: int, context: dict | None) -> str | None:
    """共通復元器の欠測ではなく、死亡証明へ渡す候補自体の現時点の有効性を調べる。"""
    from src.exchange_death_inputs import maximum_response
    chain = tracker.latest_chain(f'{idx+1}P')
    selected = context['hidden'][idx] if context is not None else None
    fresh = maximum_response(projection, chain) if chain is not None else None
    if not selected or not fresh or not fresh.get('options'):
        return 'unverified_post_counter_candidates'
    audit = fresh.get('audit', {})
    if audit.get('withdrawn') or audit.get('accepted') is False:
        return 'unverified_post_counter_candidates'
    identity = lambda bound: (bound['score'], sorted(tuple(map(tuple, v['board'])) for v in bound['options']))
    if identity(selected) != identity(fresh):
        return 'stale_post_counter_candidates'
    return None
