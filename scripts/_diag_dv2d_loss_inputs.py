"""保存した同時刻入力から、拒否条件の数値根拠を読み取る純粋な診断補助。"""
from __future__ import annotations


def provisional(chain: dict) -> float:
    """ExchangeChainRecord.provisional_scoreと同じ分岐を保存値へ適用する。"""
    ready = (chain.get('score_ready_sec') is not None and chain.get('end_signal_sec') is not None
             and chain.get('end_confirmed') is not False)
    delta, formula = chain.get('score_delta') or 0., chain.get('formula_total') or 0.
    if ready:
        return max(delta, formula) if chain.get('score_attribution') else delta
    return max(delta, formula, chain.get('predicted_final_score') or 0.)


def independent_failure(chain: dict) -> dict:
    """D3′の各比較式と、独立根拠との差を分けて保存する。"""
    score = provisional(chain)
    display, before = chain.get('display_score'), chain.get('score_before')
    independent = max(chain.get('formula_total') or 0., chain.get('finalized_evidence_score') or 0.)
    delta = None if display is None or before is None else max(0., display-before)
    reasons = []
    if delta is None:
        reasons.append('missing_display_or_before')
    if chain.get('formula_total') is None and 'finalized_evidence_score' not in chain:
        reasons.append('missing_independent_observation')
    if delta is not None and score > delta:
        reasons.append('provisional_exceeds_display_delta')
    if score > independent:
        reasons.append('provisional_exceeds_independent_score')
    return dict(reasons=reasons, provisional_score=score, display_delta=delta,
                independent_score=independent, independent_gap=max(0., score-independent),
                formula_total=chain.get('formula_total'),
                finalized_evidence_score=chain.get('finalized_evidence_score'))


def ledger_failures(sample: dict, side: str) -> list[dict]:
    """受け側に残る未裏付け予告だけを、その送り側連鎖の条件へ結び付ける。"""
    idx = int(side == '2P')
    ledger = sample.get('ledger') or {}
    chains = {(int(c['side'] == '2P'), c['chain_id']): c for c in sample.get('chains', [])}
    result = []
    for key, amount in (ledger.get('chunks') or [[], []])[idx]:
        total = ledger.get('totals', {}).get(str(tuple(key)))
        if total is None or total[1] or tuple(key) not in chains:
            continue
        chain = chains[tuple(key)]
        result.append(dict(key=key, remaining_ojama=amount, total_ojama=total[0],
            trigger_sec=chain['trigger_sec'], **independent_failure(chain)))
    return result


def recognition_delta(exact: dict) -> dict:
    """本番初回と同時刻の旧新認識差を、行列座標と値で列挙する。"""
    old, new = (exact[mode]['args']['tuple'][0]['namespace'] for mode in ('prod', 'v2d'))
    result = {}
    for side in ('p1', 'p2'):
        before, after = old[side]['namespace'], new[side]['namespace']
        changed = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
        grids = [(value.get('confirmed_board') or {}).get('board', {}).get('array')
                 for value in (before, after)]
        cells = []
        if all(grid is not None for grid in grids):
            for row, (left, right) in enumerate(zip(*grids)):
                cells.extend(dict(row=row, col=col, prod=a, v2d=b)
                             for col, (a, b) in enumerate(zip(left, right)) if a != b)
        result[side] = dict(changed_fields=changed, changed_confirmed_cells=cells,
                            board_missing=[grid is None for grid in grids])
    return result
