"""E35⁗: 段番号と段間得点を照合し、累積得点の一定の基準差を許容する。"""
from __future__ import annotations

import math
from typing import Any

from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.exchange_post_counter_prefix import candidates, prefix_map

FIRST_STAGE = 1
MIN_OBSERVATIONS = 2


def valid_score(value: Any) -> bool:
    """欠測・非有限・負の累積を、観測済みの得点へ昇格させない。"""
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def observe_steps(overlay: Any, result: Any, entries: dict) -> None:
    """実読した段番号を基準に、式と表示の履歴を同じ連鎖へ帰属させる。"""
    for idx, side in enumerate((result.p1, result.p2)):
        chain = overlay.tracker.latest_chain(f'{idx+1}P')
        event = side.chain_event
        if chain is None:
            continue
        key = (overlay._game, chain.chain_id)
        entry = entries.setdefault(key, dict(stages={}, conflicts=[], last_count=None))
        if (event is not None and event.mechanism == CHAIN_MECHANISM_FORMULA_READ
                and event.trigger_sec >= chain.trigger_sec and not getattr(event, 'score_estimated', False)):
            record_formula(entry, event.chain_count, event.total_score)
        # 式の最新通知より表示が1段遅れるため、追跡器が受理した累積で段を同定する。
        counts = [n for n, r in entry['stages'].items() if r['formula'] is not None
                  and r['formula'] == chain.formula_total]
        if len(counts) == 1 and valid_score(chain.display_score) and valid_score(chain.score_before):
            entry['stages'][counts[0]]['display'] = chain.display_score-chain.score_before-chain.drop_bonus_score


def record_formula(entry: dict, count: int, score: float | None) -> None:
    """同一段の異なる式や段の逆行を残し、後続一致で不整合を消さない。"""
    if not isinstance(count, int) or count < FIRST_STAGE:
        entry['conflicts'].append('invalid_stage')
        return
    previous = entry['last_count']
    if previous is not None and count < previous and 'stage_regression' not in entry['conflicts']:
        entry['conflicts'].append('stage_regression')
    entry['last_count'] = count if previous is None else max(count, previous)
    row = entry['stages'].setdefault(count, dict(formula=None, display=None))
    if not valid_score(score):
        return
    if row['formula'] is not None and row['formula'] != score:
        if 'formula_revision' not in entry['conflicts']:
            entry['conflicts'].append('formula_revision')
    row['formula'] = score


def compare_steps(prefix: dict, stages: dict) -> dict:
    """全観測段を照合し、隣接段の増分が一つでも違えば不一致とする。"""
    counts = sorted(stages)
    rows, missing, mismatch = [], len(counts) < MIN_OBSERVATIONS, False
    for count in counts:
        if count not in prefix:
            mismatch = True
    for previous, count in zip(counts, counts[1:]):
        predicted = prefix[count]-prefix[previous] if count in prefix and previous in prefix else None
        observed = {k: stages[count][k]-stages[previous][k] for k in ('formula', 'display')
                    if valid_score(stages[count][k]) and valid_score(stages[previous][k])}
        contiguous = count == previous+FIRST_STAGE
        missing |= not contiguous or not observed
        mismatch |= predicted is None or any(v != predicted for v in observed.values())
        rows.append(dict(previous_count=previous, count=count, predicted_increment=predicted,
            observed_increments=observed, contiguous=contiguous,
            offsets={k: stages[count][k]-prefix[count] for k in observed if count in prefix}))
    return dict(accepted=not missing and not mismatch, missing=missing, mismatch=mismatch,
                predicted_counts=list(prefix), comparisons=rows)


def check_steps(projection: Any, chain: Any, hidden: dict | None, entry: dict | None) -> dict:
    """候補集合全てと観測済み全段を照合し、証明の許否と値を監査へ返す。"""
    entry = entry or dict(stages={}, conflicts=[], last_count=None)
    checks = [compare_steps(prefix_map(v), entry['stages']) for v in candidates(projection, chain, hidden)]
    mismatch = bool(entry['conflicts']) or any(v['mismatch'] for v in checks)
    missing = not checks or any(v['missing'] for v in checks)
    accepted = not mismatch and not missing
    return dict(accepted=accepted, reason='step_mismatch' if mismatch else (
        'step_unobserved' if missing else 'steps_consistent'),
        observed_stages={n: dict(r) for n, r in entry['stages'].items()},
        conflicts=list(entry['conflicts']), candidates=checks,
        predicted_chain_count=chain.predicted_chain_count, predicted_final_score=chain.predicted_final_score)
