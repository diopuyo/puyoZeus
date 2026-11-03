"""E35⁗′: 直近の連続2段を照合し、過去の不一致は監査だけに残す。"""
from __future__ import annotations

from typing import Any
import math

from src.exchange_post_counter_prefix import candidates, prefix_map
from src.exchange_post_counter_steps import compare_steps
from src.exchange_prefire_stage_timeout import STAGE_INTERVAL_SEC

RECENT_STEPS = 2
RECENT_TOTALS = RECENT_STEPS + 1
# 事前登録の許容遅れは1段。発火からの進みは既存実測の1.17秒/段で換算する。
MAX_STAGE_LAG = 1


def compare_recent(prefix: dict, stages: dict) -> dict:
    """最新段を終点とする2増分を検査し、古い窓へ戻って合格させない。"""
    counts = sorted(stages)
    recent_counts = counts[-RECENT_TOTALS:]
    recent = compare_steps(prefix, {n: stages[n] for n in recent_counts})
    recent['missing'] |= len(recent_counts) < RECENT_TOTALS
    recent['accepted'] = not recent['missing'] and not recent['mismatch']
    earlier = compare_steps(prefix, stages)['comparisons'][:-RECENT_STEPS]
    mismatches = [r for r in earlier if r['predicted_increment'] is None or any(
        v != r['predicted_increment'] for v in r['observed_increments'].values())]
    return dict(**recent, recent_counts=recent_counts, earlier_comparisons=earlier,
                earlier_mismatches=mismatches, earlier_mismatch=bool(mismatches))


def check_recent(projection: Any, chain: Any, hidden: dict | None, entry: dict | None,
                 *, strict_integrity: bool = False, stamp: float | None = None,
                 cap_completed: bool = False) -> dict:
    """同じ完走候補全ての最新2段を検査し、照合値を監査へ返す。"""
    entry = entry or dict(stages={}, conflicts=[], last_count=None)
    options = candidates(projection, chain, hidden) if not strict_integrity else base_candidates(projection, chain)
    checks = [compare_recent(prefix_map(v), entry['stages']) for v in options]
    # 段の逆行・同一段の実読改訂は時点を特定できないため、従来どおり不整合とする。
    mismatch = bool(entry['conflicts']) or any(v['mismatch'] for v in checks)
    missing = not checks or any(v['missing'] for v in checks)
    freshness = stage_freshness(chain, entry, stamp, cap_completed=cap_completed) if strict_integrity else {}
    missing |= strict_integrity and not freshness['fresh']
    return dict(accepted=not mismatch and not missing, **freshness,
        reason='recent_mismatch' if mismatch else ('recent_unobserved' if missing else 'recent_consistent'),
        observed_stages={n: dict(r) for n, r in entry['stages'].items()},
        conflicts=list(entry['conflicts']), candidates=checks,
        earlier_mismatch=any(v['earlier_mismatch'] for v in checks),
        predicted_chain_count=chain.predicted_chain_count, predicted_final_score=chain.predicted_final_score)


def base_candidates(projection: Any, chain: Any) -> list[dict]:
    """観測から補正した候補を一切混ぜず、発火時の元予測だけを返す。"""
    engine = getattr(projection, 'post_counter_bound', None)
    option = None if engine is None else engine.base_predictions.get(chain.chain_id)
    return [] if option is None else [option]


def stage_freshness(chain: Any, entry: dict, stamp: float | None,
                    *, cap_completed: bool = False) -> dict:
    """発火を1段目とし、既存の段間物理時間で現在段を推定する。"""
    observed = max(entry['stages'], default=None)
    elapsed = None if stamp is None else stamp-chain.trigger_sec
    current = None if elapsed is None or not math.isfinite(elapsed) or elapsed < 0 else (
        1 + math.floor(elapsed / STAGE_INTERVAL_SEC))
    # 完走後は存在しない次段へ進めず、途中の観測遅れ判定は維持する。
    if cap_completed and current is not None and (chain.predicted_chain_count or 0) > 0:
        current = min(current, chain.predicted_chain_count)
    lag = None if current is None or observed is None else current-observed
    return dict(estimated_current_stage=current, observed_latest_stage=observed,
                stage_lag=lag, fresh=lag is not None and lag <= MAX_STAGE_LAG)
