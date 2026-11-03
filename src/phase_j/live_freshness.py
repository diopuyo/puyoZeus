"""評価完了の単調時計で公開値の鮮度を確認する。決着保持の方針は扱わない。"""
from __future__ import annotations

from typing import Any

EVALUATION_FRESHNESS_SEC = 2.0
MILLISECONDS = 1000.0


def evaluation_stale(row: dict[str, Any] | None, now: float) -> bool:
    """数値の大小ではなく、評価由来で決着表示を除外する。"""
    if not row or row.get('raw_probability') is None or row.get('evaluation_error'):
        return False
    if row.get('source') == 'confirmed_death' or not row.get('match_active', True):
        return False
    calibration = row.get('input_calibration')
    if row.get('input_verifying') or (calibration and calibration['phase'] != 'ready'):
        return False
    return now-row['evaluated_at'] >= EVALUATION_FRESHNESS_SEC


def apply_freshness(payload: dict[str, Any], row: dict[str, Any], now: float) -> None:
    """評価行と判定値は不変。公開時の状態だけを理由付きholdへ変更する。"""
    if not evaluation_stale(row, now) or payload['display']['status'] not in (
            'live', 'physical_prediction', 'hold'):
        return
    expired_at = row['evaluated_at']+EVALUATION_FRESHNESS_SEC
    age_ms = (now-row['evaluated_at'])*MILLISECONDS
    payload['timing']['calculation_age_ms'] = age_ms
    payload['evaluations']['practical']['calculation_age_ms'] = age_ms
    payload['display'].update(status='hold', message='評価更新待ち（更新が遅れています）',
        update_reason='hold_started', primary_hold_reason='calculation_pending',
        all_hold_reasons=['calculation_pending'],
        # 映像時計は最後の観測で凍結し、経過時間だけ単調時計から求める。
        hold_started_ms=int(row['t_sec']*MILLISECONDS),
        hold_elapsed_ms=int((now-expired_at)*MILLISECONDS))
