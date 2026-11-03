"""v4.9f検証専用の回数上限。既定OFF、本番設定とは接続しない。"""
from __future__ import annotations
import os

FLAG = 'PUYO_PREFIRE_V49F_DETERMINISTIC'
CURVE_HANDS = 2
REPAIR_SIMULATIONS = 800
FORMULA_SIMULATIONS = 2400
CANDIDATE_SIMULATIONS = 800

def enabled() -> bool:
    return os.environ.get(FLAG, '0') == '1'

def publication(value: dict) -> None:
    """同期再生の論理公開時刻。実測時間を入力選択や公開可否へ戻さない。"""
    if enabled():
        value['measured_completion_ms'] = (value['ready_sec']-value['input_sec'])*1000
        value['ready_sec'] = value['input_sec']
        value['over_budget'] = False
