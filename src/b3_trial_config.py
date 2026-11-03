"""M3の既定OFF設定。製品のproduction_configを変更せずspawnへ伝える。"""
from __future__ import annotations
import os
from typing import Any
from src import production_config

M3_FLAGS = frozenset(('B3_LEDGER_INFLIGHT_DEFAULT', 'B3_LEDGER_OBSERVED_RULE_DEFAULT',
                     'B3_LEDGER_MODE_DEFAULT', 'B3_LEDGER_LANDING_DEFAULT',
                     'B3_ICON_LINE_ALIAS_DEFAULT'))


def enabled() -> bool:
    return os.environ.get('PUYO_B3_M3') == '1'


def score_cadence_enabled() -> bool:
    """本人試験だけ、得点窓を実観測間隔へ適応する。研究再生でも明示指定可。"""
    return os.environ.get('PUYO_LIVE_SCORE_CADENCE') == '1'


def __getattr__(name: str) -> Any:
    if name in M3_FLAGS:
        return enabled() or getattr(production_config, name, False)
    return getattr(production_config, name)


def b3_ledger_mode_enabled() -> bool:
    return enabled() or production_config.b3_ledger_mode_enabled()


def b3_ledger_landing_enabled() -> bool:
    return enabled() or production_config.b3_ledger_landing_enabled()
