"""履歴接続で再用する既存2部品の固定読込。データcatalogは読まない。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PROVIDER = ROOT.parent / 'g2_normal_completion_transaction_2026-09-09_v1/live_provider.py'
INVENTORY = ROOT.parent / 'g2_machine_inventory_producer_2026-09-09_v1/producer.py'
FIXED = {PROVIDER: '5d0b400d965700943887b2bb833393689c84a46aa2a5d302e1ff7f6e600dca6f',
         INVENTORY: 'd8e796ae0ab50c10710a1455fc72c28f04319b8392b4e4e4b698aa51b40b6dee'}


def load(alias: str, path: Path) -> Any:
    if alias in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest() != FIXED[path]:
        raise RuntimeError('history_dependency_collision_or_changed')
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


V = load('_history_runtime_base_provider', PROVIDER)
if any(name in sys.modules for name in ('_machine_inventory_diagnostic', '_machine_inventory_split')):
    raise RuntimeError('history_inventory_alias_collision')
P = load('_history_runtime_inventory', INVENTORY)
