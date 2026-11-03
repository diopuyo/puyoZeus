"""旧経路の既存試験は、採用前の値を明示して同じ挙動を検査する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEGACY_DEFAULTS = json.loads((Path(__file__).parent / 'fixtures' /
                             'legacy_launcher_defaults.json').read_text(encoding='utf-8'))


def legacy_user_config(data: dict[str, Any]) -> dict[str, Any]:
    """配布既定を外す際も、試験ごとのON/OFF指定を優先する。"""
    return {**LEGACY_DEFAULTS, **data}
