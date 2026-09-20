"""元SideInputのcall時値を不変化。元Board/SideInput/PBの参照は変更しない。"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any
import current_call_fixed as F

PROVISIONAL = F.ROOT.parent / 'g2_hidden_probability_provisional_2026-09-08_v1/provisional.py'
PROVISIONAL_SHA = '810ef16cc4ace3f1f92bc3a3d1edd49038f875ceeaf30b1364b233e5896463bf'
VALIDATORS = ('validate_scope', 'validate_side', 'confirmed_grid', 'distribution', 'prepare')


@lru_cache(maxsize=1)
def validator_codes() -> dict[str, CodeType]:
    raw = PROVISIONAL.read_bytes()
    F.require(hashlib.sha256(raw).hexdigest() == PROVISIONAL_SHA, 'provisional_source')
    module = compile(raw, str(PROVISIONAL), 'exec', dont_inherit=True)
    return {code.co_name: code for code in module.co_consts if isinstance(code, CodeType)}


def original_validators(source: Any) -> None:
    fixed = validator_codes()
    for name in VALIDATORS:
        value = source[name]
        F.require(type(value) is FunctionType and value.__globals__ is source
            and value.__code__ == fixed[name] and value.__closure__ is None,
            'original_validator_code:' + name)


def capture(current: Any, value: Any, side: Any) -> dict[str, Any]:
    source = type(side).__init__.__globals__
    F.require(Path(source['__file__']).resolve() == PROVISIONAL
        and type(side) is source['SideInput'], 'original_side_type')
    original_validators(source)
    source['validate_side'](side, value.scope[-1])
    _, hidden, visible = source['prepare'](side)
    grid = source['confirmed_grid'](side.confirmed)
    cells = current.cells(side.probability)
    proof = json.loads(value.evidence_json)
    F.require(current.encoded(hidden) == current.encoded(value.hidden), 'side_hidden')
    F.require(current.encoded(cells) == current.encoded(proof['conditional_PB']), 'side_PB')
    F.require(visible == proof['visible_sha256'], 'side_visible_digest')
    return dict(side_object_id=id(side), confirmed_object_id=id(side.confirmed),
        probability_object_id=id(side.probability),
        confirmed_grid=tuple(tuple(int(cell) for cell in row) for row in grid),
        hidden=hidden, visible_sha256=visible, conditional_PB=cells,
        validator_source_sha256=PROVISIONAL_SHA)
