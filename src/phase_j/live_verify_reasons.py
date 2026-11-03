"""検証の短絡順序を保ったまま、実際に評価した小条件だけを記録する。"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
from pathlib import Path
from time import time
from typing import Any, Callable

from src.production_config import LIVE_VERIFY_REASON_LOG_DEFAULT

FLAG_ENV = 'PUYO_LIVE_VERIFY_REASON_LOG'
OUTPUT_ENV = 'PUYO_LIVE_VERIFY_REASON_OUTPUT'
LOG_FILE = 'live_verify_reasons.jsonl'
CURRENT: ContextVar[dict | None] = ContextVar('verify_reasons', default=None)


def enabled() -> bool:
    return os.environ.get(FLAG_ENV, str(int(LIVE_VERIFY_REASON_LOG_DEFAULT))) == '1'


def detail(name: str, **values: Any) -> None:
    """未評価の条件は null のまま。OFF では値を保持しない。"""
    row = CURRENT.get()
    if row is not None:
        row['checks'][name] = values
        if values.get('passed') is False:
            row['failed_conditions'].append(name)


def side_detail(name: str, **values: Any) -> None:
    row = CURRENT.get()
    if row is not None:
        detail(row['side']+'.'+name, **values)


def select_side(side: str) -> None:
    row = CURRENT.get()
    if row is not None:
        row['side'] = side


def feedback_score(value: float) -> None:
    row = CURRENT.get()
    if row is not None:
        row['_feedback_max'] = max(value, row.get('_feedback_max', value))


def feedback_values() -> dict[str, Any]:
    from .distribution_input import MARKER_TOLERANCE
    row = CURRENT.get()
    return dict(max_correlation=row.pop('_feedback_max', None), threshold=MARKER_TOLERANCE) if row else {}


def observed(function: Callable) -> Callable:
    """同期・worker 共通の入口。猶予適用後の結果と条件の生値を区別する。"""
    @wraps(function)
    def wrapper(verifier: Any, image: Any, captured: float, size: Any, generation: int) -> bool:
        if not enabled():
            return function(verifier, image, captured, size, generation)
        checks = {side+'.'+key: None for side in ('1P', '2P')
                  for key in ('score', 'grid', 'grid_position', 'border')}
        row = dict(at=time(), captured_at=captured, generation=generation,
                   relaxed=bool(getattr(getattr(verifier, 'base', verifier), 'relaxed', False)),
                   checks=dict(feedback=None, **checks), failed_conditions=[])
        token = CURRENT.set(row)
        try:
            passed = function(verifier, image, captured, size, generation)
            context = (size, generation)
            previous = getattr(verifier, '_reason_streak', (None, 0))
            streak = 0 if passed else (previous[1] if previous[0] == context else 0)+1
            verifier._reason_streak = (context, streak)
            row.update(passed=passed, fail_streak=streak)
            row['grace'] = getattr(verifier, 'diagnostics', None)
            row.pop('side', None)
            write(row)
            return passed
        finally:
            CURRENT.reset(token)
    return wrapper


def write(row: dict[str, Any]) -> None:
    """フレーム計装と別の一検証一行。保存障害を入力判定へ伝播させない。"""
    output = os.environ.get(OUTPUT_ENV)
    if not output:
        return
    try:
        with (Path(output)/LOG_FILE).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=lambda value: value.item())+'\n')
    except OSError:
        logging.getLogger(__name__).exception('入力検証の理由を保存できませんでした')


def border_details(horizontal: list[tuple], vertical: list[tuple], width: int,
                   height: int, exempt_top: bool) -> None:
    """検出済み線分のみで全辺を計装する。Hough や OCR の再実行はしない。"""
    if CURRENT.get() is None:
        return
    from .live_device import GRID_MARGIN, BORDER_COVERAGE, BORDER_POSITION_TOLERANCE, border_coverage
    edges = {}
    for name, lines, axis, span, target in (
        ('top', horizontal, 1, width, GRID_MARGIN),
        ('bottom', horizontal, 1, width, GRID_MARGIN+height),
        ('left', vertical, 0, height, GRID_MARGIN),
        ('right', vertical, 0, height, GRID_MARGIN+width)):
        offsets = [float((line[axis]+line[axis+2])/2-target) for line in lines]
        coverage = float(border_coverage(lines, axis, span, target))
        exempt = name == 'top' and exempt_top
        edges[name] = dict(position_offsets_px=offsets,
            position_passed=any(abs(delta) <= BORDER_POSITION_TOLERANCE for delta in offsets),
            coverage=coverage, coverage_passed=coverage >= BORDER_COVERAGE,
            exempt=exempt, passed=exempt or coverage >= BORDER_COVERAGE)
    side_detail('border', passed=all(edge['passed'] for edge in edges.values()), edges=edges,
                position_tolerance_px=BORDER_POSITION_TOLERANCE, coverage_threshold=BORDER_COVERAGE)
    row = CURRENT.get()
    for name, edge in edges.items():
        for check in ('position', 'coverage'):
            if not edge['exempt'] and not edge[check+'_passed']:
                row['failed_conditions'].append(row['side']+'.border.'+name+'.'+check)
