"""原fixtureの生成時に3設定を注入する。実Coreと原constructor本体は変えない。"""
from __future__ import annotations
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any, Iterator

FLAGS = ('enable_next_history_starvation_fix', 'enable_match_transition_debounce')
PLAN = Path(__file__).resolve().parent.parent / 'video38_history_publication_probe_live_2026-09-11_v12/PLAN.json'


def configured(fixture: Any) -> Any:
    source = dict(vars(fixture))
    extra = [FunctionType(fixture.original_real.__code__, dict(source, FLAG=flag)) for flag in FLAGS]
    def original_real(original: Any, evidence: dict) -> Any:
        palette: dict[str, Any] = {}
        debounce: dict[str, Any] = {}
        merged = fixture.original_real(extra[1](extra[0](original, palette), debounce), evidence)
        def real(frozen: Any, monkeypatch: Any) -> Iterator[Any]:
            try:
                yield from merged(frozen, monkeypatch)
            finally:
                evidence['palette_constructor_fixture'] = palette
                evidence['debounce_constructor_fixture'] = debounce
        return real
    install = FunctionType(fixture.install.__code__, dict(source, original_real=original_real))
    return N(**(source | dict(install=install)))


def install(stack: Any, palette: Any, receipt: dict) -> None:
    assert json.loads(PLAN.read_bytes())['actual_collector_kwargs'][FLAGS[1]] is True
    original = palette.configured
    assert not getattr(original, '_g2_constructor_fixture', False), 'constructor_fixture_duplicate'
    def select(fixture: Any) -> Any:
        assert not receipt.get('selected'), 'constructor_fixture_reentry'
        receipt.update(selected=True, flags=list(FLAGS), entire_constructor_equivalent=False)
        return configured(fixture)
    select._g2_constructor_fixture = True
    def restore() -> None:
        assert palette.configured is select, 'constructor_fixture_foreign_hook'
        palette.configured = original
        receipt['restored'] = palette.configured is original
    stack.callback(restore)
    palette.configured = select
