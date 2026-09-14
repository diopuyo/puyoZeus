"""原fixture注入関数で設定欠落を最小再現する。模擬constructorであり実更新ではない。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
FULL = ROOT.parent / 'g2_directional_history_full_runtime_2026-09-09_v1'
PALETTE = ROOT.parent / 'g2_hidden_current_candidate_2026-09-10_v1'
sys.path.insert(0, str(FULL))


def load(alias: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


def reproduce() -> dict:
    fixture = load('_constructor_original_fixture', FULL / 'fixture_full.py')
    palette = load('_constructor_original_palette', PALETTE / 'palette_fixture.py')
    received: list[dict] = []
    class Pipeline:
        def __init__(self, **kwargs: Any) -> None:
            received.append(kwargs)
            self._enable_match_transition_debounce = kwargs.get('enable_match_transition_debounce', False)
    original = Pipeline.__init__
    def real(frozen: Any, monkeypatch: Any) -> Iterator[Any]:
        yield frozen.RecognitionPipeline()
    evidence: dict[str, Any] = {}
    configured = palette.configured(fixture)
    active = N(P=N(load=lambda alias, path: N(real=N(__wrapped__=real))))
    from contextlib import ExitStack
    with ExitStack() as stack:
        configured.install(stack, active, evidence)
        selected = active.P.load('_normal_active_legacy_fixture', None)
        generator = selected.real.__wrapped__(N(RecognitionPipeline=Pipeline), None)
        pipe = next(generator)
        generator.close()
    assert received == [dict(enable_ojama_write_accounting_guard=True, enable_next_history_starvation_fix=True)]
    assert pipe._enable_match_transition_debounce is False and Pipeline.__init__ is original
    return dict(received_kwargs=received, debounce=False, constructor_restored=True,
                missing_flag_reproduced=True, actual_pipeline=False, updates=0, quality_gate_clear=False)


if __name__ == '__main__':
    result = reproduce()
    with (ROOT / 'CONSTRUCTOR_REPRO_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
