"""A1フラグが描画CLIと実overlayまで届くことを検証する。"""
from pathlib import Path
import pytest

from tests.test_exchange_event_evaluator import write_models
from tests.test_exchange_event_static_gfe_cli import start_render
from src import production_config


@pytest.mark.parametrize('enabled', [False, True])
def test_render_completion_score_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      enabled: bool) -> None:
    """既定OFFを維持し、ONでは隠し段候補と補完器を同時に有効にする。"""
    write_models(tmp_path)
    flags = ['--exchange-event-update', '--exchange-event-model-dir', str(tmp_path),
             '--prefire-snapshot', '--hidden-row-belief']
    if enabled:
        flags.append('--completion-score-coherent')
    overlay = start_render(monkeypatch, flags)
    assert overlay._prefire.coherent is enabled
    assert (overlay._completion_score is not None) is enabled
    assert '--completion-score-coherent' not in production_config.exchange_event_flags()
