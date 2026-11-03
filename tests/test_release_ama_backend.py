"""配布から参照exeを起こさず、開発用の明示OFFは保つ。"""
from pathlib import Path
from unittest.mock import patch

import pytest

from src.phase_j.live_ama_engine import AmaEngine
from src.phase_j.release_config import release_environment


@pytest.mark.parametrize('value', [None, '0', '1'])
def test_distribution_never_spawns_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    value: str | None,
) -> None:
    environment = {} if value is None else {'PUYO_OBS_AMA_RUST': value}
    monkeypatch.setattr('os.environ', release_environment(environment))
    model = tmp_path / 'model.json'
    model.write_text('{}', encoding='utf-8')
    with patch('subprocess.Popen') as spawn:
        engine = AmaEngine(tmp_path / 'ama_reference.exe', model)
        assert engine.rust_threads == 2
        assert all(type(ref).__name__ == 'NativeReference' for ref in engine.references)
        spawn.assert_not_called()
        engine.close()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and 'AMA_BACKEND native' in lines[0]
    assert ('override disabled Rust' in lines[0]) == (value == '0')


def test_development_zero_keeps_original_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('os.environ', {'PUYO_OBS_AMA_RUST': '0'})
    binary, model = tmp_path / 'ama_reference.exe', tmp_path / 'model.json'
    binary.touch()
    model.write_text('{}', encoding='utf-8')
    with patch('src.phase_j.live_ama_engine.Reference') as reference:
        engine = AmaEngine(binary, model)
        assert engine.rust_threads == 0
        assert reference.call_count == 2
        reference.assert_called_with(binary)
