"""説明書の版が違う配布物を作らないための試験。"""
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))
import build_release


@pytest.mark.parametrize('version', ['0.6.1', '0.6.2', '0.6.3'])
def test_readme_version_matches(version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'packaging').mkdir()
    (tmp_path / 'packaging/README_ja.txt').write_text(f'# puyoZeus Live {version} — ZIP 版', 'utf-8-sig')
    monkeypatch.setattr(build_release, 'ROOT', tmp_path)
    build_release.validate_readme_version(version)


@pytest.mark.parametrize('heading', ['# puyoZeus Live 0.6.1 — ZIP 版', '# puyoZeus Live 0.6.20 — ZIP 版'])
def test_mismatch_stops_before_build(heading: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'packaging').mkdir()
    (tmp_path / 'packaging/README_ja.txt').write_text(heading, 'utf-8')
    monkeypatch.setattr(build_release, 'ROOT', tmp_path)
    compiler, prepare = Mock(), Mock()
    monkeypatch.setattr(build_release, 'compiler', compiler)
    monkeypatch.setattr(build_release, 'prepare_inputs', prepare)
    monkeypatch.setattr(sys, 'argv', ['build', '--version', '0.6.2'])
    with pytest.raises(ValueError, match='--version 0.6.2'):
        build_release.main()
    compiler.assert_not_called()
    prepare.assert_not_called()
