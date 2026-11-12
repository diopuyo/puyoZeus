"""説明書の版が違う配布物を作らないための試験。"""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))
import build_release


def test_temperature_asset_is_packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """組み立てのコピー対象に較正JSONが入り、原本のバイト列を保つ。"""
    models, bundle = tmp_path/'models', tmp_path/'bundle'
    payload = b'model'
    digest = hashlib.sha256(payload).hexdigest()
    for relative in ('b3_c2_ens/ensemble_ref3_logit.json', 'b3_ex_L200H_retR/ensemble.json'):
        manifest = models/relative
        manifest.parent.mkdir(parents=True)
        members = []
        for index in range(3):
            name = f'member{index}.bin'
            (manifest.parent/name).write_bytes(payload)
            members.append(dict(name=name, sha256=digest))
        manifest.write_text(json.dumps(dict(members=members)), 'utf-8')
    relative = 'calibration/prob_temperature_v1.json'
    raw = (ROOT/'tests/fixtures/prob_temperature/prob_temperature_v1.json').read_bytes()
    (models/'calibration').mkdir()
    (models/relative).write_bytes(raw)
    (models/'b3').mkdir()
    (models/'b3/timing_table.json').write_text('{}', 'utf-8')
    (bundle/'app/models/b3').mkdir(parents=True)
    (bundle/'python/Lib/site-packages/puyo_core').mkdir(parents=True)
    native = tmp_path/'native.pyd'
    native.write_bytes(payload)
    monkeypatch.setattr(build_release, 'RC_NATIVE', native)
    monkeypatch.setattr(build_release, 'RC_NATIVE_SHA', digest)
    hashes = build_release.copy_rc_assets(bundle, models)
    assert (bundle/'app/models'/relative).read_bytes() == raw
    assert hashes[relative] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('version', ['0.6.1', '0.6.2', '0.6.3', '0.6.4'])
def test_readme_version_matches(version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'packaging').mkdir()
    (tmp_path / 'packaging/README_ja.txt').write_text(f'# puyoZeus Live {version} — ZIP 版', 'utf-8-sig')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/DISTRIBUTION_APP_GUIDE.md').write_bytes(
        (tmp_path / 'packaging/README_ja.txt').read_bytes())
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


def test_bundled_readme_mismatch_stops_before_build(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """同梱元だけの旧版と、版は同じだが本文が異なる場合を拒む。"""
    heading = '# puyoZeus Live 0.6.5 — ZIP 版\n'
    (tmp_path/'packaging').mkdir()
    (tmp_path/'docs').mkdir()
    (tmp_path/'packaging/README_ja.txt').write_text(heading+'本文\n', encoding='utf8')
    monkeypatch.setattr(build_release, 'ROOT', tmp_path)
    compiler, prepare = Mock(), Mock()
    monkeypatch.setattr(build_release, 'compiler', compiler)
    monkeypatch.setattr(build_release, 'prepare_inputs', prepare)
    monkeypatch.setattr(sys, 'argv', ['build', '--version', '0.6.5'])
    for content, message in ((heading.replace('0.6.5', '0.6.4'), '--version 0.6.5'),
                             (heading+'別の本文\n', 'バイト列')):
        (tmp_path/'docs/DISTRIBUTION_APP_GUIDE.md').write_text(content, encoding='utf8')
        with pytest.raises(ValueError, match=message):
            build_release.main()
    compiler.assert_not_called()
    prepare.assert_not_called()
