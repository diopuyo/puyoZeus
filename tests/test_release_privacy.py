"""配布の来歴匿名化と、コピー済みバイトコードのパス除去を確かめる。"""
from __future__ import annotations

import hashlib
import json
import marshal
from pathlib import Path
import py_compile
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))
from build_bundle import precompile
from release_privacy import portable_provenance, refresh_model_hashes, sanitize_bundle_provenance


def test_provenance_preserves_fields_hashes_and_relative_members() -> None:
    original = {'input_root': '/mnt/c/Users/ryouj/work/data', 'members': [{'name': 'seed/model.joblib', 'sha256': 'abc'}],
                'input_sha256': {'C:/puyo1_work/run/rows.json': 'def'}, 'count': 17}
    result = portable_provenance(original)
    assert result['input_root'] == '<build>/work/data'
    assert result['members'] == original['members']
    assert result['input_sha256'] == {'<build>/run/rows.json': 'def'}
    assert result['count'] == 17
    assert original['input_root'].startswith('/mnt/')


def test_bundle_copy_keeps_model_bytes_and_refreshes_manifest_hash(tmp_path: Path) -> None:
    models = tmp_path / 'app/models/test'
    models.mkdir(parents=True)
    model = models / 'seed.joblib'
    model.write_bytes(b'unchanged-model')
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = models / 'manifest.json'
    manifest.write_text(json.dumps({'input_root': 'C:/Users/ryouj/work', 'members': [{'name': model.name, 'sha256': digest}]}))
    sanitize_bundle_provenance(tmp_path)
    data = json.loads(manifest.read_text())
    assert hashlib.sha256((models / data['members'][0]['name']).read_bytes()).hexdigest() == digest
    assert data['input_root'] == '<build>/work'
    assert refresh_model_hashes(tmp_path, {'test/manifest.json': 'old'})['test/manifest.json'] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_recompile_replaces_existing_absolute_filename(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / 'app/module.py'
    source.parent.mkdir()
    source.write_text('def value():\n    return 42\n')
    compiled = Path(py_compile.compile(str(source), dfile='C:/Users/ryouj/old/module.py'))
    real_run = subprocess.run
    def run_with_current_python(command: list[str], **kwargs):
        return real_run([sys.executable, *command[1:]], **kwargs)
    monkeypatch.setattr(subprocess, 'run', run_with_current_python)
    precompile(tmp_path / 'python', source.parent)
    code = marshal.loads(compiled.read_bytes()[16:])
    assert code.co_filename.replace('\\', '/') == 'app/module.py'
    namespace: dict = {}
    exec(code, namespace)
    assert namespace['value']() == 42
    assert namespace['value'].__code__.co_filename == code.co_filename


def test_direct_url_keeps_archive_integrity(tmp_path: Path) -> None:
    path = tmp_path / 'python/Lib/site-packages/demo.dist-info/direct_url.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'url': 'file:///C:/puyo1_work/cache/demo.whl', 'archive_info': {'hash': 'sha256=abc'}}))
    sanitize_bundle_provenance(tmp_path)
    data = json.loads(path.read_text())
    assert data['url'] == '<build>/cache/demo.whl'
    assert data['archive_info'] == {'hash': 'sha256=abc'}
