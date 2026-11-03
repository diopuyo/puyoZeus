"""既存実行系のコード同値と、表示がnullの初期原票を検収する。"""
from __future__ import annotations
import importlib.util
import json
import marshal
from pathlib import Path
import sys
from hashlib import sha256

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT/'packaging'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pyc_allows_only_header_and_filename_changes() -> None:
    check = module('obs_runtime_check')
    prefix = importlib.util.MAGIC_NUMBER
    code = compile('def f(x):\n    return x + 3\n', 'packaged/path.py', 'exec')
    other = compile('def f(x):\n    return x + 3\n', 'existing/path.py', 'exec')
    changed = compile('def f(x):\n    return x + 4\n', 'existing/path.py', 'exec')
    left = prefix+b'\0'*12+marshal.dumps(code)
    assert check.equivalent_pyc(left, prefix+b'\1'*12+marshal.dumps(other))
    assert not check.equivalent_pyc(left, prefix+b'\1'*12+marshal.dumps(changed))
    assert not check.equivalent_pyc(left, b'bad')


def test_runtime_rejects_source_changes_even_with_equal_pyc(tmp_path: Path) -> None:
    check = module('obs_runtime_check')
    bundle, runtime = tmp_path/'bundle', tmp_path/'runtime'
    (bundle/'python/__pycache__').mkdir(parents=True)
    (runtime/'__pycache__').mkdir(parents=True)
    source = b'x = 3\n'
    name = '__pycache__/example.cpython-312.pyc'
    code = compile(source, 'packaged/example.py', 'exec')
    cached = importlib.util.MAGIC_NUMBER+b'\0'*12+marshal.dumps(code)
    (bundle/'python'/name).write_bytes(cached)
    (runtime/name).write_bytes(importlib.util.MAGIC_NUMBER+b'\1'*12+
        marshal.dumps(compile(source, 'existing/example.py', 'exec')))
    (runtime/'example.py').write_bytes(source)
    files = {'example.py': sha256(source).hexdigest(), name: sha256(cached).hexdigest()}
    (bundle/'obs_runtime_manifest.json').write_text(json.dumps(dict(files=files)), encoding='utf-8')
    result = check.validate(bundle, runtime)
    assert result['passed'] and result['exact_files'] == 1
    assert len(result['pyc_metadata_only']) == 1
    (runtime/'example.py').write_bytes(b'x = 4\n')
    with pytest.raises(ValueError, match='実行系の不一致'):
        check.validate(bundle, runtime)


def test_initial_null_frames_are_unavailable() -> None:
    valid = module('smoke_live_obs_20261020').valid_evaluations
    assert valid(dict(evaluations=None)) == (False, False, False)
    assert valid(dict(evaluations=dict(practical=None, v49f=None, ama_plain=None))) == (False, False, False)
    assert valid(dict(evaluations=dict(practical=dict(p1_win_probability=0.),
        v49f=dict(p1=1.), ama_plain=dict(p1=.5)))) == (True, True, True)


def test_smoke_failure_is_saved_in_a_fresh_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    smoke = module('smoke_live_obs_20261020')
    monkeypatch.setattr(smoke, 'arguments', lambda: SimpleNamespace(out=tmp_path))

    def failed(options: object) -> None:
        raise OSError('試験用の保存失敗')

    monkeypatch.setattr(smoke, 'main', failed)
    with pytest.raises(OSError, match='保存失敗'):
        smoke.run_cli()
    saved = json.loads((tmp_path/'fatal_receipt.json').read_text(encoding='utf-8'))
    assert saved['exception'] == 'OSError' and '試験用の保存失敗' in saved['stack']


def test_bootstrap_removes_old_app_and_checks_running_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    bootstrap = module('obs_runtime_bootstrap')
    runtime = Path(sys.executable).parent
    old_app = (runtime.parent/'app').resolve()
    monkeypatch.setattr(sys, 'path', [str(old_app), *sys.path])
    monkeypatch.delitem(sys.modules, 'src', raising=False)
    monkeypatch.setitem(sys.modules, 'obs_runtime_check',
                        SimpleNamespace(validate=lambda bundle, runtime: dict(passed=True)))
    value = bootstrap.prepare(tmp_path, runtime)
    assert value['passed'] and value['dont_write_bytecode']
    assert sys.path[0] == str(tmp_path/'app')
    assert all(Path(path).resolve() != old_app for path in sys.path)
    with pytest.raises(ValueError, match='Pythonが一致しない'):
        bootstrap.prepare(tmp_path, tmp_path/'wrong_runtime')


def test_bootstrap_normal_exit_has_no_failure_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    bootstrap = module('obs_runtime_bootstrap')
    monkeypatch.setattr(bootstrap, 'arguments',
                        lambda: SimpleNamespace(receipt=tmp_path/'runtime_receipt.json'))

    def finished(options: object) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(bootstrap, 'main', finished)
    with pytest.raises(SystemExit) as result:
        bootstrap.run_cli()
    assert result.value.code == 0 and not (tmp_path/'runtime_failure.json').exists()
