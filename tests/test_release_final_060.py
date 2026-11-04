"""正式版のモデル表記と利用者向け文書を検証する。"""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def test_model_label_cli_reaches_bundle(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / 'packaging'))
    spec = importlib.util.spec_from_file_location('release_final_test', ROOT / 'packaging/build_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    finish = Mock(return_value=Path('bundle'))
    monkeypatch.setattr(module, 'safe_output', lambda value: value)
    monkeypatch.setattr(module, 'prepare_inputs', lambda value: None)
    monkeypatch.setattr(module, 'finish_bundle', finish)
    monkeypatch.setattr(module, 'artifacts', lambda *args: {})
    monkeypatch.setattr(sys, 'argv', ['build', '--package-only', '--zip-only',
                                     '--version', '0.6.0', '--model-label', 'c2ens3-exm150s3'])
    module.main()
    assert finish.call_args.args[1] == '0.6.0'
    assert finish.call_args.args[-1] == 'c2ens3-exm150s3'


def test_zip_user_guide_covers_both_languages() -> None:
    guide = (ROOT / 'docs/DISTRIBUTION_APP_GUIDE.md').read_text('utf-8-sig')
    for required in ('puyoZeus Live.exe', '1920', '1080', 'SmartScreen',
                     'logs/reports', '.invalid', '非公式', 'unofficial', 'English'):
        assert required in guide
    for obsolete in ('preview.8', 'Inno Setup', 'D:', 'v6', '12 ポイント'):
        assert obsolete not in guide


def test_release_license_copy_is_repeatable(tmp_path: Path, monkeypatch: Any) -> None:
    """同じ原文を逆順に列挙しても、2回の包装の別名と内容が一致する。"""
    sys.path.insert(0, str(ROOT / 'packaging'))
    import build_release
    original_rglob = Path.rglob
    outputs = []
    for index in range(2):
        bundle = tmp_path / str(index)
        info = bundle / 'python/Lib/site-packages/numpy-2.4.4.dist-info'
        inputs = {'licenses/numpy/fft/pocketfft/LICENSE.md': 'public fft',
                  'licenses/numpy/random/LICENSE.md': 'public random',
                  'licenses/numpy/_core/LICENSE.md': 'internal core',
                  'licenses/numpy/ma/LICENSE': 'ma',
                  'licenses/numpy/_core/src/umath/svml/LICENSE': 'svml'}
        for name, content in inputs.items():
            source = info / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(content, encoding='utf-8')
        def ordered_rglob(path: Path, pattern: str) -> Any:
            return iter(sorted(original_rglob(path, pattern), reverse=bool(index)))
        monkeypatch.setattr(Path, 'rglob', ordered_rglob)
        build_release.copy_flat_release_licenses(bundle)
        outputs.append({path.relative_to(bundle / 'LICENSES').as_posix(): path.read_bytes()
                        for path in original_rglob(bundle / 'LICENSES', '*') if path.is_file()})
    assert outputs[0] == outputs[1] == {'numpy-2.4.4/LICENSE.md': b'public fft',
                                      'numpy-2.4.4/LICENSE': b'svml'}
