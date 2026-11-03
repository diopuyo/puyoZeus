"""正式版のモデル表記と利用者向け文書を検証する。"""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock

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
    for required in ('puyoZeus Live.vbs', '1920', '1080', 'SmartScreen',
                     'logs/reports', '.invalid', '非公式', 'unofficial', 'English'):
        assert required in guide
    for obsolete in ('preview.8', 'Inno Setup', 'D:', 'v6', '12 ポイント'):
        assert obsolete not in guide
