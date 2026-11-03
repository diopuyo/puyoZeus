"""収集ラッパーが本番認識の全既定値を保持する回帰テスト。"""
from functools import wraps
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import collect_e34b as capture
from scripts import visualize_advantage_overlay as overlay
from src.production_config import recognition_load_default_kwargs
from scripts import collect_e34c as checked


def test_capture_preserves_effective_production_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """実描画の既定値解決を通し、保持印以外の全引数と採用値を照合する。"""
    original = capture.RecognitionPipeline.load_default
    signature = inspect.signature(original)
    effective: dict[str, Any] = {}

    @wraps(original)
    def spy(**kwargs: Any) -> Any:
        bound = signature.bind(**kwargs)
        bound.apply_defaults()
        effective.update(bound.arguments)
        return None

    monkeypatch.setattr(capture.RecognitionPipeline, 'load_default', spy)
    monkeypatch.setattr(capture.ExchangeEventRecorder, 'update', capture.ExchangeEventRecorder.update)
    monkeypatch.setattr(capture.ExchangeEventRecorder, 'write', capture.ExchangeEventRecorder.write)
    capture.install_capture(SimpleNamespace(ranges=(), source='unused'))
    assert inspect.signature(capture.RecognitionPipeline.load_default) == signature
    defaults = signature.bind()
    defaults.apply_defaults()
    adopted = recognition_load_default_kwargs()
    resolved = {key: overlay._pipeline_default(key) for key, value in defaults.arguments.items()
                if isinstance(value, bool)}
    capture.RecognitionPipeline.load_default(**dict(resolved, **adopted))
    expected = dict(defaults.arguments, **adopted, enable_landing_chain_record_hold=True,
                    enable_chain_active_record_hold=True)
    assert effective == expected
    assert all(effective[key] == value for key, value in adopted.items())


@pytest.mark.parametrize('field', checked.CORE_FIELDS)
def test_prefix_mismatch_stops_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str,
) -> None:
    """盤面・状態・得点・NEXTのどれが違っても300行目で全収集を止める。"""
    side = {key: None for key in checked.CORE_FIELDS}
    old = dict(kind='update', args={'tuple': [dict(namespace={
        'p1': dict(namespace=side), 'p2': dict(namespace=side)}), None, None, 0.]})
    monkeypatch.setattr(checked, 'OUT', tmp_path)
    monkeypatch.setattr(checked, 'updates', lambda _: iter([old]*checked.PREFIX_ROWS))
    check = checked.InputCheck('test')
    for _ in range(checked.PREFIX_ROWS-1):
        check.observe(old)
    changed = dict(old, args={'tuple': [dict(namespace={
        'p1': dict(namespace=dict(side, **{field: 'changed'})),
        'p2': dict(namespace=side)}), None, None, 0.]})
    with pytest.raises(RuntimeError, match='最初の300行'):
        check.observe(changed)
    assert (tmp_path/'STOP.json').exists()


def test_full_comparison_excludes_only_hold() -> None:
    """保持印は許可し、欠測列・追加列・得点差を黙って除かない。"""
    assert checked.differences({'score': 3}, {'score': 3, 'prefire_origin_hold': True}) == []
    assert checked.differences({'score': 3}, {'score': 4}) == ['.score']
    assert checked.differences({'midchain_board': None}, {}) == ['.midchain_board']


def test_record_schema_preserves_e31_contract() -> None:
    """保存対象はE31と揃え、保持印と認識内容を変更しない。"""
    side = SimpleNamespace(chain_event=SimpleNamespace(trigger_sec=2., before_board='new'),
                           prefire_snapshot={'board': 'observed'}, prefire_origin_hold=True)
    other = SimpleNamespace(chain_event=None)
    row = dict(kind='update', args=(SimpleNamespace(p1=side, p2=other), None, None, 3., 1))
    checked.record_schema(row, {(1, 0, 2.)})
    assert side.prefire_snapshot == {'board': 'observed'}
    assert side.prefire_origin_hold and side.midchain_board is None
    assert not hasattr(side.chain_event, 'before_board')
    checked.record_schema(row, set())
    assert not hasattr(side, 'prefire_snapshot')


def test_review_schema_keeps_recorded_origin() -> None:
    """E14由来のreviewは既存のbefore_boardを失わない。"""
    side = SimpleNamespace(chain_event=SimpleNamespace(trigger_sec=2., before_board='observed'))
    other = SimpleNamespace(chain_event=None)
    row = dict(kind='update', args=(SimpleNamespace(p1=side, p2=other), None, None, 3., 1))
    checked.record_schema(row, set(), keep_before_board=True)
    assert side.chain_event.before_board == 'observed'


@pytest.mark.parametrize('present', [False, True])
def test_detect_original_record_schema(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    """旧入力の通知列の有無を調べ、値の複製はしない。"""
    event = dict(namespace={'before_board': None} if present else {})
    row = dict(args={'tuple': [dict(namespace={
        'p1': dict(namespace=dict(chain_event=event)),
        'p2': dict(namespace=dict(chain_event=None))})]})
    def records(source: str) -> Any:
        yield row
    monkeypatch.setattr(checked, 'updates', records)
    assert checked.has_before_board('test') is present


def test_command_preserves_original_evaluation_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """認識以外の付随列も再現するため、旧収集の評価スイッチを増減しない。"""
    import json
    monkeypatch.setattr(capture, 'ROOT', tmp_path)
    monkeypatch.setattr(capture, 'original_status', lambda source: tmp_path/source/'status.json')
    for source in (*capture.SOURCES, 'zenchi', 'review'):
        status = capture.original_status(source) if source != 'review' else (
            capture.ROOT/'logs/review_zenchi_g41_43_e14/status.json')
        old = ['--worker', '--video', 'original.mp4', '--out', 'original/overlay.mp4',
               '--exchange-event-model-dir', f'original_{source}', '--production-recognition',
               '--start-sec', '25', '--warmup-sec', '30', '--review-data-panel']
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps(dict(command=['python', '-m', 'module', *old])), encoding='utf-8')
        new = checked.command(source)
        ignored = {'--worker', '--no-render', '--review-data-panel'}
        old_flags = {value for value in old if value.startswith('--')} - ignored
        new_flags = {value for value in new if value.startswith('--')} - ignored
        assert old_flags == new_flags
        for flag in old_flags - set(checked.OUTPUT_FLAGS):
            index = old.index(flag)+1
            if index < len(old) and not old[index].startswith('--'):
                assert new[new.index(flag)+1] == old[index]
