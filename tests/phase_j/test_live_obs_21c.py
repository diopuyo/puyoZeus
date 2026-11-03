"""21cの明示採用と、合計処理量に依存しないHTTP実遅延の回帰検査。"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest


def test_native_preparation_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_obs_prepare import prepare_recognition
    from src.image_reader import ColorClassifier
    classifier = ColorClassifier()
    pipe = SimpleNamespace(_placement_reconcile=SimpleNamespace(hsv=classifier))
    monkeypatch.delenv('PUYO_OBS_PREPARE_EXACT', raising=False)
    prepare_recognition(pipe)
    assert not classifier._enable_native_hsv
    monkeypatch.setenv('PUYO_OBS_PREPARE_EXACT', '1')
    monkeypatch.setenv('PUYO_CNN_BACKEND', '')
    monkeypatch.setenv('PUYO_LIVE_V49F', '0')
    prepare_recognition(pipe)
    assert classifier._enable_native_hsv


def test_model_root_cache_keeps_working_directories_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_obs_prepare import resolve_at, resolved_model_root
    resolve_at.cache_clear()
    first, second = tmp_path/'a', tmp_path/'b'
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    assert resolved_model_root(Path('models')) == first/'models'
    assert resolved_model_root(Path('models')) == first/'models'
    assert resolve_at.cache_info().hits == 1
    monkeypatch.chdir(second)
    assert resolved_model_root(Path('models')) == second/'models'


def test_cached_root_selects_identical_onnx_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import cnn_onnx
    from src.phase_j.live_obs_prepare import resolve_at
    resolve_at.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PUYO_LIVE_ONNX_PARALLEL', '0')
    session = object()
    key = f'{Path("models/onnx").resolve()}|fixed-model|None'
    monkeypatch.setattr(cnn_onnx, '_SESSIONS', {key: session})
    monkeypatch.delenv('PUYO_OBS_PREPARE_EXACT', raising=False)
    assert cnn_onnx.session_for('fixed-model') is session
    monkeypatch.setenv('PUYO_OBS_PREPARE_EXACT', '1')
    assert cnn_onnx.session_for('fixed-model') is session


def test_http_latency_uses_first_publication_completion(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'packaging'))
    from summarize_live_obs_actual_latency import publication_delays, summarize
    output = tmp_path/'output'
    output.mkdir()
    rows = [dict(kind='publication', game=0, frame=1, t_sec=1., captured_at=10.,
                 published_at=10.1, monotonic=10.6),
            dict(kind='publication', game=0, frame=1, t_sec=1., captured_at=10.,
                 published_at=11., monotonic=11.1),
            dict(kind='publication', game=0, frame=2, t_sec=1200., captured_at=12.,
                 published_at=12.1, monotonic=13.)]
    (output/'b3_time_gate.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    assert len(publication_delays(tmp_path)) == 1
    value = summarize(tmp_path)['capture_to_http_available_ms']
    assert value['p50'] == pytest.approx(600.)
    assert value['n'] == value['over_500'] == 1


def test_timing_preserves_deepcopied_calibrator_owner() -> None:
    from copy import deepcopy
    from src.phase_j.live_obs_recognition_timing import instrument_method
    class Calibrator:
        def __init__(self) -> None:
            self.count = 0
        def update(self) -> None:
            self.count += 1
    instrument_method(Calibrator, 'update', 'calibration_test')
    original = Calibrator()
    copied = deepcopy(original)
    copied.update()
    assert copied.count == 1 and original.count == 0


def test_b3_latency_uses_origin_instead_of_latest_recognition() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'packaging'))
    from summarize_live_obs_actual_latency import b3_input_delays
    rows = [dict(kind='result', game=1, frame=10, t_sec=1., captured_at=10., accepted=True),
            dict(kind='publication', game=1, frame=12, state_frame=10,
                 t_sec=1.2, captured_at=10.2, monotonic=10.7),
            dict(kind='publication', game=1, frame=13, state_frame=10,
                 t_sec=1.3, captured_at=10.3, monotonic=10.8)]
    result = b3_input_delays(rows)
    assert len(result) == 1 and result[0]['frame'] == 10
    assert result[0]['actual_ms'] == pytest.approx(700.)


def test_parity_excludes_only_measured_time() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'packaging'))
    from compare_live_recognition_20261021 import comparison_bytes
    before = b'{"t":1,"image_compute_ms":100,"board":[1,2]}'
    after = b'{"t":1,"image_compute_ms":10,"board":[1,2]}'
    assert comparison_bytes(before) == comparison_bytes(after)
    assert comparison_bytes(before) != comparison_bytes(after.replace(b'[1,2]', b'[1,3]'))
    assert comparison_bytes(before) != comparison_bytes(after.replace(b'"t":1', b'"t":2'))
