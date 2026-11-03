"""画素差分キャッシュの失効・微小変化・出力所有権を確認する。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.live_delta_cells import ExactPatchCache, FULL_REFRESH_SEC, KEY_LIMIT, read_boards
from src.phase_j.live_speed import validate


class Predictor:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, patches: list[np.ndarray]) -> np.ndarray:
        self.calls.append(len(patches))
        return np.asarray([[p.sum(), p.size] for p in patches], dtype=np.float32).reshape(-1, 2)


def patch(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_same_frame_dedup_and_order() -> None:
    cache, infer = ExactPatchCache(), Predictor()
    cache.begin(0.0, 'a', False)
    result = cache.predict([patch(1), patch(2), patch(1)], infer)
    np.testing.assert_array_equal(result[:, 0], [48, 96, 48])
    assert infer.calls == [2]
    cache.predict([patch(2)], infer)
    assert infer.calls == [2]


@pytest.mark.parametrize('temporal', [False, True])
def test_previous_frame(temporal: bool) -> None:
    cache, infer = ExactPatchCache(), Predictor()
    cache.begin(0.0, 'a', temporal)
    cache.predict([patch()], infer)
    cache.begin(0.1, 'a', temporal)
    cache.predict([patch()], infer)
    assert len(infer.calls) == (1 if temporal else 2)


@pytest.mark.parametrize('channel', [0, 1, 2])
def test_single_dn_change_never_reuses(channel: int) -> None:
    cache, infer = ExactPatchCache(), Predictor()
    cache.begin(0.0, 'a', True)
    cache.predict([patch()], infer)
    changed = patch()
    changed[0, 0, channel] = 1
    cache.begin(0.1, 'a', True)
    cache.predict([changed], infer)
    assert infer.calls == [1, 1]


@pytest.mark.parametrize('stamp,context', [(FULL_REFRESH_SEC, 'a'), (0.1, 'b'), (0.0, 'a'), (-1.0, 'a')])
def test_invalidation(stamp: float, context: str) -> None:
    cache, infer = ExactPatchCache(), Predictor()
    cache.begin(0.0, 'a', True)
    cache.predict([patch()], infer)
    cache.begin(stamp, context, True)
    cache.predict([patch()], infer)
    assert infer.calls == [1, 1]


def test_mutable_inputs_outputs_do_not_corrupt_cache() -> None:
    cache, infer = ExactPatchCache(), Predictor()
    cache.begin(0.0, 'a', True)
    source = patch()
    result = cache.predict([source], infer)
    source[:] = 1
    result[:] = -1
    cache.begin(0.1, 'a', True)
    np.testing.assert_array_equal(cache.predict([patch()], infer), [[0, 48]])
    assert infer.calls == [1]


def test_absent_patch_does_not_survive_extra_frame() -> None:
    cache, infer = ExactPatchCache(), Predictor()
    for stamp, value in [(0.0, 1), (0.1, 2), (0.2, 1)]:
        cache.begin(stamp, 'a', True)
        cache.predict([patch(value)], infer)
    assert infer.calls == [1, 1, 1]


def test_shape_is_part_of_key() -> None:
    cache, infer = ExactPatchCache(), Predictor()
    cache.begin(0.0, 'a', True)
    cache.predict([patch(), patch().reshape(2, 8, 3)], infer)
    assert infer.calls == [2]


def test_empty_and_limit() -> None:
    cache, infer = ExactPatchCache(), Predictor()
    assert cache.predict([], infer).shape == (0, 2)
    patches = [np.array([i], dtype=np.int32) for i in range(KEY_LIMIT + 1)]
    cache.predict(patches, infer)
    assert not cache.current


def test_flags_default_off_and_validate() -> None:
    assert validate({})['live_delta_cells'] is False
    assert validate({})['live_cnn_dedup'] is False
    with pytest.raises(ValueError):
        validate({'live_delta_cells': 1})


def test_scope_restores_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_DELTA_CELLS', '1')
    cnn = SimpleNamespace(_predict_proba_batch_uncached=Predictor())

    def fail(*args: object, **kwargs: object) -> None:
        assert cnn._live_delta_cache is not None
        raise ValueError('意図した失敗')

    pipe = SimpleNamespace(_reader=SimpleNamespace(read_both_boards=fail,
        _classifier=SimpleNamespace(_cnn=cnn)), _match_active_started_time=0.0)
    with pytest.raises(ValueError):
        read_boards(pipe, patch(), 0.0)
    assert cnn._live_delta_cache is None
    assert pipe._live_delta_cache is None


def test_off_uses_unmodified_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIVE_DELTA_CELLS', raising=False)
    monkeypatch.delenv('PUYO_LIVE_CNN_DEDUP', raising=False)
    pipe = SimpleNamespace(_reader=SimpleNamespace(read_both_boards=lambda *a, **k: 'original'))
    assert read_boards(pipe, patch(), 0.0) == 'original'
    assert not hasattr(pipe, '_live_delta_cache')


@pytest.mark.parametrize('change', ['match', 'calibration', 'recovery'])
def test_reader_context_invalidates_but_postprocessing_runs(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    monkeypatch.setenv('PUYO_LIVE_DELTA_CELLS', '1')
    infer = Predictor()
    cnn = SimpleNamespace(_predict_proba_batch_uncached=infer)
    hsv = SimpleNamespace(_ranges={1: (0, 10)})
    postprocess: list[bool] = []

    def read(*args: object, **kwargs: object) -> np.ndarray:
        postprocess.append(True)
        return cnn._live_delta_cache.predict([patch()], infer)

    pipe = SimpleNamespace(_reader=SimpleNamespace(read_both_boards=read,
        _classifier=SimpleNamespace(_cnn=cnn, _hsv=hsv)), _match_active_started_time=0.0)
    read_boards(pipe, patch(), 0.0)
    read_boards(pipe, patch(), 0.1)
    assert infer.calls == [1]
    if change == 'match':
        pipe._match_active_started_time = 0.2
    elif change == 'calibration':
        hsv._ranges[1] = (1, 11)
    else:
        pipe._live_delta_cache = None
    read_boards(pipe, patch(), 0.2)
    assert infer.calls == [1, 1]
    assert len(postprocess) == 3
