"""ONNX バックエンド (src/cnn_onnx.py、既定 OFF) の試験: 既定 OFF の bit-identical / fail-loud / 往復一致。"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from src import cnn_onnx  # noqa: E402
from src.patch_classifier import PATCH_RESIZE_H, PATCH_RESIZE_W, CnnPatchClassifier, CnnPatchClassifierLarge  # noqa: E402

BATCH = 32


def random_batch(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(BATCH, 6, PATCH_RESIZE_H, PATCH_RESIZE_W, generator=generator)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cnn_onnx.BACKEND_ENV, raising=False)
    cnn_onnx._SESSIONS.clear()


def test_default_off_is_bit_identical_to_the_torch_model() -> None:
    """環境変数が無ければ _forward は self._model の出力そのもの (同一テンソル値)。"""
    cnn = CnnPatchClassifier(seed=1)
    batch = random_batch()
    with torch.no_grad():
        assert torch.equal(cnn._forward(batch), cnn._model(batch))
    assert not cnn_onnx.backend_requested() and cnn_onnx._SESSIONS == {}


@pytest.mark.parametrize('value', ['', 'torch', '0', 'ONNX ', 'onnx'])
def test_backend_flag_parsing(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(cnn_onnx.BACKEND_ENV, value)
    assert cnn_onnx.backend_requested() == (value.strip().lower() == 'onnx')


def test_missing_onnx_is_loud_not_silent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ONNX 指定なのに対応ファイルが無い → 例外 (黙って torch に戻さない)。"""
    monkeypatch.setenv(cnn_onnx.BACKEND_ENV, 'onnx')
    monkeypatch.chdir(tmp_path)  # models/onnx が無い場所
    cnn = CnnPatchClassifier(seed=1)
    with pytest.raises(cnn_onnx.OnnxBackendError, match='ONNX がありません'):
        cnn._forward(random_batch())


def test_state_digest_tracks_weights_and_is_stable() -> None:
    a, b = CnnPatchClassifier(seed=1), CnnPatchClassifier(seed=2)
    assert cnn_onnx.state_digest(a._model.state_dict()) == cnn_onnx.state_digest(a._model.state_dict())
    assert cnn_onnx.state_digest(a._model.state_dict()) != cnn_onnx.state_digest(b._model.state_dict())
    large = CnnPatchClassifierLarge(seed=1)
    assert cnn_onnx.state_digest(large._model.state_dict()) != cnn_onnx.state_digest(a._model.state_dict())


@pytest.mark.parametrize('large', [False, True])
def test_export_roundtrip_matches_torch(large: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """書き出した ONNX を _forward 経由 (既定の models/onnx を相対で引く) で使い、torch と argmax が一致する。"""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    cnn = (CnnPatchClassifierLarge if large else CnnPatchClassifier)(seed=3)
    digest, path = cnn_onnx.export_state(cnn._model.state_dict(), tmp_path / 'models' / 'onnx')
    assert path.name == f'{digest}.onnx' and digest == cnn_onnx.state_digest(cnn._model.state_dict())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(cnn_onnx.BACKEND_ENV, 'onnx')
    batch = random_batch(seed=5)
    before = dict(cnn_onnx._CALLS)
    with torch.no_grad():
        expected = cnn._model(batch)
        actual = cnn._forward(batch)
    assert cnn_onnx._CALLS['calls'] == before['calls'] + 1  # 実際に ONNX で計算された証拠
    assert cnn_onnx._CALLS['patches'] == before['patches'] + BATCH
    assert torch.equal(expected.argmax(1), actual.argmax(1))
    assert float((expected - actual).abs().max()) < 1e-4  # 数値差はあるが微小 (bit 一致は主張しない)
