"""visualize_advantage_overlay.generate() への Platt 校正組み込みの配線テスト。

重い動画処理・モデル学習を避けるため、generate() が「校正器ロードを
処理開始の一番最初(cv2.VideoCapture より前)に行い、欠損時は即座に例外を
送出する(fail-fast)」という配線だけを検証する。実際の校正の数値挙動は
tests/test_probability_calibration.py・tests/test_advantage_components.py で
別途検証済み。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.probability_calibration import CalibrationFileMissingError  # noqa: E402


def test_generate_raises_before_video_open_when_calibration_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enable_platt_calibration=True かつ校正器ファイル欠損 → cv2.VideoCapture
    に到達する前に CalibrationFileMissingError を送出する(fail-fast、
    重い動画処理・モデル学習を無駄にしないことの配線確認)。
    """
    missing_path = tmp_path / "no_such_platt_calibration.json"
    monkeypatch.setattr(vao, "PLATT_CALIBRATION_PATH", missing_path)

    def _fail_if_called(*_a: object, **_k: object) -> None:
        raise AssertionError("_train_model が呼ばれた = fail-fast 配線が壊れている")

    monkeypatch.setattr(vao, "_train_model", _fail_if_called)
    with pytest.raises(CalibrationFileMissingError):
        vao.generate(
            Path("dummy_video_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, enable_platt_calibration=True,
        )


def test_generate_does_not_require_calibration_file_when_flag_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enable_platt_calibration=False (従来挙動) なら校正器ファイル欠損でも
    エラーにならず先へ進む(後方互換、旧挙動の完全再現)。
    _train_model 以降は重いため、そこに到達したら成功とみなし打ち切る。
    """
    missing_path = tmp_path / "no_such_platt_calibration.json"
    monkeypatch.setattr(vao, "PLATT_CALIBRATION_PATH", missing_path)

    class _ReachedTrainModel(Exception):
        pass

    def _reached(*_a: object, **_k: object) -> None:
        raise _ReachedTrainModel()

    monkeypatch.setattr(vao, "_train_model", _reached)
    with pytest.raises(_ReachedTrainModel):
        vao.generate(
            Path("dummy_video_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, enable_platt_calibration=False,
        )
