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
        raise AssertionError("_acquire_model が呼ばれた = fail-fast 配線が壊れている")

    monkeypatch.setattr(vao, "_acquire_model", _fail_if_called)
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
    モデル確保 (`_acquire_model`、2026-08-14 成果物直読み追加により
    `_train_model` 直呼びから置き換わった) 以降は重いため、そこに到達したら
    成功とみなし打ち切る。
    """
    missing_path = tmp_path / "no_such_platt_calibration.json"
    monkeypatch.setattr(vao, "PLATT_CALIBRATION_PATH", missing_path)

    class _ReachedAcquireModel(Exception):
        pass

    def _reached(*_a: object, **_k: object) -> None:
        raise _ReachedAcquireModel()

    monkeypatch.setattr(vao, "_acquire_model", _reached)
    with pytest.raises(_ReachedAcquireModel):
        vao.generate(
            Path("dummy_video_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, enable_platt_calibration=False,
        )


# =============================================================================
# 位相別 Platt (2026-08-11 Phase1-2 追加) の配線テスト
# =============================================================================

def test_generate_raises_before_video_open_when_phase_calibration_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enable_phase_calibration=True かつ校正器ファイル欠損 → 動画を開く前に
    CalibrationFileMissingError (全位相共通 Platt と同じ fail-fast 設計)。
    """
    missing_path = tmp_path / "no_such_phase_platt_calibration.json"
    monkeypatch.setattr(vao, "PHASE_CALIBRATION_PATH", missing_path)

    def _fail_if_called(*_a: object, **_k: object) -> None:
        raise AssertionError("_acquire_model が呼ばれた = fail-fast 配線が壊れている")

    monkeypatch.setattr(vao, "_acquire_model", _fail_if_called)
    with pytest.raises(CalibrationFileMissingError):
        vao.generate(
            Path("dummy_video_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, enable_phase_calibration=True,
        )


def test_generate_does_not_require_phase_calibration_file_when_flag_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enable_phase_calibration=False (既定) なら校正器ファイル欠損でもエラーに
    ならない (後方互換、既存呼出元は挙動不変)。モデル確保 (`_acquire_model`、
    2026-08-14 成果物直読み追加) 以降は重いため到達したら成功とみなし打ち切る。
    """
    missing_path = tmp_path / "no_such_phase_platt_calibration.json"
    monkeypatch.setattr(vao, "PHASE_CALIBRATION_PATH", missing_path)

    class _ReachedAcquireModel(Exception):
        pass

    def _reached(*_a: object, **_k: object) -> None:
        raise _ReachedAcquireModel()

    monkeypatch.setattr(vao, "_acquire_model", _reached)
    with pytest.raises(_ReachedAcquireModel):
        vao.generate(
            Path("dummy_video_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, enable_phase_calibration=False,
        )


def test_generate_rejects_both_calibration_flags_simultaneously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全位相共通 Platt と位相別 Platt を同時 True にすると、校正器ロードや
    _train_model 呼び出しより前に ValueError で明示的に拒否する。
    """
    def _fail_if_called(*_a: object, **_k: object) -> None:
        raise AssertionError("校正器ロード/_train_model が呼ばれた = ガードが壊れている")

    monkeypatch.setattr(vao, "load_platt_calibration", _fail_if_called)
    monkeypatch.setattr(vao, "load_phase_platt_calibration", _fail_if_called)
    monkeypatch.setattr(vao, "_train_model", _fail_if_called)
    with pytest.raises(ValueError):
        vao.generate(
            Path("dummy_video_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
            enable_platt_calibration=True, enable_phase_calibration=True,
        )
