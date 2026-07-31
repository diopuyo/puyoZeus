"""src/probability_calibration.py の単体テスト (Platt scaling 後段校正)。

回帰防止の要点:
  - 単調性 (校正しても確率の大小関係は保たれる)
  - 0〜1 クリップ
  - a=1,b=0 (恒等変換) で入力が変わらない (= 校正なしと数値一致)
  - 校正器ファイル欠損時の挙動 (required True/False)
  - 保存 → 読込の往復一致
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probability_calibration import (  # noqa: E402
    CalibrationFileMissingError, PlattCalibrationParams,
    apply_platt_calibration, is_identity_calibration,
    load_platt_calibration, save_platt_calibration,
)


def test_identity_calibration_leaves_input_unchanged() -> None:
    """a=1, b=0 (恒等変換) なら任意の入力確率が変化しない。"""
    params = PlattCalibrationParams(a=1.0, b=0.0)
    for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        assert abs(apply_platt_calibration(p, params) - p) < 1e-6


def test_is_identity_calibration_detects_default() -> None:
    """is_identity_calibration が a=1,b=0 を正しく検出する。"""
    assert is_identity_calibration(PlattCalibrationParams(a=1.0, b=0.0))
    assert not is_identity_calibration(PlattCalibrationParams(a=0.8, b=-0.1))


def test_monotonic_for_typical_calibration() -> None:
    """典型的な校正係数 (自信過剰を弱める a<1) でも単調増加は保たれる。"""
    params = PlattCalibrationParams(a=0.7, b=-0.05)
    ps = [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]
    calibrated = [apply_platt_calibration(p, params) for p in ps]
    assert calibrated == sorted(calibrated)


def test_output_clipped_to_unit_interval() -> None:
    """出力は常に [0,1] に収まる (極端な a/b・入力でも)。"""
    params = PlattCalibrationParams(a=5.0, b=10.0)
    for p in (1e-9, 0.5, 1.0 - 1e-9):
        v = apply_platt_calibration(p, params)
        assert 0.0 <= v <= 1.0


def test_overconfident_high_prob_is_compressed_toward_half() -> None:
    """自信過剰補正 (a<1) は高信頼予測を0.5側へ圧縮する(実測傾向の回帰確認)。"""
    params = PlattCalibrationParams(a=0.7, b=0.0)
    calibrated = apply_platt_calibration(0.9, params)
    assert 0.5 < calibrated < 0.9


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """保存したJSONを読み込むと同じ a/b/meta が復元される。"""
    path = tmp_path / "platt.json"
    meta = {"n_samples": 1234, "source_csv": "dummy.csv", "ece_before": 0.05}
    params = PlattCalibrationParams(a=0.65, b=-0.02, meta=meta)
    save_platt_calibration(params, path)
    loaded = load_platt_calibration(path)
    assert loaded is not None
    assert loaded.a == pytest.approx(0.65)
    assert loaded.b == pytest.approx(-0.02)
    assert loaded.meta["n_samples"] == 1234
    assert loaded.meta["source_csv"] == "dummy.csv"


def test_load_missing_file_required_raises(tmp_path: Path) -> None:
    """required=True (既定) で校正器ファイルが無ければ例外を送出する(黙って未校正で通さない)。"""
    missing = tmp_path / "no_such_calibration.json"
    with pytest.raises(CalibrationFileMissingError):
        load_platt_calibration(missing)


def test_load_missing_file_optional_warns_and_returns_none(tmp_path: Path) -> None:
    """required=False なら警告付きで None を返す(呼出元がフォールバック可能)。"""
    missing = tmp_path / "no_such_calibration.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load_platt_calibration(missing, required=False)
    assert result is None
    assert len(caught) == 1
    assert "見つかりません" in str(caught[0].message)


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    """出力先ディレクトリが無くても自動作成される。"""
    path = tmp_path / "nested" / "dir" / "platt.json"
    save_platt_calibration(PlattCalibrationParams(a=1.0, b=0.0), path)
    assert path.exists()
