"""確率の Platt scaling 後段校正 (stateless)。

## 背景
model_indicator_win.py の HistGBC (全指標特徴 + 66動画データ) は系統的に
自信過剰と実測された(終盤で予測0.8台の局面群の実勝率が64.1%等)。
Platt scaling (全位相共通) を後段に挟むことで ECE が改善することを
scripts/_calibration_fit_2026-07-29.py (コミット7a42784) で実測済み
(ECE全体 0.0264→0.0189、終盤 0.0559→0.0354、user承認済み)。

## 設計
Platt scaling はロジスティック回帰1本 (logit(raw_p) を1特徴量とする) であり、
傾き a・切片 b の2係数だけで完全に記述できる:

    calibrated_p = sigmoid(a * logit(raw_p) + b)

推論時にこの2係数さえあれば良いため、本モジュールは sklearn に依存しない
(軽量・高速・stateless)。係数の学習 (sklearn LogisticRegression) は
scripts/fit_platt_calibration.py が担い、本モジュールは
「学習済み係数の保存・読込・適用」のみを担当する(責務分離)。

## 校正器欠損時の方針 (黙って未校正で通すのは禁止)
load_platt_calibration(path, required=True) が既定。ファイルが無ければ
CalibrationFileMissingError を送出する。呼出元 (例:
scripts/visualize_advantage_overlay.py) は動画処理ループの外側・処理開始前に
一度だけこの関数を呼ぶ設計のため、ここで即座に例外を出しても
「重い動画処理を無駄にする」リスクは無い(fail-fast)。
required=False を明示指定した場合のみ、警告を出した上で None を返す
(呼出元が独自にフォールバック方針を決められる救済経路)。
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# logit変換で0/1近傍が発散しないためのクリップ幅
# (scripts/_calibration_fit_2026-07-29.py の LOGIT_EPS と同一値)
CALIBRATION_LOGIT_EPS: float = 1e-6

# 恒等変換 (a=1, b=0) とみなす許容誤差 (回帰テスト用)
IDENTITY_TOLERANCE: float = 1e-9


class CalibrationFileMissingError(RuntimeError):
    """校正器ファイルが見つからない場合に送出する例外。

    「黙って未校正のまま通す」ことを防ぐためのガード
    (呼出元が明示的に enable_platt_calibration=False を指定しない限り
    校正なしで処理を続けさせない)。
    """


@dataclass(frozen=True)
class PlattCalibrationParams:
    """Platt scaling の学習済み係数。

    calibrated_logit = a * raw_logit + b。meta には学習に使った
    データ・日付・ECE 等の由来情報を保持する(監査用、必須ではない)。
    """

    a: float
    b: float
    meta: dict[str, Any] = field(default_factory=dict)


def _clip_prob(p: float) -> float:
    """0/1 近傍をクリップする (logit 発散防止)。"""
    return min(1.0 - CALIBRATION_LOGIT_EPS, max(CALIBRATION_LOGIT_EPS, p))


def _logit(p: float) -> float:
    """クリップ済み確率を logit 変換する。"""
    p_clip = _clip_prob(p)
    return math.log(p_clip / (1.0 - p_clip))


def _sigmoid(x: float) -> float:
    """logit を確率へ戻す (指数オーバーフロー回避のため符号で分岐)。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_platt_calibration(raw_p: float, params: PlattCalibrationParams) -> float:
    """生の予測確率を Platt scaling で校正する。0〜1 にクリップして返す (stateless)。"""
    calibrated_logit = params.a * _logit(raw_p) + params.b
    calibrated_p = _sigmoid(calibrated_logit)
    return min(1.0, max(0.0, calibrated_p))


def is_identity_calibration(params: PlattCalibrationParams) -> bool:
    """a=1, b=0 (恒等変換) かどうかを判定する (回帰テスト・切り戻し確認用)。"""
    return (
        abs(params.a - 1.0) < IDENTITY_TOLERANCE
        and abs(params.b) < IDENTITY_TOLERANCE
    )


def save_platt_calibration(params: PlattCalibrationParams, path: Path) -> None:
    """係数 + メタ情報を JSON で保存する (pickle 不使用、可読形式)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "platt_scaling_common",
        "a": params.a,
        "b": params.b,
        "meta": params.meta,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_platt_calibration(
    path: Path, *, required: bool = True
) -> PlattCalibrationParams | None:
    """Platt 校正器をロードする (欠損時の挙動は required で制御)。

    required=True (既定): ファイルが無ければ CalibrationFileMissingError を送出。
    required=False: ファイルが無ければ警告を出して None を返す。
    """
    if not path.exists():
        msg = (
            f"Platt校正器ファイルが見つかりません: {path}"
            " (校正なしで進める場合は enable_platt_calibration=False を指定)"
        )
        if required:
            raise CalibrationFileMissingError(msg)
        warnings.warn(msg, stacklevel=2)
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return PlattCalibrationParams(
        a=float(d["a"]), b=float(d["b"]), meta=dict(d.get("meta", {}))
    )
