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


# =============================================================================
# 位相別 Platt scaling (2026-08-11 Phase1-2 追加)
# =============================================================================
# 背景 (memory project_calibration_overconfident_2026-07-29): 全位相共通 Platt
# (上記 PlattCalibrationParams 一式) は終盤の ECE が最も改善しにくい
# (終盤0.0559→0.0354、序盤0.0398→0.0159 ほどは縮まらない)。B-1 (対称化修正)
# + B-2 (進行度列 match_progress) で表示用モデル (tier1、
# scripts/visualize_advantage_overlay.py) が変わったため、進行度に応じて
# 3 つの Platt パラメータを切り替える構成を追加する (Phase1-2 ロードマップ)。
#
# 既存の PlattCalibrationParams / apply_platt_calibration / save/load 等は
# 一切変更しない (完全後方互換、既存の単一 Platt 経路は本追加の影響を受けない)。
# 学習は scripts/fit_phase_platt_calibration.py が担う (責務分離は単一Plattと同じ)。

# 位相ラベル一覧 (この順で表示・保存する)
PHASE_NAMES: tuple[str, ...] = ("序盤", "中盤", "終盤")

# 進行度 [0,1] の位相境界。 match_progress は「両者の盤面ぷよ総数の平均」
# という物理量 (学習データ由来の分位点ではない) なので、均等3分割という
# 最も単純な境界を採用する (シーン逆算でなく値域そのものから決めた定数)。
PHASE_BOUND_EARLY: float = 1.0 / 3.0
PHASE_BOUND_LATE: float = 2.0 / 3.0


def phase_label_for_progress(
    progress: float,
    early_bound: float = PHASE_BOUND_EARLY,
    late_bound: float = PHASE_BOUND_LATE,
) -> str:
    """進行度 [0,1] から位相ラベル (序盤/中盤/終盤) を返す (stateless純関数)。"""
    if progress <= early_bound:
        return "序盤"
    if progress <= late_bound:
        return "中盤"
    return "終盤"


@dataclass(frozen=True)
class PhaseCalibrationParams:
    """位相別 Platt scaling の学習済みパラメータ一式。

    Attributes:
        phases: 位相ラベル (PHASE_NAMES の要素) → PlattCalibrationParams。
        early_bound: 序盤/中盤の進行度境界。
        late_bound: 中盤/終盤の進行度境界。
        meta: 学習由来情報 (監査用)。
    """

    phases: dict[str, PlattCalibrationParams]
    early_bound: float = PHASE_BOUND_EARLY
    late_bound: float = PHASE_BOUND_LATE
    meta: dict[str, Any] = field(default_factory=dict)


def select_phase_platt(
    progress: float, params: PhaseCalibrationParams
) -> PlattCalibrationParams:
    """進行度から適用すべき位相の PlattCalibrationParams を選ぶ (stateless)。"""
    label = phase_label_for_progress(progress, params.early_bound, params.late_bound)
    return params.phases[label]


def apply_phase_platt_calibration(
    raw_p: float, progress: float, params: PhaseCalibrationParams
) -> float:
    """進行度に応じた位相別 Platt scaling を適用する (0〜1 にクリップ、stateless)。"""
    return apply_platt_calibration(raw_p, select_phase_platt(progress, params))


def save_phase_platt_calibration(params: PhaseCalibrationParams, path: Path) -> None:
    """位相別 Platt 係数一式を JSON で保存する (pickle 不使用)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "platt_scaling_phase",
        "early_bound": params.early_bound,
        "late_bound": params.late_bound,
        "phases": {
            name: {"a": p.a, "b": p.b, "meta": p.meta}
            for name, p in params.phases.items()
        },
        "meta": params.meta,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_phase_platt_calibration(
    path: Path, *, required: bool = True
) -> PhaseCalibrationParams | None:
    """位相別 Platt 校正器をロードする (欠損時の挙動は required で制御)。

    required=True (既定): ファイルが無ければ CalibrationFileMissingError を送出。
    required=False: ファイルが無ければ警告を出して None を返す。
    """
    if not path.exists():
        msg = (
            f"位相別Platt校正器ファイルが見つかりません: {path}"
            " (校正なしで進める場合は enable_phase_calibration=False を指定)"
        )
        if required:
            raise CalibrationFileMissingError(msg)
        warnings.warn(msg, stacklevel=2)
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    phases = {
        name: PlattCalibrationParams(
            a=float(v["a"]), b=float(v["b"]), meta=dict(v.get("meta", {})),
        )
        for name, v in d.get("phases", {}).items()
    }
    missing_phases = [p for p in PHASE_NAMES if p not in phases]
    if missing_phases:
        raise ValueError(f"位相別Platt校正器に不足位相があります: {missing_phases}")
    return PhaseCalibrationParams(
        phases=phases,
        early_bound=float(d.get("early_bound", PHASE_BOUND_EARLY)),
        late_bound=float(d.get("late_bound", PHASE_BOUND_LATE)),
        meta=dict(d.get("meta", {})),
    )
