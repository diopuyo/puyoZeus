"""HiddenRowFineTuner: hidden row 推論の確率を Platt scaling で補正.

実装方針 (Path B = calibration):
    既存の `infer_hidden_row()` のヒューリスティック確率 (例: 1/3 or 1.0) を
    観測 reveal 確率に合わせる「再キャリブレーション」を行う。
    モデル本体は触らず、後 hook で sigmoid(a*p + b) を適用する。

なぜ Calibration か:
    - 既存ヒューリスティックは「列候補 N 個のうち 1 つ」で probability 1/N と
      固定されているが、実際には close cell や row=HIDDEN_ROWS との一致度で
      予測の信頼度が変化する
    - 観測した reveal 結果を使い、predicted vs observed を回帰し、
      P_calibrated = sigmoid(a * P_heuristic + b) で再較正する

入力 PseudoLabelSample (component="hidden_row"):
    input_data["predicted_dist"]: {color: prob}
    input_data["predicted_color"]: 最尤色
    label: observed_color (実 reveal 色)
    metadata["match"]: bool

出力:
    data/verify/hidden_row_calibration.json:
        {"a": float, "b": float, "n_samples": int,
         "brier_before": float, "brier_after": float}

依存:
    sklearn.linear_model.LogisticRegression が望ましいが、
    scikit-learn が無くても動作するよう scipy/numpy だけで Newton 反復を
    実装してフォールバック。
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.hidden_row_inferrer import DEFAULT_CALIBRATION_PATH
from src.self_supervised.online_fine_tuner import OnlineFineTuner
from src.self_supervised.pseudo_label import PseudoLabelSample

# ============================
# 定数
# ============================

# fine_tune に必要な最小サンプル数
MIN_SAMPLES_FOR_FIT: int = 8

# Platt scaling Newton 反復の最大回数
PLATT_MAX_ITER: int = 100

# 収束判定の loss delta
PLATT_TOL: float = 1e-6

# default backup suffix
DEFAULT_BACKUP_SUFFIX: str = ".bak"


# ============================
# Fine Tuner
# ============================


class HiddenRowFineTuner(OnlineFineTuner):
    """hidden_row_inferrer の出力確率を Platt scaling で補正する fine-tuner."""

    def __init__(
        self,
        calibration_path: Path | str = DEFAULT_CALIBRATION_PATH,
        backup_suffix: str = DEFAULT_BACKUP_SUFFIX,
    ) -> None:
        self._path = Path(calibration_path)
        self._backup_suffix = str(backup_suffix)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_done: bool = False

    # ------------------------------------------------------------------
    # OnlineFineTuner API
    # ------------------------------------------------------------------

    def fine_tune(
        self, samples: list[PseudoLabelSample],
    ) -> dict[str, Any]:
        """擬似ラベルから Platt scaling のパラメータ a, b を学習し保存する.

        Returns:
            {brier_before, brier_after, a, b, n_samples}
        """
        pairs = self._extract_pairs(samples)
        if len(pairs) < MIN_SAMPLES_FOR_FIT:
            return {
                "n_samples": len(pairs),
                "brier_before": None,
                "brier_after": None,
                "a": None,
                "b": None,
                "skipped_reason": "not_enough_samples",
            }
        ps, ys = self._to_arrays(pairs)
        brier_before = float(np.mean((ps - ys) ** 2))
        a, b = self._fit_platt(ps, ys)
        # 学習後 brier
        ps_calibrated = _sigmoid_array(a * ps + b)
        brier_after = float(np.mean((ps_calibrated - ys) ** 2))
        # 保存 (backup → 上書き)
        if not self._backup_done and self._path.exists():
            self._make_backup()
        self._save_calibration(a, b, brier_before, brier_after, len(pairs))
        return {
            "n_samples": int(len(pairs)),
            "brier_before": brier_before,
            "brier_after": brier_after,
            "a": float(a),
            "b": float(b),
        }

    def rollback(self) -> None:
        """backup から復元する.

        backup が無ければ calibration ファイルを削除して既定挙動に戻す.
        """
        backup_path = self._path.with_suffix(
            self._path.suffix + self._backup_suffix,
        )
        if backup_path.exists():
            shutil.copy2(backup_path, self._path)
            return
        # backup 無し → ファイル削除でデフォルトへ戻す
        if self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass
        self._backup_done = False

    # ------------------------------------------------------------------
    # 内部 helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pairs(
        samples: list[PseudoLabelSample],
    ) -> list[tuple[float, int]]:
        """サンプル列から (predicted_prob_for_observed, hit{0,1}) 列を作る.

        predicted_prob_for_observed:
            元の predicted_dist[observed_color] (色付き確率).
            欠落 (色が dist に無い) なら 0.0.
        hit:
            metadata["match"] か、predicted_color == label で再判定.
        """
        out: list[tuple[float, int]] = []
        for s in samples:
            if s.component != "hidden_row":
                continue
            input_data = s.input_data
            if not isinstance(input_data, dict):
                continue
            dist = input_data.get("predicted_dist")
            if not isinstance(dist, dict):
                continue
            try:
                observed = int(s.label)
            except (TypeError, ValueError):
                continue
            # dist のキーは int だが JSON 経由で str の場合あり
            p_observed = _lookup_color_prob(dist, observed)
            if not (0.0 <= p_observed <= 1.0):
                continue
            hit = 1 if bool(
                s.metadata.get("match")
                if isinstance(s.metadata, dict) else False
            ) else 0
            # metadata["match"] が無い場合は predicted_color == observed で再判定
            if not isinstance(s.metadata, dict) or "match" not in s.metadata:
                pred_color = input_data.get("predicted_color")
                hit = 1 if pred_color == observed else 0
            out.append((float(p_observed), int(hit)))
        return out

    @staticmethod
    def _to_arrays(
        pairs: list[tuple[float, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        ps = np.array([p for p, _ in pairs], dtype=np.float64)
        ys = np.array([y for _, y in pairs], dtype=np.float64)
        # 数値安定化: 0/1 を端から少し離す
        ps = np.clip(ps, 1e-6, 1.0 - 1e-6)
        return ps, ys

    @staticmethod
    def _fit_platt(
        ps: np.ndarray, ys: np.ndarray,
    ) -> tuple[float, float]:
        """Platt scaling: y ~ Bernoulli(sigmoid(a * p + b)).

        Newton-Raphson で対数尤度最大化.
        """
        a, b = 1.0, 0.0
        for _ in range(PLATT_MAX_ITER):
            z = a * ps + b
            sig = _sigmoid_array(z)
            # 一階微分 (a, b)
            err = sig - ys
            grad_a = float(np.sum(err * ps))
            grad_b = float(np.sum(err))
            # 二階微分 (Hessian)
            w = sig * (1.0 - sig)
            h_aa = float(np.sum(w * ps * ps)) + 1e-9
            h_bb = float(np.sum(w)) + 1e-9
            h_ab = float(np.sum(w * ps))
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            # 逆行列で更新方向
            inv_aa = h_bb / det
            inv_bb = h_aa / det
            inv_ab = -h_ab / det
            da = inv_aa * grad_a + inv_ab * grad_b
            db = inv_ab * grad_a + inv_bb * grad_b
            a_new = a - da
            b_new = b - db
            if (
                abs(a_new - a) < PLATT_TOL
                and abs(b_new - b) < PLATT_TOL
            ):
                a, b = a_new, b_new
                break
            a, b = a_new, b_new
        return float(a), float(b)

    def _save_calibration(
        self,
        a: float,
        b: float,
        brier_before: float,
        brier_after: float,
        n_samples: int,
    ) -> None:
        """calibration JSON を書き出す."""
        data = {
            "a": float(a),
            "b": float(b),
            "n_samples": int(n_samples),
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)
        self._backup_done = True

    def _make_backup(self) -> None:
        """既存 calibration ファイルを backup."""
        backup_path = self._path.with_suffix(
            self._path.suffix + self._backup_suffix,
        )
        try:
            shutil.copy2(self._path, backup_path)
        except OSError:
            pass


# ============================
# helper
# ============================


def _sigmoid_array(z: np.ndarray) -> np.ndarray:
    """配列版 sigmoid (overflow ガード付き)."""
    z_clipped = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _sigmoid_scalar(x: float) -> float:
    if x < -50.0:
        return 0.0
    if x > 50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _lookup_color_prob(dist: dict, color: int) -> float:
    """dist[color] を返す. キーが str でも int でも対応."""
    if color in dist:
        try:
            return float(dist[color])
        except (TypeError, ValueError):
            return 0.0
    skey = str(color)
    if skey in dist:
        try:
            return float(dist[skey])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


__all__ = [
    "DEFAULT_BACKUP_SUFFIX",
    "HiddenRowFineTuner",
    "MIN_SAMPLES_FOR_FIT",
    "PLATT_MAX_ITER",
    "PLATT_TOL",
]
