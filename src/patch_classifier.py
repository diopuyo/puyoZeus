"""
セルパッチ色分類器モジュール

ImageReader の HSV ルールベース分類器 (ColorClassifier) を差し替え可能にし、
学習可能な MLP / CNN ベース分類器を提供する。

分類器インタフェース:
    classify(bgr_patch: np.ndarray) -> int   (色コードを返す)

torch が未インストールなら CnnPatchClassifier はインポート不可にする。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VALID_COLORS,
    VISIBLE_ROWS,
)
from src.image_reader import BoardRegion, ColorClassifier

# ============================
# 定数定義
# ============================

# 学習・推論時のパッチリサイズ
PATCH_RESIZE_H: int = 8
PATCH_RESIZE_W: int = 8

# 特徴量次元 = H × W × 3 (BGR)
FEATURE_DIM_PATCH: int = PATCH_RESIZE_H * PATCH_RESIZE_W * 3

# cycle 32g (2026-05-19): 円形マスク = cell 四隅 (= 背景見える部分) を黒塗り
# して CNN 入力から背景情報を削減。 推論時/学習時の両方で適用される。
# global flag で制御 (= 学習時 ON ↔ 推論時 ON を必ず揃える)。
# default = False で backwards compat 維持。
USE_CIRCLE_MASK: bool = False
CIRCLE_RADIUS_RATIO: float = 0.45  # 中央 90% 直径 (= 四隅のみ削減)
_CIRCLE_MASK_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_circle_mask(h: int, w: int) -> np.ndarray:
    """中央円形を 1、 四隅を 0 とするマスクを cache 付きで返す."""
    key = (h, w)
    if key in _CIRCLE_MASK_CACHE:
        return _CIRCLE_MASK_CACHE[key]
    cy, cx = h / 2, w / 2
    r = min(h, w) * CIRCLE_RADIUS_RATIO
    yy, xx = np.ogrid[:h, :w]
    mask = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)
    _CIRCLE_MASK_CACHE[key] = mask
    return mask


def set_circle_mask_enabled(enabled: bool) -> None:
    """円形マスクの有効/無効を切り替える (= cycle 32g 用 global flag)."""
    global USE_CIRCLE_MASK
    USE_CIRCLE_MASK = bool(enabled)

# クラス定義 (内部 index → 色コード)
CLASS_INDEX_TO_COLOR: tuple[int, ...] = (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)
COLOR_TO_CLASS_INDEX: dict[int, int] = {
    c: i for i, c in enumerate(CLASS_INDEX_TO_COLOR)
}
NUM_CLASSES: int = len(CLASS_INDEX_TO_COLOR)

# MLP デフォルト構成
MLP_HIDDEN_SIZES: tuple[int, ...] = (64, 32)

# 学習ハイパーパラメータ
DEFAULT_EPOCHS: int = 60
DEFAULT_LR: float = 0.05
DEFAULT_BATCH: int = 32
DEFAULT_SEED: int = 42

# データ拡張用ノイズ強度
DEFAULT_NOISE_STD: float = 6.0   # 画素 (0-255) に対する標準偏差
DEFAULT_HUE_JITTER: int = 3


# ============================
# 特徴量抽出
# ============================


def patch_to_feature(bgr_patch: np.ndarray) -> np.ndarray:
    """
    BGR パッチを固定サイズにリサイズし、0-1 正規化したフラットベクトルを返す。

    Args:
        bgr_patch: shape=(H, W, 3) の uint8 BGR 画像。

    Returns:
        np.ndarray: shape=(FEATURE_DIM_PATCH,) の float64 配列。
    """
    if bgr_patch.size == 0:
        return np.zeros(FEATURE_DIM_PATCH, dtype=np.float64)
    resized = cv2.resize(
        bgr_patch, (PATCH_RESIZE_W, PATCH_RESIZE_H),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float64).flatten() / 255.0


# ============================
# 基底クラス
# ============================


class PatchClassifier(ABC):
    """パッチ色分類器の抽象基底。"""

    @abstractmethod
    def classify(self, bgr_patch: np.ndarray) -> int:
        """BGR パッチから色コードを返す (ColorClassifier 互換)。"""
        ...


class PuyoPresenceGate:
    """
    パッチが「puyo である」かを視覚特徴から判定する前段ゲート。

    puyo は以下の特徴を持つ:
        - 中央付近に暗い点 (目) が 1 個以上存在する
        - 3D 陰影による色のグラデーション (平坦色でない)
        - 画素サンプルに一定以上の彩度を持つ puyo 色が含まれる

    strict_pair_eyes=True の場合はさらに厳格に:
        - 上半分に水平ペアの 2 眼が存在する (致命列 X 印の排除に有効)
    """

    EYE_MIN_COUNT: int = 1          # puyo の目は 2 個だが片目だけ見える場合も許容
    EYE_AREA_MIN: int = 2
    EYE_AREA_RATIO_MAX: float = 0.2
    EYE_BRIGHTNESS_MAX: int = 80    # 暗点閾値 (眼の瞳)
    CENTER_MARGIN_RATIO: float = 0.12

    COLOR_STD_MIN: float = 12.0     # 3D 陰影による V 分散 (平坦 UI は低い)

    SATURATION_PIXEL_RATIO: float = 0.20  # 飽和 puyo 色画素の最低比率
    SATURATION_MIN: int = 80              # HSV S 閾値

    # 厳格モード (X印対策): 水平ペア眼のパラメータ
    strict_pair_eyes: bool = False
    PAIR_Y_TOLERANCE_RATIO: float = 0.20    # 2眼のY座標差の許容値 (中央領域の高さ比)
    PAIR_X_DISTANCE_MIN_RATIO: float = 0.25 # 2眼のX距離最小値 (中央領域の幅比)
    PAIR_X_DISTANCE_MAX_RATIO: float = 0.70 # 2眼のX距離最大値 (遠すぎは×印の端)
    PAIR_UPPER_HALF_RATIO: float = 0.55     # 眼は上半分~55%に位置すべき
    PAIR_EDGE_MARGIN_RATIO: float = 0.15    # 眼は端から15%以内にあってはいけない
    PAIR_SYMMETRY_TOLERANCE: float = 0.20   # 左右対称性 (|dx_left - dx_right|/w)

    def is_puyo(self, bgr_patch: np.ndarray) -> bool:
        """パッチが puyo であれば True。"""
        if bgr_patch.size == 0:
            return False
        h, w = bgr_patch.shape[:2]
        mh = int(h * self.CENTER_MARGIN_RATIO)
        mw = int(w * self.CENTER_MARGIN_RATIO)
        center = bgr_patch[mh:h-mh, mw:w-mw]
        if center.size == 0:
            return False

        # 厳格モード: 水平に並ぶ 2 眼が必須 (X 印や非puyo UI は通さない)
        if self.strict_pair_eyes:
            return self._has_horizontal_eye_pair(center)

        has_eyes = self._has_eyes(center)
        has_shading = self._has_shading(center)
        has_saturation = self._has_saturation(center)

        if has_eyes:
            return True
        return has_shading and has_saturation

    @classmethod
    def _has_eyes(cls, center_bgr: np.ndarray) -> bool:
        """中央領域に暗い点が 2 個以上あるか。"""
        gray = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < cls.EYE_BRIGHTNESS_MAX).astype(np.uint8) * 255
        num, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
        total_area = center_bgr.shape[0] * center_bgr.shape[1]
        count = sum(
            1 for i in range(1, num)
            if cls.EYE_AREA_MIN <= stats[i, cv2.CC_STAT_AREA]
                <= total_area * cls.EYE_AREA_RATIO_MAX
        )
        return count >= cls.EYE_MIN_COUNT

    @classmethod
    def _has_shading(cls, center_bgr: np.ndarray) -> bool:
        """3D 陰影の有無: HSV の V チャンネル標準偏差で判定。"""
        hsv = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2HSV)
        v_std = float(np.std(hsv[:, :, 2]))
        return v_std >= cls.COLOR_STD_MIN

    @classmethod
    def _has_saturation(cls, center_bgr: np.ndarray) -> bool:
        """飽和色画素が一定比率以上あるか (平坦背景・薄色UI を除外)。"""
        hsv = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2HSV)
        s_chan = hsv[:, :, 1]
        sat_ratio = float(np.mean(s_chan >= cls.SATURATION_MIN))
        return sat_ratio >= cls.SATURATION_PIXEL_RATIO

    def _has_horizontal_eye_pair(self, center_bgr: np.ndarray) -> bool:
        """
        上半分に水平に並んだ左右対称 2 眼があるか (X 印対策)。

        検出条件:
            - 暗点が 2 個以上存在
            - 各候補が中央領域に位置する (端から15%以上内側)
            - 上半分 (55%) に位置
            - 2 個の Y 差が小さい (水平)
            - X 距離は 25%~70% (近すぎず離れすぎず)
            - 左右の中心からの距離が対称
        """
        h, w = center_bgr.shape[:2]
        gray = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < self.EYE_BRIGHTNESS_MAX).astype(np.uint8) * 255
        num, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=4)
        total_area = h * w

        edge_x_min = w * self.PAIR_EDGE_MARGIN_RATIO
        edge_x_max = w * (1.0 - self.PAIR_EDGE_MARGIN_RATIO)
        upper_y_max = h * self.PAIR_UPPER_HALF_RATIO

        candidates: list[tuple[float, float]] = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.EYE_AREA_MIN:
                continue
            if area > total_area * self.EYE_AREA_RATIO_MAX:
                continue
            cx_eye, cy_eye = centroids[i]
            # 端/下部は除外 (X 印の角が候補にならないように)
            if cx_eye < edge_x_min or cx_eye > edge_x_max:
                continue
            if cy_eye > upper_y_max:
                continue
            candidates.append((cx_eye, cy_eye))

        if len(candidates) < 2:
            return False

        y_tol = h * self.PAIR_Y_TOLERANCE_RATIO
        x_min = w * self.PAIR_X_DISTANCE_MIN_RATIO
        x_max = w * self.PAIR_X_DISTANCE_MAX_RATIO
        sym_tol = w * self.PAIR_SYMMETRY_TOLERANCE
        center_x = w / 2

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                x1, y1 = candidates[i]
                x2, y2 = candidates[j]
                if abs(y1 - y2) > y_tol:
                    continue
                dx = abs(x1 - x2)
                if dx < x_min or dx > x_max:
                    continue
                # 左右対称性: 各点の中心からの距離が近い
                left_dist = abs(min(x1, x2) - center_x)
                right_dist = abs(max(x1, x2) - center_x)
                if abs(left_dist - right_dist) > sym_tol:
                    continue
                return True
        return False


class GatedCnnClassifier(PatchClassifier):
    """
    Puyo 存在ゲート + CNN 色分類 + HSV ハイブリッド補正を組み合わせた分類器。

    動作:
        1. ゲート: puyo 不在 → COLOR_EMPTY
        2. CNN で色分類
        3. HSV 強色 (純度の高い色) が見えている場合は CNN 判定を上書き
           - 特に赤は CNN が苦手なので HSV が優先

    致命列 (col 2) の最上段 (row 1) は死亡警告 "X" が表示され、
    回転アニメーションの方向によって通常ゲートを通過してしまうため、
    2 眼必須の厳格ゲートを追加適用する。
    """

    # 厳格判定を適用する (row, col) セット (可視行インデックス)
    STRICT_CELLS: frozenset[tuple[int, int]] = frozenset({(0, 2)})

    # HSV で色を強制判定するためのルール (h_min, h_max, s_min, v_min, color_code)
    # CNN が苦手な色をオーバーライド
    HSV_STRONG_COLOR_RULES: tuple[tuple[int, int, int, int, int], ...] = (
        (0, 10, 140, 110, COLOR_RED),
        (165, 180, 140, 110, COLOR_RED),
        # 紫: 青・赤との混同対策 (H=125-160 の高彩度・高明度域)
        (125, 160, 100, 160, COLOR_PURPLE),
    )
    HSV_STRONG_RATIO: float = 0.45

    def __init__(
        self,
        color_classifier: PatchClassifier,
        gate: PuyoPresenceGate | None = None,
        strict_gate: PuyoPresenceGate | None = None,
        enable_hsv_override: bool = True,
        ui_matcher: "UiMaskMatcher | None" = None,
    ) -> None:
        self._color = color_classifier
        self._gate = gate or PuyoPresenceGate()
        if strict_gate is None:
            strict_gate = PuyoPresenceGate()
            strict_gate.strict_pair_eyes = True
        self._strict_gate = strict_gate
        self._hsv_override = enable_hsv_override
        # UI オーバーレイ（×マーク等）マッチャー。None の場合はデフォルトを遅延ロード。
        # テンプレート未配置なら空マッチャーで常に is_ui=False。
        if ui_matcher is None:
            from src.ui_mask import UiMaskMatcher as _UM
            ui_matcher = _UM.load_default()
        self._ui_matcher = ui_matcher

    def classify(self, bgr_patch: np.ndarray) -> int:
        """通常ゲート経由の色分類 (位置情報なし)。"""
        if not self._gate.is_puyo(bgr_patch):
            return COLOR_EMPTY
        cnn_color = self._color.classify(bgr_patch)
        # UI オーバーレイ判定: ×マーク等は puyo ではないので empty に差し替え
        if self._ui_matcher.is_ui(bgr_patch):
            return COLOR_EMPTY
        if self._hsv_override:
            override = self._hsv_strong_color(bgr_patch)
            if override is not None and override != cnn_color:
                return override
        return cnn_color

    def classify_at(
        self, bgr_patch: np.ndarray, visible_row: int, col: int,
    ) -> int:
        """位置情報付き色分類。特定セルでは厳格ゲート使用。"""
        gate = (
            self._strict_gate if (visible_row, col) in self.STRICT_CELLS
            else self._gate
        )
        if not gate.is_puyo(bgr_patch):
            return COLOR_EMPTY
        cnn_color = self._color.classify(bgr_patch)
        if self._ui_matcher.is_ui(bgr_patch):
            return COLOR_EMPTY
        if self._hsv_override:
            override = self._hsv_strong_color(bgr_patch)
            if override is not None and override != cnn_color:
                return override
        return cnn_color

    @classmethod
    def _hsv_strong_color(cls, bgr_patch: np.ndarray) -> int | None:
        """
        パッチ中央に強い特定色がある場合、その色コードを返す。

        CNN が色を間違えやすい (特に赤→青) 場合の補正用。
        中央 70% の領域で、HSV 規則に一致するピクセルが一定比率以上
        あれば、その色を強制判定する。
        """
        h, w = bgr_patch.shape[:2]
        mh, mw = int(h * 0.15), int(w * 0.15)
        center = bgr_patch[mh:h-mh, mw:w-mw]
        if center.size == 0:
            return None
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        for h_min, h_max, s_min, v_min, color in cls.HSV_STRONG_COLOR_RULES:
            mask = (
                (hsv[:, :, 0] >= h_min) & (hsv[:, :, 0] <= h_max)
                & (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)
            )
            ratio = float(mask.sum()) / total
            if ratio >= cls.HSV_STRONG_RATIO:
                return color
        return None


class HsvPatchClassifier(PatchClassifier):
    """既存 ColorClassifier を新インタフェースに適合させるアダプタ。"""

    def __init__(self, classifier: ColorClassifier | None = None) -> None:
        self._inner = classifier or ColorClassifier()

    def classify(self, bgr_patch: np.ndarray) -> int:
        return self._inner.classify(bgr_patch)

    def accuracy(self, samples: "Sequence[PatchSample]") -> float:
        """評価用の精度計算 (MLP/CNN と同等インタフェース)。"""
        if not samples:
            return 0.0
        correct = sum(
            1 for s in samples if self.classify(s.patch) == s.color
        )
        return correct / len(samples)


# ============================
# 学習サンプル生成
# ============================


@dataclass
class PatchSample:
    """学習サンプル 1 件。"""
    patch: np.ndarray        # shape=(H, W, 3) uint8
    color: int               # VALID_COLORS のいずれか


def generate_training_patches(
    per_class: int = 80,
    patch_size: int = 16,
    noise_std: float = DEFAULT_NOISE_STD,
    hue_jitter: int = DEFAULT_HUE_JITTER,
    seed: int = DEFAULT_SEED,
) -> list[PatchSample]:
    """
    fixtures.COLOR_HSV_SAMPLES を起点に、ノイズ/色ジッタ付きの学習用パッチを生成する。

    Args:
        per_class: 各色あたりのサンプル数。
        patch_size: パッチ辺の長さ (px)。
        noise_std: 画素値に加える Gaussian ノイズ (0-255 スケール)。
        hue_jitter: Hue ジッタ幅 (±hue_jitter の一様乱数)。
        seed: 乱数シード。

    Returns:
        list[PatchSample]: 合計 per_class × 7 件のサンプル。
    """
    from tests.fixtures import COLOR_HSV_SAMPLES

    rng = np.random.default_rng(seed)
    samples: list[PatchSample] = []

    for color_code, (h, s, v) in COLOR_HSV_SAMPLES.items():
        for _ in range(per_class):
            patch = _synthesize_patch(
                rng, h, s, v, patch_size, noise_std, hue_jitter, color_code,
            )
            samples.append(PatchSample(patch=patch, color=color_code))
    return samples


def _synthesize_patch(
    rng: np.random.Generator,
    h: int, s: int, v: int,
    size: int,
    noise_std: float,
    hue_jitter: int,
    color_code: int,
) -> np.ndarray:
    """単一パッチを合成する (HSV ベース→BGR 変換+ノイズ)。"""
    if color_code == COLOR_EMPTY:
        # 空セル: 0〜少しノイズの黒
        patch = rng.normal(0, noise_std, (size, size, 3)).clip(0, 255)
        return patch.astype(np.uint8)

    # HSV 基準値に jitter を加えて BGR へ
    jh = int(h + rng.integers(-hue_jitter, hue_jitter + 1)) % 180
    js = int(np.clip(s + rng.integers(-20, 21), 40, 255))
    jv = int(np.clip(v + rng.integers(-20, 21), 50, 255))

    hsv = np.full((size, size, 3), (jh, js, jv), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    noise = rng.normal(0, noise_std, bgr.shape)
    patch = np.clip(bgr.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return patch


# ============================
# MLP 分類器 (numpy)
# ============================


class MlpPatchClassifier(PatchClassifier):
    """
    numpy のみで実装した学習可能 MLP 分類器。

    構成: FEATURE_DIM_PATCH → hidden[0] → hidden[1] → NUM_CLASSES
    活性: ReLU (隠れ層), Softmax (出力)
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = MLP_HIDDEN_SIZES,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._layers: list[dict[str, np.ndarray]] = []
        rng = np.random.default_rng(seed)
        dims = [FEATURE_DIM_PATCH, *hidden_sizes, NUM_CLASSES]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / in_d)   # He 初期化
            self._layers.append({
                "W": rng.normal(0, scale, size=(in_d, out_d)),
                "b": np.zeros(out_d),
            })

    # ============================
    # 推論
    # ============================

    def classify(self, bgr_patch: np.ndarray) -> int:
        feat = patch_to_feature(bgr_patch)
        logits = self._forward(feat.reshape(1, -1))[-1]
        pred_idx = int(np.argmax(logits[0]))
        return CLASS_INDEX_TO_COLOR[pred_idx]

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        """各クラスの確率分布を返す (shape=(NUM_CLASSES,))。"""
        feat = patch_to_feature(bgr_patch)
        logits = self._forward(feat.reshape(1, -1))[-1]
        return _softmax(logits)[0]

    # ============================
    # 学習
    # ============================

    def fit(
        self,
        samples: Sequence[PatchSample],
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH,
    ) -> list[float]:
        """
        サンプルで学習する。

        Returns:
            list[float]: エポック毎の平均クロスエントロピー損失。
        """
        if not samples:
            raise ValueError("学習サンプルが空です")
        X, y = self._prepare_batch(samples)
        n = X.shape[0]
        losses: list[float] = []
        rng = np.random.default_rng(0)

        for _ in range(epochs):
            indices = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                total += self._step(X[idx], y[idx], lr) * (end - start)
            losses.append(total / n)
        return losses

    def accuracy(self, samples: Sequence[PatchSample]) -> float:
        """サンプル集合に対する分類精度を返す。"""
        if not samples:
            return 0.0
        correct = sum(
            1 for s in samples if self.classify(s.patch) == s.color
        )
        return correct / len(samples)

    # ============================
    # 永続化
    # ============================

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self._layers):
            flat[f"W{i}"] = layer["W"]
            flat[f"b{i}"] = layer["b"]
        np.savez(path, n_layers=np.array([len(self._layers)]), **flat)

    @classmethod
    def load(cls, path: Path) -> "MlpPatchClassifier":
        path = Path(path)
        npz = path if path.suffix else path.with_suffix(".npz")
        if not npz.exists():
            npz = path.with_suffix(".npz")
        data = np.load(npz)
        n = int(data["n_layers"][0])
        # ダミー hidden_sizes で初期化、その後上書き
        model = cls.__new__(cls)
        model._layers = []
        for i in range(n):
            model._layers.append({"W": data[f"W{i}"], "b": data[f"b{i}"]})
        return model

    # ============================
    # 内部メソッド
    # ============================

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        """前向き計算。各層の出力 (活性化後) リストを返す。"""
        acts = [X]
        for i, layer in enumerate(self._layers):
            z = acts[-1] @ layer["W"] + layer["b"]
            if i < len(self._layers) - 1:
                z = np.maximum(0.0, z)  # ReLU
            acts.append(z)
        return acts

    def _step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """1 ミニバッチ分の勾配ステップ。平均損失を返す。"""
        acts = self._forward(X)
        probs = _softmax(acts[-1])
        n = X.shape[0]

        # クロスエントロピー損失
        log_probs = np.log(probs[np.arange(n), y] + 1e-12)
        loss = float(-np.mean(log_probs))

        # 出力層の勾配
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        # 逆伝播
        grad_out = dlogits
        for i in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[i]
            a_prev = acts[i]
            grad_W = a_prev.T @ grad_out
            grad_b = grad_out.sum(axis=0)
            if i > 0:
                grad_prev = grad_out @ layer["W"].T
                # ReLU 微分
                grad_prev *= (acts[i] > 0)
                grad_out = grad_prev
            layer["W"] -= lr * grad_W
            layer["b"] -= lr * grad_b
        return loss

    @staticmethod
    def _prepare_batch(
        samples: Sequence[PatchSample],
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([patch_to_feature(s.patch) for s in samples])
        y = np.array(
            [COLOR_TO_CLASS_INDEX[s.color] for s in samples],
            dtype=np.int64,
        )
        return X, y


def _softmax(x: np.ndarray) -> np.ndarray:
    """数値安定な softmax (axis=-1)。"""
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x_shift)
    return e / np.sum(e, axis=-1, keepdims=True)


# ============================
# CNN 分類器 (torch、利用可能時のみ)
# ============================


def _torch_available() -> bool:
    """
    torch が import 可能かを検査する。

    部分インストール状態 (wheel 展開途中) では OSError が発生し得るため、
    ImportError だけでなく例外全般を false として扱う。
    """
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class CnnPatchClassifier(PatchClassifier):
    """
    torch ベースの小型 CNN 分類器。

    torch がインストールされていない場合、コンストラクタで
    ImportError を送出する。

    cycle 71v (案 D、 2026-05-13): 既存 25KB model.
    CnnPatchClassifierLarge (= 4 層 conv + BatchNorm、 ~100KB) も提供.
    """

    # 入力チャンネル数: BGR(3) + HSV(3) = 6
    INPUT_CHANNELS: int = 6

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        if not _torch_available():
            raise ImportError(
                "torch が未インストールです。"
                "`pip install torch` で導入してください。"
            )
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self._torch = torch
        # BGR+HSV 6チャンネル入力で色相情報を直接利用
        self._model = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )
        self._model.eval()
        self._device: str = "cpu"

    def to_device(self, device: str = "cuda") -> "CnnPatchClassifier":
        """Z-3C: 推論を指定 device で実行する。

        Args:
            device: "cuda" or "cpu"。CUDA が利用不可なら "cpu" にフォールバック。

        Returns:
            self (chainable)
        """
        torch = self._torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = device
        self._model = self._model.to(device)
        return self

    # ============================
    # 推論
    # ============================

    def classify(self, bgr_patch: np.ndarray) -> int:
        tensor = self._patch_to_tensor(bgr_patch).to(self._device)
        with self._torch.no_grad():
            logits = self._model(tensor)
        idx = int(self._torch.argmax(logits, dim=1).item())
        return CLASS_INDEX_TO_COLOR[idx]

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        """各クラスの確率を返す (NUM_CLASSES,)。TTA / アンサンブル用。"""
        tensor = self._patch_to_tensor(bgr_patch).to(self._device)
        with self._torch.no_grad():
            logits = self._model(tensor)
            probs = self._torch.nn.functional.softmax(logits, dim=1)
        return probs[0].cpu().numpy()

    def predict_proba_batch(
        self, bgr_patches: list[np.ndarray],
    ) -> np.ndarray:
        """複数 patch をまとめて推論 (Z-3C: GPU 高速化)。

        Args:
            bgr_patches: BGR patch のリスト。空 patch は zero tensor で詰める。

        Returns:
            shape=(N, NUM_CLASSES) の確率配列。
        """
        if not bgr_patches:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        torch = self._torch
        tensors = [self._patch_to_tensor(p)[0] for p in bgr_patches]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def classify_batch(self, bgr_patches: list[np.ndarray]) -> list[int]:
        """複数 patch をまとめて色 code に分類 (Z-3C: GPU 高速化)。"""
        if not bgr_patches:
            return []
        probs = self.predict_proba_batch(bgr_patches)
        idxs = np.argmax(probs, axis=1)
        return [CLASS_INDEX_TO_COLOR[int(i)] for i in idxs]

    def predict_proba_grid(
        self, frame: np.ndarray, region: BoardRegion,
    ) -> np.ndarray:
        """各 cell の max softmax を 12x6 grid で返す (Phase I.c-1)。

        OnlineHsvCalibrator が信頼サンプル抽出に使用 (CNN 確信度フィルタ)。

        Args:
            frame: BGR 1080p frame.
            region: 1P or 2P BoardRegion.

        Returns:
            shape=(VISIBLE_ROWS, BOARD_COLS) の max softmax 配列。float32.
        """
        h, w = frame.shape[:2]
        patches: list[np.ndarray] = []
        for vrow in range(VISIBLE_ROWS):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(int(x1), w - 1))
                x2 = max(x1 + 1, min(int(x2), w))
                y1 = max(0, min(int(y1), h - 1))
                y2 = max(y1 + 1, min(int(y2), h))
                patches.append(frame[y1:y2, x1:x2])
        probs = self.predict_proba_batch(patches)
        grid = np.max(probs, axis=1).reshape(VISIBLE_ROWS, BOARD_COLS)
        return grid.astype(np.float32)

    # ============================
    # 学習
    # ============================

    def fit(
        self,
        samples: Sequence[PatchSample],
        epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 32,
        class_weighted: bool = True,
    ) -> list[float]:
        """サンプルで学習する (CPU)。class_weighted=True で逆頻度重み付き損失を使用。"""
        if not samples:
            raise ValueError("学習サンプルが空です")
        torch = self._torch
        import torch.nn as nn
        import torch.optim as optim

        X = torch.stack([self._patch_to_tensor(s.patch)[0] for s in samples])
        y = torch.tensor(
            [COLOR_TO_CLASS_INDEX[s.color] for s in samples],
            dtype=torch.long,
        )

        self._model.train()
        optimizer = optim.Adam(self._model.parameters(), lr=lr)

        # クラス逆頻度重みで少数クラス (赤等) の損失を増幅
        weight = None
        if class_weighted:
            counts = torch.bincount(y, minlength=NUM_CLASSES).float()
            counts = counts.clamp(min=1.0)
            weight = (1.0 / counts)
            weight = weight / weight.sum() * NUM_CLASSES  # 平均1.0に正規化
        criterion = nn.CrossEntropyLoss(weight=weight)

        losses: list[float] = []
        n = X.size(0)
        for _ in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                optimizer.zero_grad()
                logits = self._model(X[idx])
                loss = criterion(logits, y[idx])
                loss.backward()
                optimizer.step()
                total += loss.item() * (end - start)
            losses.append(total / n)
        self._model.eval()
        return losses

    def accuracy(self, samples: Sequence[PatchSample]) -> float:
        correct = sum(
            1 for s in samples if self.classify(s.patch) == s.color
        )
        return correct / len(samples) if samples else 0.0

    # ============================
    # 内部ユーティリティ
    # ============================

    def _patch_to_tensor(self, bgr_patch: np.ndarray):
        """BGR+HSV 6 ch 正規化テンソルに変換する (shape=(1, 6, H, W))。

        cycle 71v (2026-05-14): 元は CnnPatchClassifierLarge のみに置かれており、
        base class での classify/predict/fit が AttributeError になっていた。
        base に移動して両クラスから共有 (= Large は継承で利用)。
        cycle 32g (2026-05-19): USE_CIRCLE_MASK=True 時に中央円形マスク
        (= 四隅 0 塗り) を適用、 背景情報を CNN 入力から削減。
        """
        torch = self._torch
        if bgr_patch.size == 0:
            return torch.zeros(1, self.INPUT_CHANNELS, PATCH_RESIZE_H, PATCH_RESIZE_W)
        resized = cv2.resize(
            bgr_patch, (PATCH_RESIZE_W, PATCH_RESIZE_H),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        combined = np.concatenate([resized, hsv], axis=2)  # (H, W, 6)
        normalized = combined.astype(np.float32) / 255.0
        # cycle 32g: 円形マスク適用 (= 中央 puyo 領域のみ通し、 四隅背景を 0 塗り)
        if USE_CIRCLE_MASK:
            mask = _get_circle_mask(PATCH_RESIZE_H, PATCH_RESIZE_W)
            normalized = normalized * mask[..., None]  # (H, W, 6)
        return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

    # ============================
    # 永続化
    # ============================

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(self._model.state_dict(), str(path))

    @classmethod
    def load(cls, path: Path) -> "CnnPatchClassifier":
        classifier = cls()
        state = classifier._torch.load(str(path), weights_only=True, map_location="cpu")
        classifier._model.load_state_dict(state)
        classifier._model.eval()
        return classifier


class CnnPatchClassifierLarge(CnnPatchClassifier):
    """cycle 71v (案 D, 2026-05-13): 中規模 CNN.

    既存 CnnPatchClassifier (= 25KB, 2 層 conv) を拡張:
    - 4 層 conv (= 32 → 32 → 64 → 64 channels)
    - BatchNorm + Dropout 追加
    - 100KB+ 程度のモデル
    - holdout 99% → 99.5%+ 期待

    インスタンス化方法は CnnPatchClassifier と同じ. 構造のみ拡張で他 method は継承.
    """

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        super().__init__(seed=seed)
        import torch.nn as nn  # noqa: F401 (= 親で torch import 済)
        # 構造を拡張版で上書き
        self._model = nn.Sequential(
            # block 1: 32 ch
            nn.Conv2d(self.INPUT_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # block 2: 64 ch
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 集約
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )
        self._model.eval()

    # `_patch_to_tensor` は base class CnnPatchClassifier に集約済 (cycle 71v 修正).
    # Large は継承でそのまま利用するため、 ここでの再定義は不要.


class OjamaShapeGate:
    """ojama (= 灰色 + ヒビ模様 + 円形) を視覚特徴で判定する gate.

    cycle 50 (2026-05-21): cycle 32 ojama 構造的除外の撤回に伴い新設。
    score OCR 差分は cell-level ojama 推論不可能 (= 端数ランダム降下) のため
    CNN 学習復活が必須。 採取条件は seed 段階で本 gate を通過したもののみ。

    判定基準 (= AND 結合):
        - 中央領域の HSV S 低い (= 灰色) かつ V 中央域 (= 白文字でない)
        - Canny edge ratio が 0.10 以上 (= ヒビ模様)
        - max contour 円形度 (= 面積 / 外接円面積) >= 0.50
    """

    GRAY_S_MAX: int = 50
    GRAY_V_MIN: int = 70
    GRAY_V_MAX: int = 180  # 白文字 (WIN/LOSE) を除外
    EDGE_DENSITY_MIN: float = 0.10
    CIRCULARITY_MIN: float = 0.50
    CENTER_MARGIN_RATIO: float = 0.20

    @classmethod
    def relaxed(cls) -> "OjamaShapeGate":
        """cycle 58 (= 2026-05-23 案 A): 閾値緩和版を返す.

        cycle 56_v4 で 11/27 動画 ojama 採取 0 件だった元の閾値を緩和:
        - GRAY_S_MAX 50 → 80 (= 彩度許容)
        - GRAY_V_MAX 180 → 210 (= 明るめ許容)
        - EDGE_DENSITY_MIN 0.10 → 0.05 (= ヒビ少めでも許容)
        - CIRCULARITY_MIN 0.50 → 0.30 (= 円形度許容)

        トレードオフ: 文字エフェクト混入リスク増。 ユーザー目視で seed
        review が必要 (= cycle 32 系 7 連敗の教訓)。
        """
        gate = cls()
        gate.GRAY_S_MAX = 80
        gate.GRAY_V_MAX = 210
        gate.EDGE_DENSITY_MIN = 0.05
        gate.CIRCULARITY_MIN = 0.30
        return gate

    def is_ojama(self, bgr_patch: np.ndarray) -> bool:
        """patch が ojama 候補なら True (= 灰色 + ヒビ + 円形)."""
        if bgr_patch.size == 0:
            return False
        h, w = bgr_patch.shape[:2]
        cm_h = int(h * self.CENTER_MARGIN_RATIO)
        cm_w = int(w * self.CENTER_MARGIN_RATIO)
        center = bgr_patch[cm_h:h-cm_h, cm_w:w-cm_w]
        if center.size == 0:
            return False
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        s_med = float(np.median(hsv[:, :, 1]))
        v_med = float(np.median(hsv[:, :, 2]))
        if s_med >= self.GRAY_S_MAX:
            return False
        if not (self.GRAY_V_MIN <= v_med <= self.GRAY_V_MAX):
            return False
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.mean(edges > 0))
        if edge_ratio < self.EDGE_DENSITY_MIN:
            return False
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        circ_area = 3.14159 * float(radius) * float(radius)
        if circ_area <= 0:
            return False
        circularity = area / circ_area
        return circularity >= self.CIRCULARITY_MIN
