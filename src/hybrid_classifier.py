"""HSV + CNN ハイブリッド分類器 (Phase U-A)。

HSV を主、CNN を補助にして両者を組み合わせる:
    - HSV と CNN が一致 → そのまま採用 (高確信)
    - 不一致で CNN 確信度高 → CNN を採用
    - 不一致で CNN 確信度低 → HSV を採用 (HSV の決定論的判定を優先)

利用例:
    cnn = CnnPatchClassifier()
    cnn._model.load_state_dict(torch.load('models/cnn_phase_u_v1.pt'))
    classifier = HybridClassifier(cnn_classifier=cnn)
    reader = ImageReader(classifier=classifier)
"""
from __future__ import annotations

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.image_reader import (
    BoardRegion,
    ColorClassifier,
)
from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR,
    COLOR_TO_CLASS_INDEX,
    CnnPatchClassifier,
    PuyoPresenceGate,
)

# CNN 確信度がこの値超で HSV と異なる場合 CNN 採用.
# 2026-05-11 サイクル71: 0.75 → 0.70 に引き下げ (= CNN メイン化方針).
# 単色ピクセル多数決 (vote) が「目」 「縁」 「ハイライト」 で揺らぐ問題への
# 対策として、 画像全体パターンを学習した CNN を主軸化する.
# 過去 Phase B-7 の慎重値 (= 0.90) は state machine への CNN ぶれ伝搬を
# 抑える目的だったが、 cnn_global_best.pt の精度向上で safe 域.
# cycle 9 (2026-05-15, Innovation I): 0.70 → 0.55 を試行したが
#   constraint_replaced 全体 +41% (v89m3 で +185 件) と悪化、 cycle 10 で revert。
# cycle 10 (2026-05-15, revert): 0.55 → 0.70 (= cycle 5 baseline 復元)。
DEFAULT_CNN_OVERRIDE_PROB: float = 0.70


class HybridClassifier:
    """HSV ColorClassifier + CnnPatchClassifier + UiMaskMatcher を組み合わせる分類器。

    classify(bgr_patch) -> color_code を提供。
    predict_proba は CNN のものをそのまま提供 (融合判定で利用可能)。
    """

    # cycle 71s 案 C: CNN 確信度がこの値未満かつ HSV と不一致なら UNKNOWN マーク.
    # cycle 71u (2026-05-13 副作用対策): 案 C を事実上無効化 (= 0.0).
    # 案 1a (UNKNOWN 補完) と組み合わさり「過去誤色が長期維持」 する複合副作用
    # により認識精度悪化. v15 状態に戻すため閾値を 0 に.
    LOW_CONFIDENCE_UNKNOWN_THRESHOLD: float = 0.0

    def __init__(
        self,
        hsv_classifier: ColorClassifier | None = None,
        cnn_classifier: CnnPatchClassifier | None = None,
        cnn_override_prob: float = DEFAULT_CNN_OVERRIDE_PROB,
        use_ui_mask: bool = True,
        mask_ojama_logit: bool = False,
        use_puyo_gate: bool = False,
        ui_mask_cells: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        """
        Args:
            ui_mask_cells: 案B (2026-07-30)。指定すると classify_batch の
                UI マスク判定 (is_ui 呼出) を、この集合に含まれる raw row/col
                セルだけに限定する (それ以外は判定省略 = 常に is_ui=False 扱い)。
                座標系は src.ui_mask.UI_MASK_TARGET_CELLS と同じ raw row
                (HIDDEN_ROWS 込み)。
                None (既定) では従来通り全セルで is_ui を判定する
                (backwards compat、bit-identical)。
                実際に絞り込みが効くのは classify_batch に cell_positions も
                渡した場合のみ (両方揃わないと従来動作にフォールバックする)。
        """
        self._hsv = hsv_classifier or ColorClassifier()
        self._cnn = cnn_classifier
        self._cnn_override_prob = float(cnn_override_prob)
        # 2026-05-11 サイクル63: 解像度依存で CNN trust を下げる
        # (低解像度で CNN mode collapse → HSV を信頼).
        self._cnn_override_prob_default = float(cnn_override_prob)
        if use_ui_mask:
            from src.ui_mask import UiMaskMatcher
            self._ui_matcher = UiMaskMatcher.load_default()
        else:
            self._ui_matcher = None
        # 案B (2026-07-30): UI マスク判定対象セルの限定 (既定 None = 全セル)。
        self._ui_mask_cells: frozenset[tuple[int, int]] | None = ui_mask_cells
        # cycle 32e (2026-05-19): ojama を CNN 学習対象外にしているため、
        # 推論時に ojama logit を mask する (= argmax 候補から除外)。
        # default=False で backwards compat 維持。 cycle 32e model 利用時に
        # True を渡すこと。
        self._mask_ojama_logit: bool = bool(mask_ojama_logit)
        # cycle 32e (2026-05-19): PuyoPresenceGate (= 視覚特徴で「puyo らしさ」
        # 判定) を CNN 推論前に挟む。 gate=False の patch は HSV-only 経路に
        # fallback (= EMPTY 強制ではなく安全側、 cycle 30 副作用回避)。
        # default=False で backwards compat 維持。
        self._use_puyo_gate: bool = bool(use_puyo_gate)
        self._puyo_gate: PuyoPresenceGate | None = (
            PuyoPresenceGate() if use_puyo_gate else None
        )

    def classify(self, bgr_patch: np.ndarray) -> int:
        # UI オーバーレイ (X 印など) → EMPTY 強制
        if (
            self._ui_matcher is not None
            and bgr_patch.size > 0
            and self._ui_matcher.is_ui(bgr_patch)
        ):
            return COLOR_EMPTY
        # cycle 32e: PuyoPresenceGate で「puyo らしさ」 を事前 check。
        # gate=False (= puyo の眼/陰影/飽和色が見つからない) なら背景候補 →
        # HSV-only 経路に倒す (= CNN の background→puyo 誤分類を避ける)。
        # EMPTY 強制ではなく HSV fallback で 落下中 puyo 等の救済も維持。
        gate_passed = True
        if self._puyo_gate is not None and bgr_patch.size > 0:
            gate_passed = self._puyo_gate.is_puyo(bgr_patch)
            if not gate_passed:
                return self._hsv.classify(bgr_patch)
        # CNN 主、HSV 補助のロジック (CNN holdout 98.6% で高精度)
        if self._cnn is None:
            return self._hsv.classify(bgr_patch)
        try:
            probs = self._cnn.predict_proba(bgr_patch)
            # cycle 32e: ojama logit を mask (= argmax 候補から除外)
            if self._mask_ojama_logit:
                ojama_idx = COLOR_TO_CLASS_INDEX.get(COLOR_OJAMA)
                if ojama_idx is not None:
                    probs = probs.copy()
                    probs[ojama_idx] = -1e9
            best_idx = int(np.argmax(probs))
            cnn_color = CLASS_INDEX_TO_COLOR[best_idx]
            cnn_prob = float(probs[best_idx])
        except Exception:
            return self._hsv.classify(bgr_patch)
        # 高確信度なら CNN を採用
        if cnn_prob >= self._cnn_override_prob:
            return cnn_color
        # 低確信度: HSV と一致なら CNN、不一致なら HSV or UNKNOWN
        hsv_color = self._hsv.classify(bgr_patch)
        if cnn_color == hsv_color:
            return cnn_color
        # cycle 71s 案 C (2026-05-13): CNN 確信度が非常に低い場合 (= < 0.5) で
        # HSV と不一致なら、 確定せず UNKNOWN マーク. 後段 vote refinement や
        # 案 1a/B で次フレームの観測で確定する.
        if cnn_prob < self.LOW_CONFIDENCE_UNKNOWN_THRESHOLD:
            from src.board import COLOR_UNKNOWN
            return COLOR_UNKNOWN
        return hsv_color

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        """CNN の確率分布を提供 (融合判定で利用)。"""
        if self._cnn is None:
            # HSV のみの場合、選択色を 1.0 として返す
            n = len(CLASS_INDEX_TO_COLOR)
            probs = np.zeros(n, dtype=np.float32)
            color = self._hsv.classify(bgr_patch)
            for i, c in enumerate(CLASS_INDEX_TO_COLOR):
                if c == color:
                    probs[i] = 1.0
                    break
            return probs
        return self._cnn.predict_proba(bgr_patch)

    def predict_proba_batch(
        self, bgr_patches: list[np.ndarray],
    ) -> np.ndarray:
        """Z-3C: 複数 patch をまとめて確率推論。CNN がバッチ対応していれば
        高速 GPU 推論を使う。HSV-only fallback は逐次。
        """
        if not bgr_patches:
            n = len(CLASS_INDEX_TO_COLOR)
            return np.zeros((0, n), dtype=np.float32)
        if self._cnn is not None and hasattr(self._cnn, "predict_proba_batch"):
            return self._cnn.predict_proba_batch(bgr_patches)
        # fallback: 個別 predict_proba を回す
        return np.stack([self.predict_proba(p) for p in bgr_patches])

    def classify_batch(
        self, bgr_patches: list[np.ndarray],
        bg_distances: list[float | None] | None = None,
        cell_positions: list[tuple[int, int]] | None = None,
    ) -> list[int]:
        """Z-3C: 複数 patch をまとめて色 code に分類。

        UI mask + CNN batch + HSV 併合の流れを 1 度で処理。
        ImageReader の cell ループ高速化用。

        cycle 34 (2026-05-20): bg_distances optional 引数追加。
        各 patch の bg_fp 距離が指定された場合、 distance が小さい cell の
        CNN 出力に EMPTY logit soft prior を加算 (= 背景誤認の補正)。
        backwards compat: bg_distances=None で旧挙動と完全同一。

        案B (2026-07-30): cell_positions optional 引数追加。
        bgr_patches と同じ順序・同じ長さの (raw_row, col) リストを渡すと、
        __init__ の ui_mask_cells に含まれないセルの is_ui 呼出を省略できる
        (matchTemplate 呼出削減)。cell_positions=None、または ui_mask_cells
        未指定、または長さ不一致の場合は従来通り全セルで判定する
        (backwards compat、bit-identical)。
        """
        if not bgr_patches:
            return []
        n = len(bgr_patches)
        # cycle 37 (2026-05-20): soft prior 撤回 (BOOST_MAX=0)。
        # cycle 34-36 全 boost 強度で bg_dominant 悪化判明 → 撤回。
        # tier 1 < 25.0 (image_reader.py) のみで bg 対策。
        BG_DIST_BOOST_MAX: float = 0.0
        BG_DIST_BOOST_SCALE: float = 50.0
        BG_DIST_MAX_RELEVANT: float = 150.0
        # UI mask の事前判定 (バッチ化不可、cell ごと)
        # 案B: ui_mask_cells + cell_positions が両方揃った時のみセル限定を有効化。
        # 長さ不一致は呼出元の不整合 (バグ) の疑いがあるため安全側 (全セル判定) に倒す。
        restrict_cells = (
            self._ui_mask_cells is not None
            and cell_positions is not None
            and len(cell_positions) == n
        )
        ui_mask = [False] * n
        if self._ui_matcher is not None:
            for i, p in enumerate(bgr_patches):
                if restrict_cells and cell_positions[i] not in self._ui_mask_cells:
                    continue  # UI 描画され得ない位置 → is_ui 呼出省略 (常に False 扱い)
                if p.size > 0 and self._ui_matcher.is_ui(p):
                    ui_mask[i] = True
        if self._cnn is None:
            # HSV のみ
            out: list[int] = []
            for i, p in enumerate(bgr_patches):
                if ui_mask[i]:
                    out.append(COLOR_EMPTY)
                else:
                    out.append(self._hsv.classify(p))
            return out
        # CNN バッチ推論
        probs_batch = self._cnn.predict_proba_batch(bgr_patches)
        empty_idx = COLOR_TO_CLASS_INDEX.get(COLOR_EMPTY, 0)
        out2: list[int] = []
        for i, (patch, probs) in enumerate(zip(bgr_patches, probs_batch)):
            if ui_mask[i]:
                out2.append(COLOR_EMPTY)
                continue
            # cycle 34: ojama mask + soft prior 適用のため probs を copy
            probs_eff = probs.copy() if (
                self._mask_ojama_logit
                or (bg_distances is not None and bg_distances[i] is not None)
            ) else probs
            # cycle 32e: ojama logit を mask
            if self._mask_ojama_logit:
                ojama_idx = COLOR_TO_CLASS_INDEX.get(COLOR_OJAMA)
                if ojama_idx is not None:
                    probs_eff[ojama_idx] = -1e9
            # cycle 34: bg_fp 距離 soft prior (= EMPTY logit ブースト)
            if bg_distances is not None and bg_distances[i] is not None:
                d = float(bg_distances[i])
                if d < BG_DIST_MAX_RELEVANT:
                    import math
                    boost = BG_DIST_BOOST_MAX * math.exp(-d / BG_DIST_BOOST_SCALE)
                    probs_eff[empty_idx] = float(probs_eff[empty_idx]) + boost
            best_idx = int(np.argmax(probs_eff))
            cnn_color = CLASS_INDEX_TO_COLOR[best_idx]
            cnn_prob = float(probs_eff[best_idx])
            if cnn_prob >= self._cnn_override_prob:
                # CNN 高確信経路: そのまま採用
                out2.append(cnn_color)
                continue
            # 低確信度: HSV と一致なら CNN、不一致なら HSV or UNKNOWN
            hsv_color = self._hsv.classify(patch)
            if cnn_color == hsv_color:
                out2.append(cnn_color)
            elif cnn_prob < self.LOW_CONFIDENCE_UNKNOWN_THRESHOLD:
                # cycle 71s 案 C: 低確信度 + 不一致 → UNKNOWN マーク (= 後段 vote 確定)
                from src.board import COLOR_UNKNOWN as _COLOR_UNKNOWN
                out2.append(_COLOR_UNKNOWN)
            else:
                out2.append(hsv_color)
        return out2


    def set_cnn_override_prob(self, prob: float) -> None:
        """CNN override 閾値を更新 (低解像度時 HSV 信頼用).

        prob >= 1.0 で CNN は事実上無効化 (= HSV 主軸).
        """
        self._cnn_override_prob = float(prob)

    def predict_proba_and_hsv_grid(
        self, frame: np.ndarray, region: BoardRegion,
    ) -> tuple[np.ndarray, np.ndarray]:
        """OnlineHsvCalibrator 用 grid 取得 (Phase I.c-2).

        Returns:
            (proba_grid, hsv_grid):
              - proba_grid: shape=(VISIBLE_ROWS, BOARD_COLS) float32, CNN max softmax
              - hsv_grid: shape=(VISIBLE_ROWS, BOARD_COLS) int32, HSV-only 判定色
            CNN 不在時は proba_grid 全て 0.0 (= 低確信度として扱われる)
        """
        # CNN proba grid
        if self._cnn is not None:
            proba_grid = self._cnn.predict_proba_grid(frame, region)
        else:
            proba_grid = np.zeros(
                (VISIBLE_ROWS, BOARD_COLS), dtype=np.float32,
            )
        # HSV grid (cell ごとに HSV-only 判定)
        h, w = frame.shape[:2]
        hsv_grid = np.zeros((VISIBLE_ROWS, BOARD_COLS), dtype=np.int32)
        for vrow in range(VISIBLE_ROWS):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(int(x1), w - 1))
                x2 = max(x1 + 1, min(int(x2), w))
                y1 = max(0, min(int(y1), h - 1))
                y2 = max(y1 + 1, min(int(y2), h))
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                hsv_grid[vrow, col] = int(self._hsv.classify(patch))
        return proba_grid, hsv_grid


__all__ = [
    "DEFAULT_CNN_OVERRIDE_PROB",
    "HybridClassifier",
]
