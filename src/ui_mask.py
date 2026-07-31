"""
UI オーバーレイ（攻撃予告×マーク等）を puyo 誤検出から除外するモジュール。

ぷよぷよeスポーツの対戦画面には、盤面上に「×マーク」（相手への攻撃予告）など
UI オーバーレイが描かれる。これらは赤成分が強いため、CNN が赤ぷよと誤認しやすい。

本モジュールは:
    - 事前に収集したテンプレート画像との正規化相互相関 (NCC) で検出
    - 閾値を超えたら puyo ではなく UI と判定 → empty に差し替え

使い方:
    matcher = UiMaskMatcher.load_default()
    is_ui = matcher.is_ui(patch_bgr)

テンプレート画像:
    models/ui_templates/x_mark.png   (r01 c2 の×マーク、拡大版)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# デフォルトテンプレートディレクトリ
DEFAULT_TEMPLATE_DIR: Path = Path("models/ui_templates")

# テンプレートサイズに揃えるための正規化解像度
NORM_SIZE: tuple[int, int] = (64, 64)

# NCC 判定閾値（0-1）: この値以上でマッチ
DEFAULT_NCC_THRESHOLD: float = 0.75

# 案A (2026-07-30): load_default が読み込む既定テンプレート glob。
# 実測 (scripts/_diag_ui_mask_fire_positions_2026-07-30.py、3動画x400フレーム、
# 約44,000回照合) で telop_challenger* / match_end_* の計4枚は一度も
# 閾値0.75に到達せず (発火ゼロ)。models/ui_templates 直下の x_mark*.png 6枚
# のみに絞ることで matchTemplate 呼出数を 10 テンプレ→6 テンプレに削減する
# (memory project_recognition_profile_matchtemplate_2026-07-30)。
# 全 png を読みたい場合は load_default(glob_pattern="*.png") を明示指定する。
X_MARK_GLOB_PATTERN: str = "x_mark*.png"

# UI_MASK_TARGET_CELLS: ×印 UI オーバーレイの実測発火セル位置。
# 座標系は「raw row」= board.py の行全体 (0=隠し段, 1=画面内最上段、
# HIDDEN_ROWS 込み)。patch_classifier.GatedCnnClassifier.STRICT_CELLS が
# 使う visible_row (= raw row - HIDDEN_ROWS) とは異なるので流用時に注意。
# 実測根拠: scripts/_diag_ui_mask_fire_positions_2026-07-30.py で 112 発火
# 全件が raw row=1, col=2 に一致し、隣接セルへの滲みはゼロ (data/verify/
# ui_mask_fire_2026-07-30/summary.json)。user ドメイン知識「左から3列目の
# 12段目 (=画面内最上段)」とも一致。
UI_MASK_TARGET_CELLS: frozenset[tuple[int, int]] = frozenset({(1, 2)})


@dataclass(frozen=True)
class UiMatchResult:
    """UI マッチ結果。"""
    is_ui: bool
    template_name: str | None
    score: float


class UiMaskMatcher:
    """
    テンプレートマッチングで UI オーバーレイを検出する。

    Usage:
        matcher = UiMaskMatcher.load_default()
        result = matcher.match(bgr_patch)
        if result.is_ui:
            # empty 扱い
    """

    def __init__(
        self,
        templates: dict[str, np.ndarray],
        threshold: float = DEFAULT_NCC_THRESHOLD,
    ) -> None:
        self._templates = templates
        self._threshold = threshold

    @classmethod
    def load_default(
        cls,
        template_dir: Path = DEFAULT_TEMPLATE_DIR,
        threshold: float = DEFAULT_NCC_THRESHOLD,
        glob_pattern: str = X_MARK_GLOB_PATTERN,
    ) -> "UiMaskMatcher":
        """
        既定ディレクトリから glob_pattern に一致する png を読み込む。

        案A (2026-07-30): 既定を "x_mark*.png" に変更 (旧既定は "*.png" 全読み)。
        telop_challenger* / match_end_* の4枚は実測でゼロ発火のため除外し、
        matchTemplate 呼出数を削減する (X_MARK_GLOB_PATTERN の docstring 参照)。
        全 png を読みたい場合は glob_pattern="*.png" を明示指定すること。
        存在しないディレクトリは空マッチャー（常に is_ui=False）。
        """
        templates: dict[str, np.ndarray] = {}
        if template_dir.exists():
            for p in sorted(template_dir.glob(glob_pattern)):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                # グレースケール + 正規化
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, NORM_SIZE, interpolation=cv2.INTER_AREA)
                templates[p.stem] = resized
        return cls(templates=templates, threshold=threshold)

    def match(self, bgr_patch: np.ndarray) -> UiMatchResult:
        """パッチに対して全テンプレートを試し、最大 NCC を返す。"""
        if not self._templates or bgr_patch.size == 0:
            return UiMatchResult(is_ui=False, template_name=None, score=0.0)
        gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, NORM_SIZE, interpolation=cv2.INTER_AREA)

        best_name: str | None = None
        best_score: float = -1.0
        for name, tmpl in self._templates.items():
            # 単純な NCC（同サイズなので matchTemplate で 1 要素結果）
            result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])
            if score > best_score:
                best_score = score
                best_name = name

        return UiMatchResult(
            is_ui=best_score >= self._threshold,
            template_name=best_name,
            score=best_score,
        )

    def is_ui(self, bgr_patch: np.ndarray) -> bool:
        """短縮メソッド: マッチすれば True。"""
        return self.match(bgr_patch).is_ui
