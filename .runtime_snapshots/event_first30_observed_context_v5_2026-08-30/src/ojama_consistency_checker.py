"""ojama 推論のクロスチェック整合性検証モジュール (Phase O)。

3 つの方式 (score OCR 差分 / 視覚版 CNN / ChainResult シミュ) の出力を比較し、
整合性スコアと最終的な信頼ojama個数を返す。

原理:
    1. score OCR 差分 → ojama 個数 (確定的、正解率 100%、readable 87.2%)
    2. 視覚版 CNN → 6 アイコン (CNN 0.92、ただし視覚 only)
    3. score OCR 個数 → アイコン分解 (`ojama_count_to_icons`)
    4. (3) と (2) のアイコン構成を比較 → 一致度から信頼度
    5. 最終的に最も信頼できる方式を選んで ojama 個数返す

利用例:
    checker = OjamaConsistencyChecker()
    result = checker.cross_check(
        score_delta_ojama=240,             # score OCR ベース
        visual_icons=[("rock", 6), ("large", 4)],  # CNN 出力
    )
    print(result.final_ojama, result.confidence, result.method_used)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.scoring import (
    OJAMA_ICON_VALUES,
    OJAMA_MAX_ICONS_DISPLAY,
    icons_to_ojama_count,
    ojama_count_to_icons,
)

# 採用方式
METHOD_SCORE_DELTA: str = "score_delta"
METHOD_VISUAL_CNN: str = "visual_cnn"
METHOD_AGREED: str = "agreed"
METHOD_FALLBACK_SCORE: str = "fallback_score"
METHOD_FALLBACK_VISUAL: str = "fallback_visual"
METHOD_NONE: str = "none"

# アイコン一致判定の許容差: 「rock 6 個」と「rock 5 個 + small 30 個」のような
# わずかな差異は同じとみなす (端数表示落ちを考慮)
ICON_DIFF_TOLERANCE_OJAMA: int = 6  # 6 個 = large 1 個分の差まで許容
# 信頼度しきい値
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class ConsistencyResult:
    """クロスチェック結果。

    Attributes:
        final_ojama: 採用した最終 ojama 個数 (None = 推論不能)
        confidence: 0.0-1.0、整合性ベースの信頼度
        method_used: 採用した方式 (score_delta / visual_cnn / agreed / ...)
        score_delta_ojama: score OCR ベース個数 (None なら未測定)
        visual_ojama: 視覚版アイコンから集計した個数 (None なら未測定)
        agreement: 2 方式間の差 (絶対値)、None=どちらか欠落
        details: 詳細 (デバッグ用)
    """
    final_ojama: int | None
    confidence: float
    method_used: str
    score_delta_ojama: int | None
    visual_ojama: int | None
    agreement: int | None
    details: dict = field(default_factory=dict)


class OjamaConsistencyChecker:
    """3 方式間のクロスチェッカー。"""

    def __init__(
        self,
        diff_tolerance: int = ICON_DIFF_TOLERANCE_OJAMA,
        high_conf: float = HIGH_CONFIDENCE_THRESHOLD,
        medium_conf: float = MEDIUM_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._tol = diff_tolerance
        self._high = high_conf
        self._medium = medium_conf

    def cross_check(
        self,
        score_delta_ojama: int | None = None,
        visual_icons: list[tuple[str, int]] | None = None,
    ) -> ConsistencyResult:
        """score OCR ベースと視覚版アイコンの整合性をチェックする。

        Args:
            score_delta_ojama: score OCR で計算した ojama 個数 (None なら未測定)
            visual_icons: 視覚版で見えたアイコン構成 (None なら未測定)

        Returns:
            ConsistencyResult
        """
        # どちらも None
        if score_delta_ojama is None and visual_icons is None:
            return ConsistencyResult(
                final_ojama=None,
                confidence=0.0,
                method_used=METHOD_NONE,
                score_delta_ojama=None,
                visual_ojama=None,
                agreement=None,
                details={"reason": "両方未測定"},
            )

        # score 単独
        if visual_icons is None:
            return ConsistencyResult(
                final_ojama=int(max(0, score_delta_ojama or 0)),
                confidence=self._high,  # score OCR は正解率 100%
                method_used=METHOD_FALLBACK_SCORE,
                score_delta_ojama=score_delta_ojama,
                visual_ojama=None,
                agreement=None,
                details={"reason": "視覚版なし、score単独"},
            )

        # 視覚版から ojama 個数を集計
        visual_ojama = icons_to_ojama_count(visual_icons)

        # 視覚版単独
        if score_delta_ojama is None:
            return ConsistencyResult(
                final_ojama=visual_ojama,
                confidence=self._medium,  # 視覚版のみは信頼度中
                method_used=METHOD_FALLBACK_VISUAL,
                score_delta_ojama=None,
                visual_ojama=visual_ojama,
                agreement=None,
                details={"reason": "score未測定、視覚単独"},
            )

        # 両方ある → 整合性チェック
        score_ojama = max(0, int(score_delta_ojama))
        diff = abs(score_ojama - visual_ojama)
        # score 個数からアイコン分解 (端数表示落ち考慮)
        score_icons = ojama_count_to_icons(score_ojama)
        score_icons_total = icons_to_ojama_count(score_icons)
        # 視覚で見える分との差 (表示落ち端数を引いて比較)
        # アイコンに収まる分だけで比較する
        diff_displayed = abs(score_icons_total - visual_ojama)

        # 一致 (許容差内)
        if diff_displayed <= self._tol:
            return ConsistencyResult(
                final_ojama=score_ojama,  # score 採用 (正解率 100%)
                confidence=self._high,
                method_used=METHOD_AGREED,
                score_delta_ojama=score_ojama,
                visual_ojama=visual_ojama,
                agreement=diff,
                details={
                    "score_icons_total": score_icons_total,
                    "diff_displayed": diff_displayed,
                    "tolerance": self._tol,
                },
            )

        # 不一致 → score を採用するが信頼度低下
        # 大きな差 = 視覚版エラー or score OCR の連鎖中異常
        # score OCR は 100% なので score を信じる、ただし confidence 低
        confidence = max(0.2, self._medium - (diff_displayed / 100.0))
        return ConsistencyResult(
            final_ojama=score_ojama,
            confidence=float(confidence),
            method_used=METHOD_SCORE_DELTA,
            score_delta_ojama=score_ojama,
            visual_ojama=visual_ojama,
            agreement=diff,
            details={
                "score_icons_total": score_icons_total,
                "diff_displayed": diff_displayed,
                "reason": "不一致、scoreベース採用",
            },
        )


__all__ = [
    "HIGH_CONFIDENCE_THRESHOLD",
    "ICON_DIFF_TOLERANCE_OJAMA",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "METHOD_AGREED",
    "METHOD_FALLBACK_SCORE",
    "METHOD_FALLBACK_VISUAL",
    "METHOD_NONE",
    "METHOD_SCORE_DELTA",
    "METHOD_VISUAL_CNN",
    "ConsistencyResult",
    "OjamaConsistencyChecker",
]
