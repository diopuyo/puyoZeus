"""
オーバーレイ描画エンジン

AnalysisResult を元にフレーム画像へ有利不利表示を合成する。
OpenCV を用いて描画し、動画合成 (video_compositer) と
配信オーバーレイ (stream_overlay) の両方から利用される。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.analyzer import AnalysisResult
from src.old.indicators import ALL_INDICATOR_NAMES
from src.old.scorer import (
    ADVANTAGE_EVEN,
    PLAYER_1P,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
)

# ============================
# 色定数 (BGR)
# ============================

COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: tuple[int, int, int] = (0, 0, 0)
COLOR_GRAY: tuple[int, int, int] = (128, 128, 128)
COLOR_1P: tuple[int, int, int] = (255, 100, 100)   # 水色寄り青
COLOR_2P: tuple[int, int, int] = (100, 100, 255)   # 赤寄り
COLOR_EVEN: tuple[int, int, int] = (200, 200, 200)
COLOR_BACKGROUND: tuple[int, int, int] = (20, 20, 20)

# ============================
# レイアウト定数
# ============================

# スコアバー
SCORE_BAR_WIDTH_RATIO: float = 0.4    # フレーム幅に対する割合
SCORE_BAR_HEIGHT: int = 24
SCORE_BAR_TOP_MARGIN: int = 40
SCORE_BAR_RADIUS: int = 4

# 指標バー (各プレイヤー)
INDICATOR_BAR_WIDTH: int = 140
INDICATOR_BAR_HEIGHT: int = 10
INDICATOR_BAR_SPACING: int = 18
INDICATOR_PANEL_MARGIN: int = 30
INDICATOR_PANEL_PADDING: int = 10

# フォント
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SCORE: float = 0.9
FONT_SCALE_LABEL: float = 0.45
FONT_THICKNESS: int = 1

# 背景パネルの透過率
PANEL_ALPHA: float = 0.55

# 8指標の日本語表示名
INDICATOR_LABELS_JA: dict[str, str] = {
    "main_chain_maturity": "本線",
    "extension_potential": "伸ばし",
    "sub_chain_quality":   "副砲",
    "harassment_resistance": "催促耐性",
    "death_risk":          "窒息",
    "offset_power":        "相殺",
    "second_chain_potential": "セカンド",
    "field_efficiency":    "効率",
}


# ============================
# スタイル設定
# ============================


@dataclass
class OverlayStyle:
    """
    オーバーレイ描画スタイル。将来的にテーマ切替に対応。
    """
    show_score_bar: bool = True
    show_indicator_panels: bool = True
    show_advantage_label: bool = True
    panel_alpha: float = PANEL_ALPHA
    color_1p: tuple[int, int, int] = COLOR_1P
    color_2p: tuple[int, int, int] = COLOR_2P
    color_even: tuple[int, int, int] = COLOR_EVEN


# ============================
# OverlayRenderer
# ============================


class OverlayRenderer:
    """
    AnalysisResult をフレームに合成描画するレンダラー。

    Usage:
        renderer = OverlayRenderer()
        output_frame = renderer.render(input_frame, analysis_result)
    """

    def __init__(self, style: OverlayStyle | None = None) -> None:
        """
        Args:
            style: 描画スタイル (None ならデフォルト)。
        """
        self._style = style or OverlayStyle()

    # ============================
    # 公開メソッド
    # ============================

    def render(
        self,
        frame: np.ndarray,
        result: AnalysisResult,
    ) -> np.ndarray:
        """
        フレームに分析結果を合成して返す (非破壊)。

        Args:
            frame: BGR フレーム (H×W×3)。
            result: 分析結果。

        Returns:
            np.ndarray: オーバーレイを合成した新しいフレーム。
        """
        canvas = frame.copy()

        if self._style.show_score_bar:
            self._draw_score_bar(canvas, result)

        if self._style.show_indicator_panels:
            self._draw_indicator_panel(
                canvas, result, player_side=PLAYER_1P
            )
            self._draw_indicator_panel(
                canvas, result, player_side="2P"
            )

        return canvas

    def render_transparent(
        self,
        width: int,
        height: int,
        result: AnalysisResult,
    ) -> np.ndarray:
        """
        透過 BGRA 画像にオーバーレイのみ描画する (配信用)。

        Args:
            width: 出力画像幅。
            height: 出力画像高さ。
            result: 分析結果。

        Returns:
            np.ndarray: shape=(H, W, 4) の BGRA 透過画像。
        """
        # 透過背景 (A=0)
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        # 描画用に BGR 作業用キャンバスを生成
        work = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        self._draw_score_bar(work, result, mask=mask)
        self._draw_indicator_panel(work, result, PLAYER_1P, mask=mask)
        self._draw_indicator_panel(work, result, "2P", mask=mask)

        bgra[..., :3] = work
        bgra[..., 3] = mask
        return bgra

    # ============================
    # 描画ブロック
    # ============================

    def _draw_score_bar(
        self,
        canvas: np.ndarray,
        result: AnalysisResult,
        mask: np.ndarray | None = None,
    ) -> None:
        """画面上部に総合スコアバーを描画する。"""
        h, w = canvas.shape[:2]
        bar_w = int(w * SCORE_BAR_WIDTH_RATIO)
        bar_x = (w - bar_w) // 2
        bar_y = SCORE_BAR_TOP_MARGIN

        # 背景バー
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + SCORE_BAR_HEIGHT),
            COLOR_BACKGROUND,
            -1,
        )
        if mask is not None:
            cv2.rectangle(
                mask,
                (bar_x, bar_y),
                (bar_x + bar_w, bar_y + SCORE_BAR_HEIGHT),
                255, -1,
            )

        # スコア位置 (中心基準)
        score = result.score.total_score
        ratio = score / SCORE_RANGE_MAX  # -1.0〜+1.0
        center_x = bar_x + bar_w // 2
        fill_x = int(center_x + ratio * bar_w / 2)

        color = self._color_for_score(score)
        x1, x2 = (center_x, fill_x) if score >= 0 else (fill_x, center_x)
        cv2.rectangle(
            canvas,
            (x1, bar_y + 2),
            (x2, bar_y + SCORE_BAR_HEIGHT - 2),
            color, -1,
        )
        if mask is not None:
            cv2.rectangle(
                mask,
                (x1, bar_y + 2),
                (x2, bar_y + SCORE_BAR_HEIGHT - 2),
                255, -1,
            )

        # 中心線
        cv2.line(
            canvas,
            (center_x, bar_y),
            (center_x, bar_y + SCORE_BAR_HEIGHT),
            COLOR_WHITE, 1,
        )

        # ラベル
        if self._style.show_advantage_label:
            text = self._advantage_text(result)
            self._put_text_centered(
                canvas, text,
                center_x, bar_y + SCORE_BAR_HEIGHT + 22,
                scale=FONT_SCALE_SCORE, color=COLOR_WHITE,
                mask=mask,
            )

    def _draw_indicator_panel(
        self,
        canvas: np.ndarray,
        result: AnalysisResult,
        player_side: str,
        mask: np.ndarray | None = None,
    ) -> None:
        """プレイヤー毎の指標バーパネルを描画する。"""
        h, w = canvas.shape[:2]
        indicators = (
            result.player1.indicators
            if player_side == PLAYER_1P
            else result.player2.indicators
        )

        panel_h = (
            INDICATOR_PANEL_PADDING * 2
            + INDICATOR_BAR_SPACING * len(ALL_INDICATOR_NAMES)
        )
        panel_w = INDICATOR_BAR_WIDTH + INDICATOR_PANEL_PADDING * 2

        if player_side == PLAYER_1P:
            px = INDICATOR_PANEL_MARGIN
        else:
            px = w - INDICATOR_PANEL_MARGIN - panel_w
        py = h - INDICATOR_PANEL_MARGIN - panel_h

        self._draw_panel_background(canvas, px, py, panel_w, panel_h, mask)

        # 各指標のバー
        for i, name in enumerate(ALL_INDICATOR_NAMES):
            by = py + INDICATOR_PANEL_PADDING + i * INDICATOR_BAR_SPACING
            label = INDICATOR_LABELS_JA.get(name, name)
            score = indicators.score_of(name)
            color = (
                self._style.color_1p if player_side == PLAYER_1P
                else self._style.color_2p
            )
            self._draw_indicator_bar(
                canvas,
                x=px + INDICATOR_PANEL_PADDING,
                y=by,
                label=label,
                score=score,
                color=color,
                mask=mask,
            )

    # ============================
    # 描画ユーティリティ
    # ============================

    def _draw_panel_background(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        mask: np.ndarray | None,
    ) -> None:
        """半透明の背景パネルを描画する。"""
        overlay = canvas.copy()
        cv2.rectangle(
            overlay, (x, y), (x + w, y + h),
            COLOR_BACKGROUND, -1,
        )
        alpha = self._style.panel_alpha
        cv2.addWeighted(
            overlay, alpha, canvas, 1 - alpha, 0, canvas,
        )
        if mask is not None:
            panel_alpha_value = int(255 * alpha)
            cv2.rectangle(
                mask, (x, y), (x + w, y + h),
                panel_alpha_value, -1,
            )

    def _draw_indicator_bar(
        self,
        canvas: np.ndarray,
        x: int, y: int,
        label: str,
        score: float,
        color: tuple[int, int, int],
        mask: np.ndarray | None,
    ) -> None:
        """1つの指標バーを描画する。"""
        bar_x = x + 60  # ラベル分のオフセット
        bar_w_max = INDICATOR_BAR_WIDTH - 60
        fill_w = int(bar_w_max * max(0.0, min(1.0, score)))

        cv2.rectangle(
            canvas,
            (bar_x, y),
            (bar_x + bar_w_max, y + INDICATOR_BAR_HEIGHT),
            COLOR_GRAY, 1,
        )
        if fill_w > 0:
            cv2.rectangle(
                canvas,
                (bar_x, y),
                (bar_x + fill_w, y + INDICATOR_BAR_HEIGHT),
                color, -1,
            )

        cv2.putText(
            canvas, label, (x, y + INDICATOR_BAR_HEIGHT),
            FONT, FONT_SCALE_LABEL, COLOR_WHITE, FONT_THICKNESS,
            cv2.LINE_AA,
        )

        if mask is not None:
            cv2.rectangle(
                mask,
                (x, y),
                (x + INDICATOR_BAR_WIDTH, y + INDICATOR_BAR_HEIGHT),
                255, -1,
            )

    def _put_text_centered(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int, cy: int,
        scale: float,
        color: tuple[int, int, int],
        mask: np.ndarray | None,
    ) -> None:
        """中心寄せでテキストを描画する。"""
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, FONT_THICKNESS)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(
            canvas, text, (tx, ty),
            FONT, scale, color, FONT_THICKNESS, cv2.LINE_AA,
        )
        if mask is not None:
            cv2.rectangle(
                mask, (tx, ty - th), (tx + tw, ty + 4), 255, -1,
            )

    def _advantage_text(self, result: AnalysisResult) -> str:
        """有利側ラベルとスコアを返す。"""
        side = result.score.advantage_side()
        score = result.score.total_score
        if side == ADVANTAGE_EVEN:
            return f"EVEN {score:+.1f}"
        return f"{side} 有利 {score:+.1f}"

    def _color_for_score(self, score: float) -> tuple[int, int, int]:
        """スコア符号に応じた色を返す。"""
        if score > 0:
            return self._style.color_1p
        if score < 0:
            return self._style.color_2p
        return self._style.color_even
