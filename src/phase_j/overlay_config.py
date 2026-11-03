"""表示設定の検証とHTMLへの安全な既定値埋め込み。"""
from __future__ import annotations

import json
from typing import Any

LAYOUTS = ('center', 'card', 'ribbon', 'minimal', 'data', 'l-right', 'l-left', 'l', 'vertical')
CORNERS = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
FONT_SIZES = ('small', 'medium', 'large')
THEMES = ('stylish', 'plain', 'normal', 'esports', 'pop', 'pop-cream', 'classic', 'neon', 'arena', 'noir', 'battle-a')
POSITION_LIMITS = {'x': 1920, 'y': 1080}
MIDLINES = ('split', 'line', 'glow', 'adaptive')
MIN_SCALE, MAX_SCALE = 70, 85
MASCOT_EDGE_COUNT, PERCENT = 4, 100
BOOL_KEYS = frozenset({
    'band', 'numbers', 'recent', 'full', 'status', 'details',
    'content_pending', 'content_sent', 'content_offset', 'content_rate',
    'content_hands', 'content_return', 'content_margin', 'content_bound', 'content_unknown',
    'content_active', 'content_step', 'content_length', 'content_remaining', 'content_seconds',
    'content_k1', 'content_colors', 'content_ojama', 'content_model',
})
BOOL_KEYS |= frozenset({'vgraph', 'mascot', 'mascot_eyes', 'mascot_motion', 'mascot_expressions', 'mascot_flip_right', 'mascot_bloodshot'})
# 試験用の情報パネル (L字配置のときだけ表示)。既定OFF。
BOOL_KEYS |= frozenset({'debug_panel'})
ENUMS = dict(layout=LAYOUTS, corner=CORNERS, font=FONT_SIZES, theme=THEMES, midline=MIDLINES,
             language=('auto', 'ja', 'en'), mascot_position=('left', 'right', 'card'),
             mascot_size=('small', 'medium', 'large'), mascot_dopamine=('low', 'normal', 'high'),
             mascot_speed=('0.5', '1', '1.5', '2'))


def validate_overlay(value: Any) -> dict[str, Any]:
    """省略キーは保持せず、配置ごとの初期値をブラウザで解決する。"""
    if not isinstance(value, dict):
        raise ValueError('overlay はオブジェクトで指定してください')
    unknown = value.keys() - BOOL_KEYS - ENUMS.keys() - {'scale', 'mascot_band_edges'} - POSITION_LIMITS.keys()
    if unknown:
        raise ValueError(f'overlay の未知の設定項目: {sorted(unknown)}')
    for key, item in value.items():
        if key in POSITION_LIMITS and (type(item) not in (int, float) or
                                       not 0 <= item <= POSITION_LIMITS[key]):
            raise ValueError(f'overlay.{key} は 0〜{POSITION_LIMITS[key]} です')
        if key in BOOL_KEYS and type(item) is not bool:
            raise ValueError(f'overlay.{key} は true / false です')
        if key in ENUMS and item not in ENUMS[key]:
            raise ValueError(f'overlay.{key} は {ENUMS[key]} から選んでください')
        if key == 'scale' and (type(item) not in (int, float) or
                               not MIN_SCALE <= item <= MAX_SCALE):
            raise ValueError(f'overlay.scale は {MIN_SCALE}〜{MAX_SCALE} です')
    if 'mascot_band_edges' in value:
        edges = value['mascot_band_edges']
        if (not isinstance(edges, list) or len(edges) != MASCOT_EDGE_COUNT or
                any(type(edge) not in (int, float) or not 0 < edge < PERCENT for edge in edges) or
                any(left >= right for left, right in zip(edges, edges[1:]))):
            raise ValueError('overlay.mascot_band_edges は0〜100の間の昇順4値です')
    return dict(value)


def with_debug_defaults(overlay: dict[str, Any]) -> dict[str, Any]:
    """デバッグ期間の既定 (L字+試験用パネル) を足す。利用者が書いた項目は常に優先する。

    既定値は src/production_config.py の LIVE_DEBUG_OVERLAY_* が唯一の出どころ。
    validate_overlay (省略キーを保持しない) には混ぜず、配信サーバの設定にだけ適用する。
    """
    from src.production_config import LIVE_DEBUG_OVERLAY_DEFAULT, LIVE_DEBUG_OVERLAY_VALUES
    if not LIVE_DEBUG_OVERLAY_DEFAULT:
        return dict(overlay)
    return {**dict(LIVE_DEBUG_OVERLAY_VALUES), **overlay}


def configured_html(html: str, options: dict[str, Any]) -> bytes:
    """script終了タグを含む入力もHTMLとして解釈させない。"""
    from .distribution_config import os_language
    encoded = json.dumps({'os_language': os_language(), **options}, ensure_ascii=True).replace('<', '\\u003c')
    return html.replace('/*OVERLAY_DEFAULTS*/{}', encoded).encode('utf-8')
