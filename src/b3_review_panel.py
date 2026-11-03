"""B3 レビュー動画のデータ帯 (下帯) と勝率グラフの描画。

見た目は既存の review_data_panel (E11) の流儀を踏襲する: 幅 1280 基準で 18px、
元画像には触れず下へ連結、1P=水色・2P=橙。値は入力行から機械的に整形するだけで、
モデルや評価器には依存しない (入力は src/b3_review_input.py の行)。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.review_data_panel import (FONT_SIZE, LINE_HEIGHT, PADDING, PANEL_BACKGROUND,
                                       REFERENCE_WIDTH, _text_tile)
from src.b3_review_input import _num, derive, missing_name

PERCENT = 100
GRAPH_HEIGHT = 170                    # 基準幅 1280 でのグラフ帯の高さ (px)
GRAPH_LEFT, GRAPH_RIGHT_PAD = 56, 16
GRAPH_TOP, GRAPH_BOTTOM = 22, 20
COMMON_LINES = 4
SIDE_LINES = 10
PANEL_HEIGHT = GRAPH_HEIGHT + (COMMON_LINES + SIDE_LINES) * LINE_HEIGHT + 2 * PADDING
FONT_CANDIDATES = (r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc")
COLOR_TEXT = (242, 244, 248)
COLOR_P1, COLOR_P2 = (123, 198, 255), (255, 176, 143)
COLOR_B3, COLOR_REF = (255, 255, 255), (120, 200, 120)   # B3 表示値 / 現行本番 R
COLOR_STATIC, COLOR_EXCHANGE = (60, 70, 90), (90, 60, 60)  # 状態の下地
COLOR_PLAYHEAD = (255, 255, 0)
COLOR_DEAD = (255, 90, 90)
STATE_LABELS = {"static": "静止(F_static)", "exchange": "撃ち合い(F_exchange)", "none": "判定なし"}
DEATH_LABELS = {"none": "なし", "suspect": "疑い", "proven_dead": "証明済み死亡"}
SIDE_TITLES = {"p1": "1P攻撃(a)→受け=2P(b)", "p2": "2P攻撃(a)→受け=1P(b)"}


@lru_cache(maxsize=32)
def default_font(size: int) -> ImageFont.ImageFont:
    """meiryo (無ければ既定フォント)。size のみに依存する純関数。"""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def panel_size(width: int, height: int) -> tuple[int, int]:
    """幅に比例して下帯を足した出力サイズ。"""
    return width, height + round(PANEL_HEIGHT * width / REFERENCE_WIDTH)


def fmt(value: Any, spec: str = "{:.3g}") -> str:
    """未観測 (None/NaN) は '--'。0 と区別する。"""
    x = _num(value)
    return "--" if x is None else spec.format(x)


def pct(value: Any) -> str:
    """確率 → '57.3%'。"""
    x = _num(value)
    return "--" if x is None else f"{PERCENT * x:.1f}%"


def common_lines(row: dict[str, Any]) -> list[str]:
    """共通 4 行: 時刻・状態、4 つの勝率、比較基準 R と差、欠測。"""
    mode = STATE_LABELS.get(row.get("mode"), "--")
    ps, pr = _num(row.get("p_display")), _num(row.get("p_ref_display"))
    diff = "--" if ps is None or pr is None else f"{PERCENT * (ps - pr):+.1f}pt"
    phase = {0: "序盤", 1: "中盤", 2: "終盤"}.get(int(_num(row.get("phase")) or -1), "--")
    return [f"B3 REVIEW  {fmt(row.get('t_sec'), '{:.2f}')}s  {row.get('video_id') or '--'}"
            f" 第{fmt(row.get('game_idx'), '{:.0f}')}試合  局面:{phase}  状態:{mode}"
            f"  {row.get('state_reason') or ''}",
            f"1P勝率  F_static {pct(row.get('p_static'))}   F_exchange {pct(row.get('p_exchange'))}"
            f"   採用 {pct(row.get('p_selected'))}   表示(平滑後) {pct(row.get('p_display'))}",
            f"比較基準R(現行本番)  生 {pct(row.get('p_ref'))}   表示 {pct(row.get('p_ref_display'))}"
            f"   差(B3表示−R) {diff}   sl_rate {fmt(row.get('sl_rate'))}点/個",
            f"欠測: {missing_name(row.get('missing_code'))}  {row.get('missing_note') or ''}"
            f"   (正解 1P勝ち={fmt(row.get('won_p1'), '{:.0f}')})"]


def side_lines(row: dict[str, Any], side: str) -> list[str]:
    """側ごと 10 行: 時間軸 H/窓・Q・I・M・残受け/残送り・台帳 v2・アイコン照合・死亡証明・欠測。"""
    def v(name: str, spec: str = "{:.3g}") -> str:
        return fmt(row.get(f"{side}_{name}"), spec)
    gap = derive(row, 0)[f"{side}_ledger_icon_gap"]
    death = row.get(f"{side}_death_state")
    return [
        f"{SIDE_TITLES[side]}  連鎖中(a):{v('active', '{:.0f}')}",
        f"H 中央値/速い {v('H_med')}/{v('H_fast')}手  探索 {v('done_h')}/{v('req_h')}手"
        f" 打切り{v('trunc', '{:.0f}')}  1手 {v('hand_med')}/{v('hand_fast')}秒",
        f"a残り {v('a_rem')}秒  b操作不能 {v('b_busy')}秒  窓 {v('window')}秒",
        f"Q 中央値/速い {v('Q_med')}/{v('Q_fast')}個",
        f"I 台帳v2 {v('I_sl')}  予測完走 {v('I_pred')}  アイコン {v('I_icon')} 個",
        f"M(余地) {v('M')}個  残受け {v('recv')}個  残送り {v('send')}個",
        f"台帳v2 予告 {v('sl_pending')} 生成 {v('sl_generated')} 相殺 {v('sl_offset')}"
        f" 着弾 {v('sl_dropped')} 繰越 {v('sl_carry')}点",
        f"画面アイコン 補正 {v('icon_corr')} 生 {v('icon_raw')}個  台帳−アイコン "
        f"{'--' if gap is None else f'{gap:+.3g}'}個",
        f"死亡証明: {DEATH_LABELS.get(death, '--')}  {row.get(f'{side}_death_basis') or ''}",
        f"欠測 時間軸:{missing_name(row.get(f'{side}_mk_ta'))}"
        f"  台帳:{missing_name(row.get(f'{side}_mk_sl'))}",
    ]


@dataclass
class GraphSeries:
    """グラフ用に全期間の系列を 1 度だけ配列化する (毎フレーム再計算しない)。"""
    t: np.ndarray
    b3: np.ndarray        # p_display (NaN=欠測)
    ref: np.ndarray       # p_ref_display
    mode: list[Optional[str]]

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> "GraphSeries":
        nan = float("nan")
        col = lambda k: np.array([(_num(r.get(k)) if _num(r.get(k)) is not None else nan)
                                  for r in rows], dtype=float)
        return cls(col("t_sec"), col("p_display"), col("p_ref_display"), [r.get("mode") for r in rows])


def _xs(series: GraphSeries, width: int, t_end: float) -> np.ndarray:
    span = max(t_end - float(series.t[0]), 1e-9)
    return GRAPH_LEFT + (series.t - series.t[0]) / span * (width - GRAPH_LEFT - GRAPH_RIGHT_PAD)


def draw_graph(width: int, series: GraphSeries, t_now: float,
               font_loader: Callable[[int], ImageFont.ImageFont]) -> Image.Image:
    """勝率 0〜100% の固定軸に B3 表示値 (白) と R (緑) と状態の下地、再生位置 (黄)。"""
    height = GRAPH_HEIGHT
    img = Image.new("RGB", (width, height), PANEL_BACKGROUND)
    draw, font = ImageDraw.Draw(img), font_loader(FONT_SIZE - 2)
    top, bottom = GRAPH_TOP, height - GRAPH_BOTTOM
    y_of = lambda p: bottom - p * (bottom - top)
    if len(series.t) == 0:
        return img
    t_end = float(series.t[-1])
    xs = _xs(series, width, t_end)
    for i in range(len(xs) - 1):
        color = {"static": COLOR_STATIC, "exchange": COLOR_EXCHANGE}.get(series.mode[i])
        if color is not None:
            draw.rectangle((xs[i], top, max(xs[i + 1], xs[i] + 1), bottom), fill=color)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        draw.line((GRAPH_LEFT, y_of(tick), width - GRAPH_RIGHT_PAD, y_of(tick)),
                  fill=(95, 105, 120), width=1)
        draw.text((2, y_of(tick) - FONT_SIZE / 2), f"{PERCENT * tick:.0f}", font=font, fill=COLOR_TEXT)
    for values, color, line_w in ((series.ref, COLOR_REF, 2), (series.b3, COLOR_B3, 3)):
        _polyline(draw, xs, values, y_of, color, line_w)
    span = max(t_end - float(series.t[0]), 1e-9)
    frac = max(0.0, min((t_now - float(series.t[0])) / span, 1.0))
    px = GRAPH_LEFT + frac * (width - GRAPH_LEFT - GRAPH_RIGHT_PAD)
    draw.line((px, top, px, bottom), fill=COLOR_PLAYHEAD, width=2)
    draw.text((GRAPH_LEFT, 0), "1P勝率  白=B3表示  緑=R(現行本番)  背景 青灰=静止/赤灰=撃ち合い",
              font=font, fill=COLOR_TEXT)
    draw.text((GRAPH_LEFT, bottom), f"{series.t[0]:.0f}秒", font=font, fill=COLOR_TEXT)
    draw.text((width - GRAPH_RIGHT_PAD - 70, bottom), f"{t_end:.0f}秒", font=font, fill=COLOR_TEXT)
    return img


def _polyline(draw: ImageDraw.ImageDraw, xs: np.ndarray, values: np.ndarray,
              y_of: Callable[[float], float], color: tuple[int, int, int], line_w: int) -> None:
    """NaN で途切れる折れ線 (欠測区間を補間で埋めない)。"""
    run: list[tuple[float, float]] = []
    for x, p in zip(xs, values):
        if np.isfinite(p):
            run.append((float(x), y_of(float(p))))
            continue
        if len(run) > 1:
            draw.line(run, fill=color, width=line_w)
        run = []
    if len(run) > 1:
        draw.line(run, fill=color, width=line_w)


def draw_b3_panel(frame: np.ndarray, row: Optional[dict[str, Any]], series: GraphSeries,
                  t_now: float, font_loader: Callable[[int], ImageFont.ImageFont] = default_font
                  ) -> np.ndarray:
    """元フレーム (BGR) の下へ帯を連結して返す。row が None なら '評価前' と出す。"""
    height, width = frame.shape[:2]
    scale = width / REFERENCE_WIDTH
    band = Image.new("RGB", (width, panel_size(width, height)[1] - height), PANEL_BACKGROUND)
    pad, step, font_size = round(PADDING * scale), round(LINE_HEIGHT * scale), round(FONT_SIZE * scale)
    graph = draw_graph(REFERENCE_WIDTH, series, t_now, font_loader)
    band.paste(graph.resize((width, round(GRAPH_HEIGHT * scale)), Image.LANCZOS), (0, 0))
    y0 = round(GRAPH_HEIGHT * scale) + pad
    lines = common_lines(row) if row else ["B3 REVIEW  評価前 (最初の評価行より前の時刻)", "", "", ""]
    for i, line in enumerate(lines):
        band.paste(_text_tile(line, font_size, font_loader, COLOR_TEXT, width - 2 * pad, step),
                   (pad, y0 + i * step))
    top = y0 + len(lines) * step
    for k, side in enumerate(("p1", "p2")):
        color = COLOR_P1 if k == 0 else COLOR_P2
        for i, line in enumerate(side_lines(row, side) if row else []):
            tile = _text_tile(line, font_size, font_loader,
                              COLOR_DEAD if "証明済み" in line else color,
                              width // 2 - 2 * pad, step)
            band.paste(tile, (pad + k * width // 2, top + i * step))
    ImageDraw.Draw(band).line((width // 2, top, width // 2, band.height - pad), fill=(70, 80, 96))
    return np.concatenate((frame, cv2.cvtColor(np.asarray(band), cv2.COLOR_RGB2BGR)), axis=0)
