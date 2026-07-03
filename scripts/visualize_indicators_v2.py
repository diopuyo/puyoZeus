"""指標 v2 を日本語ラベルで盤面横に並べる目視照合用レビュー動画を生成する。

モード (--mode):
    - indicator (既定): 計算指標 21 項目をカテゴリ別に表示 (従来挙動)。
    - raw: 原観測量 (画面から直接読める生データ) のみ表示。計算指標は出さない。
      項目 = 状態 / score / 手数 / 試合内経過秒 / 色別個数(赤青緑黄紫)+色ぷよ総数 /
      お邪魔数 / 列高さ×6 / next / dnext / 予告お邪魔(score会計推定)。
      認識が生データを正しく読めているかを人が盤面と照合するためのビュー。
      フォントは大きめ・色別個数は該当色の文字色で表示。

目的:
    `src/indicators_v2.py` の各指標が「画面の盤面と合っているか」を人が目視で
    照合できる動画を作る。各 STABLE snapshot で算出した指標値を、カテゴリ
    (①進行度/②占有・危険/③火力・潜在/④お邪魔/⑤テンポ/⑥受け力) ごとに
    グルーピングし、日本語ラベル + 値 を半透明パネルで盤面の左右に表示する。

技術要件:
    - 日本語描画は PIL(Pillow) + 日本語 TrueType フォントで行う
      (cv2 FONT_HERSHEY は日本語非対応で ??? 化する。実証済み)。
      Windows / WSL 両対応でフォントを解決し、見つからなければローマ字 fallback。
    - cv2 フレーム (numpy BGR) → PIL に変換して日本語テキスト描画 → 戻す方式。
    - 認識 overlay (盤面のぷよ色記号・state 枠) は visualize_recognition の
      描画関数を流用して活かす。
    - 指標算出ロジックは scripts/collect_indicators_v2.py を流用 (重複実装回避)。
    - STABLE 時に算出、非 STABLE 中は直前の STABLE 値を凍結表示 (認識 overlay と同条件)。

駆動:
    - RecognitionPipeline.load_default (自動 HSV のみ = --no-per-video-hsv 相当)。
    - OjamaAccountingTracker を on_state_transition + on_tsumo_settled + get_snapshot
      で駆動し ④⑤指標に供給 (collect_indicators_v2._drive_ojama を流用)。

使い方 (短尺 smoke):
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_indicators_v2 \
        --video data/frames/video_124_4min.mp4 \
        --output data/indicators_v2/viz/video_124_smoke.mp4 \
        --max-sec 40 --dump-png-dir data/indicators_v2/viz/smoke_png

注意 (申し送り):
    - III-3 到達火力はチャンク2未実装のためパネルに「未実装」と表示する。
    - V 連鎖所要時間は観測 (chain_event) が無いと推定/0 になりやすく大半 0。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

from src.board import (  # noqa: E402
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    OjamaAccountingTracker,
    OjamaAccountSnapshot,
)
from src.recognition_pipeline import RecognitionPipeline, SideResult  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# 認識 overlay の描画関数を流用 (盤面ぷよ色 / state 枠)
from scripts.visualize_recognition import (  # noqa: E402
    P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y,
    draw_cell_overlay, draw_state_label, draw_global_info,
)
# 指標算出ロジックを流用 (重複実装回避)
from scripts.collect_indicators_v2 import (  # noqa: E402
    TARGET_W, TARGET_H, DEFAULT_FPS,
    _SideTracker, _should_emit, _update_game_idx, _chain_duration, _drive_ojama,
)

# ============================
# 描画定数
# ============================

# 出力解像度 (認識は 1920x1080 前提)
OUT_W: int = TARGET_W
OUT_H: int = TARGET_H

# サンプリング間隔 (秒)。30fps 認識。
DEFAULT_SAMPLE_INTERVAL: float = 0.033

# 指標パネルの配置 (1P=左余白、2P=右余白)。盤面 ROI と被らない X 帯に置く。
# 盤面: 1P x=282..666, 2P x=1258..1642。左余白 0..282、右余白 1642..1920。
# 左右余白は狭い (~280px) ので、パネルを盤面 ROI 上にも一部重ねて配置する。
_PANEL_W: int = 300
_P1_PANEL_X: int = 4
_P2_PANEL_X: int = OUT_W - _PANEL_W - 4
_PANEL_TOP_Y: int = 96
_PANEL_BOT_MARGIN: int = 8

# フォントサイズ (px)
_FS_TITLE: int = 22       # side タイトル (1P / 2P + 状態)
_FS_CATEGORY: int = 18    # カテゴリ見出し
_FS_KEY_VALUE: int = 18   # 重要指標 (大きめ)
_FS_VALUE: int = 15       # 通常指標
_LINE_GAP: int = 3        # 行間 (追加)

# 半透明パネル背景 (BGR 描画は PIL では RGB)。アルファ合成は cv2 側で行う。
_PANEL_BG_BGR: tuple[int, int, int] = (24, 24, 24)
_PANEL_ALPHA: float = 0.62

# state 日本語名
_STATE_JA: dict[BoardState, str] = {
    BoardState.STABLE: "安定 (STABLE)",
    BoardState.TSUMO_FALL: "ツモ落下中",
    BoardState.CHAIN: "連鎖中",
    BoardState.OJAMA_FALL: "お邪魔落下中",
    BoardState.MENU: "メニュー",
    BoardState.EFFECT: "エフェクト中",
    BoardState.GRAVITY_SETTLE: "重力settle中",
}

# テキスト色 (RGB)
_COL_TITLE: tuple[int, int, int] = (255, 255, 255)
_COL_CATEGORY: tuple[int, int, int] = (150, 220, 255)   # 水色
_COL_KEY: tuple[int, int, int] = (255, 240, 160)        # 黄系 (重要指標)
_COL_VALUE: tuple[int, int, int] = (220, 220, 220)      # 薄白
_COL_FROZEN: tuple[int, int, int] = (160, 160, 160)     # 凍結中 (グレー)
_COL_NOTE: tuple[int, int, int] = (200, 160, 160)       # 注記 (未実装等)

# ============================
# raw モード (原観測量) 専用定数
# ============================

# raw モードは項目が少ないのでフォントを大きめにして視認性を上げる。
_RAW_FS_TITLE: int = 30        # side タイトル (1P / 2P + 状態)
_RAW_FS_LABEL: int = 24        # 項目ラベル + 値
_RAW_LINE_GAP: int = 7         # 行間 (追加)
_RAW_PANEL_W: int = 360        # raw パネル幅 (項目が長いので広め)
_RAW_INDENT: int = 10          # ラベルのインデント

# ============================
# verify モード (選択指標の生値のみ) 専用定数
# ============================

# verify モードは10行程度なのでフォントをさらに大きくして目視照合しやすくする。
_VFY_FS_TITLE: int = 32        # side タイトル
_VFY_FS_LABEL: int = 26        # 項目ラベル + 値
_VFY_LINE_GAP: int = 8         # 行間 (追加)
_VFY_PANEL_W: int = 380        # パネル幅 (生値+ラベルが入る幅)
_VFY_INDENT: int = 10          # ラベルのインデント

# verify: net収支の符号説明をサブ行として色付き表示。
_COL_VERIFY_POS: tuple[int, int, int] = (100, 230, 100)   # 正値 (緑: 自分優勢)
_COL_VERIFY_NEG: tuple[int, int, int] = (255, 100, 100)   # 負値 (赤: 相手優勢)
_COL_VERIFY_ZERO: tuple[int, int, int] = (200, 200, 200)  # ゼロ

# 色別個数を該当色の文字色で表示するための RGB (PIL)。盤面照合しやすく。
_PUYO_TEXT_RGB: dict[int, tuple[int, int, int]] = {
    COLOR_RED:    (255, 90, 90),
    COLOR_BLUE:   (90, 150, 255),
    COLOR_GREEN:  (90, 230, 110),
    COLOR_YELLOW: (255, 225, 70),
    COLOR_PURPLE: (215, 130, 255),
}
# 色別個数の表示順 (赤青緑黄紫) と日本語ラベル。
_PUYO_ORDER: tuple[tuple[int, str], ...] = (
    (COLOR_RED, "赤"),
    (COLOR_BLUE, "青"),
    (COLOR_GREEN, "緑"),
    (COLOR_YELLOW, "黄"),
    (COLOR_PURPLE, "紫"),
)
# next / dnext を日本語色名で出すための色記号 (お邪魔会計と非依存)。
_COLOR_NAME_JA: dict[int, str] = {
    COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫",
}


# ============================
# フォント解決 (PIL)
# ============================

# 日本語フォント候補 (Windows / WSL の両パス)。先頭から存在チェック。
_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\YuGothR.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothM.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
    # Linux 環境の日本語フォント (念のため)
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _resolve_font_path() -> str | None:
    """存在する日本語 TrueType フォントのパスを返す (無ければ None)。"""
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    return None


class _FontSet:
    """サイズ別フォントを保持する。日本語フォント不在時は default + romaji。"""

    def __init__(self) -> None:
        path = _resolve_font_path()
        self.has_japanese: bool = path is not None
        self.path: str | None = path
        self._cache: dict[int, ImageFont.FreeTypeFont] = {}

    def font(self, size: int) -> ImageFont.ImageFont:
        """指定 px サイズのフォントを返す (キャッシュ)。"""
        if size in self._cache:
            return self._cache[size]
        if self.path is not None:
            try:
                f = ImageFont.truetype(self.path, size)
            except Exception:
                f = ImageFont.load_default()
        else:
            f = ImageFont.load_default()
        self._cache[size] = f
        return f


# ============================
# 指標 → 表示行の構築
# ============================

# romaji fallback ラベル (日本語フォント不在時のみ使用)
_ROMAJI: dict[str, str] = {
    "①進行度": "(1) Progress",
    "②占有・危険": "(2) Occupancy/Danger",
    "③火力・潜在": "(3) Power/Potential",
    "④お邪魔": "(4) Ojama",
    "⑤テンポ": "(5) Tempo",
    "⑥受け力": "(6) Defense",
}


def _fmt(score: float, raw: float | int, *, raw_int: bool = False) -> str:
    """score(0-1) と raw を併記した値文字列を返す。"""
    if raw_int:
        return f"{score:.2f} (生{int(raw)})"
    return f"{score:.2f} (生{raw:.2f})"


def _build_indicator_lines(
    board: Board,
    tsumo: int,
    elapsed_sec: float,
    net: int,
    forecast: int,
    board_ojama_offboard: int,
    side: SideResult,
) -> list[tuple[str, str, str]]:
    """1 side の表示行を [(kind, label, value), ...] で構築する。

    kind: "cat"(カテゴリ見出し) / "key"(重要指標) / "val"(通常) / "note"(注記)。
    indicators_v2 を直接呼び出す (collect_indicators_v2 と同じ算出関数)。
    """
    total_conn, _ = iv.connectivity_observation(board)
    tc = iv.tsumo_count_rate(tsumo)
    bp = iv.board_puyo_total(board)
    bc = iv.board_color_puyo_total(board)
    mt = iv.margin_time_rate(elapsed_sec)
    mh = iv.max_column_height(board)
    bm = iv.column_bumpiness(board)
    dm = iv.death_margin(board)
    dn = iv.death_margin_neighbor(board)
    cm = iv.current_max_chain(board)
    ifp = iv.immediate_fire_power(board, elapsed_sec)
    ce = iv.chain_efficiency(board, elapsed_sec)
    mi = iv.min_puyos_to_ignite(board)
    sc = iv.second_chain_potential(board)
    nb = iv.ojama_net_balance(net)
    fc = iv.ojama_forecast(forecast)
    bo = iv.board_ojama_count(board)
    dr = iv.dig_resistance(board)
    ab = iv.absorption_capacity(board)
    dur, dur_src = _chain_duration(side)
    dur_src_ja = {"observed": "観測", "estimated": "推定", "none": "なし"}.get(
        dur_src, dur_src,
    )

    lines: list[tuple[str, str, str]] = []
    # ① 進行度
    lines.append(("cat", "①進行度", ""))
    lines.append(("key", "手数(ツモ)", _fmt(tc.score, tc.raw, raw_int=True)))
    lines.append(("key", "盤面ぷよ数", _fmt(bp.score, bp.raw, raw_int=True)))
    lines.append(("val", "色ぷよ数", _fmt(bc.score, bc.raw, raw_int=True)))
    lines.append(("val", "マージンtime率", _fmt(mt.score, mt.raw)))
    # ② 占有・危険
    lines.append(("cat", "②占有・危険", ""))
    lines.append(("key", "窒息余裕(3列)", _fmt(dm.score, dm.raw, raw_int=True)))
    lines.append(("val", "窒息余裕(近接)", _fmt(dn.score, dn.raw, raw_int=True)))
    lines.append(("val", "最大列高さ", _fmt(mh.score, mh.raw, raw_int=True)))
    lines.append(("val", "列凸凹", _fmt(bm.score, bm.raw, raw_int=True)))
    # ③ 火力・潜在
    lines.append(("cat", "③火力・潜在", ""))
    lines.append(("key", "最大連鎖数", _fmt(cm.score, cm.raw, raw_int=True)))
    lines.append(("key", "即発火火力(お邪魔)", _fmt(ifp.score, ifp.raw, raw_int=True)))
    lines.append(("key", "連鎖効率", _fmt(ce.score, ce.raw)))
    lines.append(("val", "発火最短手数", _fmt(mi.score, mi.raw, raw_int=True)))
    lines.append(("val", "セカンド潜在", _fmt(sc.score, sc.raw, raw_int=True)))
    lines.append((
        "val", "連結(2/3/最大)",
        f"{total_conn.pair_count}/{total_conn.triple_count}/{total_conn.max_group_size}",
    ))
    lines.append(("note", "到達火力(III-3)", "未実装(チャンク2)"))
    # ④ お邪魔
    lines.append(("cat", "④お邪魔", ""))
    lines.append(("key", "お邪魔収支net", _fmt(nb.score, nb.raw, raw_int=True)))
    lines.append(("key", "予告お邪魔forecast", _fmt(fc.score, fc.raw, raw_int=True)))
    lines.append(("val", "盤面お邪魔数", _fmt(bo.score, bo.raw, raw_int=True)))
    if board_ojama_offboard > 0:
        lines.append(("val", "画面外お邪魔(推定)", f"+{board_ojama_offboard}"))
    # ⑤ テンポ
    lines.append(("cat", "⑤テンポ", ""))
    if dur is not None:
        lines.append((
            "val", "連鎖所要時間",
            f"{dur.score:.2f} (生{dur.raw:.2f}s/{dur_src_ja})",
        ))
    else:
        lines.append(("val", "連鎖所要時間", f"0.00 ({dur_src_ja})"))
    # ⑥ 受け力
    lines.append(("cat", "⑥受け力", ""))
    lines.append(("key", "受け力(掘り耐性)", _fmt(dr.score, dr.raw)))
    lines.append(("val", "吸収余地", _fmt(ab.score, ab.raw, raw_int=True)))
    return lines


# ============================
# raw モード: 原観測量の表示行構築
# ============================

# raw 行の型: (label, value, rgb)。色別個数のみ rgb で該当色を指定。
_RawLine = tuple[str, str, tuple[int, int, int]]


def _pair_color_names(pair: tuple[int, int] | None) -> str:
    """next/dnext ペア (2 色) を日本語色名 2 文字で返す (不明は「?」)。"""
    if pair is None:
        return "--"
    a = _COLOR_NAME_JA.get(int(pair[0]), "?")
    b = _COLOR_NAME_JA.get(int(pair[1]), "?")
    return f"{a}{b}"


def _count_color(board: Board, color: int) -> int:
    """盤面の可視+隠し領域で指定色のセル数を数える (生データ照合用)。"""
    return int((board._grid == color).sum())


def _build_raw_lines(
    board: Board,
    state: BoardState,
    score: int | None,
    tsumo: int,
    elapsed_sec: float,
    forecast: int,
    visible_ojama: int,
    side: SideResult,
) -> list[_RawLine]:
    """1 side の原観測量 (生データ) 表示行を構築する。

    計算指標 (連鎖数/火力/効率 等) は一切含めない。画面から直接読める量のみ。
    """
    lines: list[_RawLine] = []
    # score / 手数 / 経過秒
    score_s = "--" if score is None else f"{score:,}"
    lines.append(("score", score_s, _COL_VALUE))
    lines.append(("手数(ツモ)", f"{tsumo}", _COL_VALUE))
    lines.append(("試合内経過", f"{elapsed_sec:.1f} 秒", _COL_VALUE))
    # 色別個数 (該当色の文字色で) + 色ぷよ総数
    color_total = 0
    for color, name in _PUYO_ORDER:
        n = _count_color(board, color)
        color_total += n
        lines.append((f"  {name}", f"{n}", _PUYO_TEXT_RGB[color]))
    lines.append(("色ぷよ総数", f"{color_total}", _COL_KEY))
    # お邪魔数 (盤面可視)
    lines.append(("お邪魔数(盤面)", f"{visible_ojama}", _COL_VALUE))
    # 列高さ ×6
    heights = [board.height_of(c) for c in range(BOARD_COLS)]
    lines.append(("列高さ[0-5]", str(heights), _COL_VALUE))
    # next / dnext (色ペア)
    lines.append(("next", _pair_color_names(side.next_pair), _COL_VALUE))
    lines.append(("dnext", _pair_color_names(side.dnext_pair), _COL_VALUE))
    # 予告お邪魔 (会計モデル出力。生認識でない旨を明記)
    lines.append(("予告お邪魔", f"{forecast}", _COL_KEY))
    lines.append(("  (score会計推定)", "", _COL_NOTE))
    return lines


# ============================
# verify モード: 選択指標の生値のみ表示行構築
# ============================


def _net_balance_color(net: int) -> tuple[int, int, int]:
    """net収支の符号で文字色を返す (正=自分優勢=緑、負=赤)。"""
    if net > 0:
        return _COL_VERIFY_POS
    if net < 0:
        return _COL_VERIFY_NEG
    return _COL_VERIFY_ZERO


def _build_verify_lines(
    board: Board,
    net: int,
    forecast: int,
) -> list[_RawLine]:
    """verify モード: 選択10指標の生値のみで表示行を構築する。

    正規化スコアは表示しない。生値(整数/浮動小数)を主役に配置。
    照合ポイント: 「盤面ぷよ総数 = 色ぷよ総数 + お邪魔数」「最大連結が盤面と一致」。
    """
    # 盤面ぷよ総数 (お邪魔込み)
    bp = iv.board_puyo_total(board)
    bp_total = int(bp.raw)
    # 色ぷよ総数 (お邪魔除く)
    bc = iv.board_color_puyo_total(board)
    bc_total = int(bc.raw)
    # 盤面お邪魔数
    bo = iv.board_ojama_count(board)
    bo_total = int(bo.raw)
    # 連結観測
    total_conn, _ = iv.connectivity_observation(board)
    pair_cnt = total_conn.pair_count
    triple_cnt = total_conn.triple_count
    max_group = total_conn.max_group_size
    # 現在の最大連鎖数 (takapt 方式)
    cm = iv.current_max_chain(board)
    cm_raw = int(cm.raw)
    # 吸収余地 (空きセル数)
    ab = iv.absorption_capacity(board)
    ab_raw = int(ab.raw)

    # 盤面ぷよ数の照合サブ行: total = color + ojama
    detail = f"(色{bc_total} + お邪魔{bo_total})"
    net_col = _net_balance_color(net)

    lines: list[_RawLine] = []
    lines.append(("盤面ぷよ総数", f"{bp_total}", _COL_KEY))
    lines.append(("  内訳", detail, _COL_NOTE))
    lines.append(("色ぷよ総数", f"{bc_total}", _COL_VALUE))
    lines.append(("2連結数", f"{pair_cnt}", _COL_VALUE))
    lines.append(("3連結数", f"{triple_cnt}", _COL_VALUE))
    lines.append(("最大連結", f"{max_group}", _COL_KEY))
    lines.append(("最大連鎖数", f"{cm_raw} 連鎖", _COL_KEY))
    lines.append(("net収支", f"{net:+d}", net_col))
    lines.append(("  (+自分送過剰/-受超過)", "", _COL_NOTE))
    lines.append(("予告お邪魔(forecast)", f"{forecast}", _COL_KEY))
    lines.append(("盤面お邪魔数", f"{bo_total}", _COL_VALUE))
    lines.append(("吸収余地(空き)", f"{ab_raw}", _COL_VALUE))
    return lines


def draw_verify_panel(
    frame: np.ndarray,
    fonts: _FontSet,
    panel_x: int,
    side_label: str,
    state: BoardState,
    is_frozen: bool,
    lines: list[_RawLine],
) -> None:
    """1 side の verify パネルを frame (BGR) に描画する (大きめフォント)。

    draw_raw_panel と同パターン。_VFY_* 定数で大きめフォントを使用。
    """
    line_h = _VFY_FS_LABEL + _VFY_LINE_GAP
    title_h = _VFY_FS_TITLE + _VFY_LINE_GAP + 8
    panel_h = title_h + line_h * len(lines) + 14
    py0 = _PANEL_TOP_Y
    py1 = min(OUT_H - _PANEL_BOT_MARGIN, py0 + panel_h)
    px0 = panel_x
    px1 = panel_x + _VFY_PANEL_W

    # 半透明背景 (cv2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px1, py1), _PANEL_BG_BGR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1.0 - _PANEL_ALPHA, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px1, py1), (110, 110, 110), 1)

    # PIL でテキスト重畳
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    tx = px0 + _VFY_INDENT
    ty = py0 + 6
    state_ja = _STATE_JA.get(state, state.value)
    frozen_tag = " [凍結]" if is_frozen else ""
    if not fonts.has_japanese:
        state_ja = state.value
        frozen_tag = " [FROZEN]" if is_frozen else ""
    title = f"{side_label}  {state_ja}{frozen_tag}"
    title_col = _COL_FROZEN if is_frozen else _COL_TITLE
    _draw_text_with_outline(draw, (tx, ty), title, fonts.font(_VFY_FS_TITLE), title_col)
    ty += title_h

    label_font = fonts.font(_VFY_FS_LABEL)
    for label, value, col in lines:
        text = f"{label}: {value}" if value != "" else label
        draw_col = _COL_FROZEN if is_frozen else col
        _draw_text_with_outline(draw, (tx, ty), text, label_font, draw_col)
        ty += line_h
        if ty > OUT_H - _PANEL_BOT_MARGIN:
            break

    out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = out


def draw_raw_panel(
    frame: np.ndarray,
    fonts: _FontSet,
    panel_x: int,
    side_label: str,
    state: BoardState,
    is_frozen: bool,
    lines: list[_RawLine],
) -> None:
    """1 side の原観測量パネルを frame (BGR) に描画する (大きめフォント)。"""
    line_h = _RAW_FS_LABEL + _RAW_LINE_GAP
    title_h = _RAW_FS_TITLE + _RAW_LINE_GAP + 8
    panel_h = title_h + line_h * len(lines) + 14
    py0 = _PANEL_TOP_Y
    py1 = min(OUT_H - _PANEL_BOT_MARGIN, py0 + panel_h)
    px0 = panel_x
    px1 = panel_x + _RAW_PANEL_W

    # 半透明背景 (cv2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px1, py1), _PANEL_BG_BGR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1.0 - _PANEL_ALPHA, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px1, py1), (110, 110, 110), 1)

    # PIL でテキスト重畳
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    tx = px0 + _RAW_INDENT
    ty = py0 + 6
    state_ja = _STATE_JA.get(state, state.value)
    frozen_tag = " [凍結]" if is_frozen else ""
    if not fonts.has_japanese:
        state_ja = state.value
        frozen_tag = " [FROZEN]" if is_frozen else ""
    title = f"{side_label}  {state_ja}{frozen_tag}"
    title_col = _COL_FROZEN if is_frozen else _COL_TITLE
    _draw_text_with_outline(draw, (tx, ty), title, fonts.font(_RAW_FS_TITLE), title_col)
    ty += title_h

    label_font = fonts.font(_RAW_FS_LABEL)
    for label, value, col in lines:
        if not fonts.has_japanese:
            col = _COL_VALUE if col not in (_COL_KEY, _COL_NOTE) else col
        text = f"{label}: {value}" if value != "" else label
        draw_col = _COL_FROZEN if is_frozen else col
        _draw_text_with_outline(draw, (tx, ty), text, label_font, draw_col)
        ty += line_h
        if ty > OUT_H - _PANEL_BOT_MARGIN:
            break

    out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = out


# ============================
# パネル描画 (PIL)
# ============================


def _draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    """黒縁付きでテキストを描画する (背景パネル上でも視認性確保)。"""
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def _line_height(kind: str) -> int:
    """行種別ごとの行高さ (px) を返す。"""
    if kind == "cat":
        return _FS_CATEGORY + _LINE_GAP + 4
    if kind == "key":
        return _FS_KEY_VALUE + _LINE_GAP + 1
    return _FS_VALUE + _LINE_GAP


def draw_indicator_panel(
    frame: np.ndarray,
    fonts: _FontSet,
    panel_x: int,
    side_label: str,
    state: BoardState,
    is_frozen: bool,
    lines: list[tuple[str, str, str]],
) -> None:
    """1 side の指標パネルを frame (BGR) に描画する。

    半透明背景を cv2 で合成 → PIL で日本語テキストを重畳 → frame に書き戻す。

    Args:
        frame: 描画対象フレーム (1920x1080 BGR, in-place 変更)。
        fonts: フォントセット。
        panel_x: パネル左端 X。
        side_label: "1P" / "2P"。
        state: 現在の state machine 状態。
        is_frozen: 非 STABLE で凍結表示中なら True (見出しに「凍結」を付す)。
        lines: _build_indicator_lines の結果。
    """
    # --- パネル高さを計算 ---
    title_h = _FS_TITLE + _LINE_GAP + 6
    body_h = sum(_line_height(k) for k, _, _ in lines)
    panel_h = title_h + body_h + 12
    py0 = _PANEL_TOP_Y
    py1 = min(OUT_H - _PANEL_BOT_MARGIN, py0 + panel_h)
    px0 = panel_x
    px1 = panel_x + _PANEL_W

    # --- 半透明背景 (cv2) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px1, py1), _PANEL_BG_BGR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1.0 - _PANEL_ALPHA, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px1, py1), (110, 110, 110), 1)

    # --- PIL でテキスト重畳 ---
    # BGR(numpy) → RGB(PIL)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    tx = px0 + 8
    ty = py0 + 6
    # タイトル
    state_ja = _STATE_JA.get(state, state.value)
    frozen_tag = " [凍結]" if is_frozen else ""
    if not fonts.has_japanese:
        state_ja = state.value
        frozen_tag = " [FROZEN]" if is_frozen else ""
    title = f"{side_label}  {state_ja}{frozen_tag}"
    title_col = _COL_FROZEN if is_frozen else _COL_TITLE
    _draw_text_with_outline(
        draw, (tx, ty), title, fonts.font(_FS_TITLE), title_col,
    )
    ty += title_h

    for kind, label, value in lines:
        if not fonts.has_japanese:
            label = _ROMAJI.get(label, label)
        if kind == "cat":
            _draw_text_with_outline(
                draw, (tx, ty), label, fonts.font(_FS_CATEGORY), _COL_CATEGORY,
            )
        elif kind == "key":
            font = fonts.font(_FS_KEY_VALUE)
            col = _COL_FROZEN if is_frozen else _COL_KEY
            _draw_text_with_outline(draw, (tx + 6, ty), f"{label}: {value}", font, col)
        elif kind == "note":
            font = fonts.font(_FS_VALUE)
            _draw_text_with_outline(
                draw, (tx + 6, ty), f"{label}: {value}", font, _COL_NOTE,
            )
        else:  # "val"
            font = fonts.font(_FS_VALUE)
            col = _COL_FROZEN if is_frozen else _COL_VALUE
            _draw_text_with_outline(draw, (tx + 6, ty), f"{label}: {value}", font, col)
        ty += _line_height(kind)
        if ty > OUT_H - _PANEL_BOT_MARGIN:
            break

    # RGB(PIL) → BGR(numpy) に書き戻す
    out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = out


# ============================
# メイン処理
# ============================


def _side_lines_or_frozen(
    side: SideResult,
    pipeline: RecognitionPipeline,
    side_label: str,
    tracker: _SideTracker,
    ojama_tracker: OjamaAccountingTracker,
    t_sec: float,
    snap: OjamaAccountSnapshot,
    cache: dict[str, list[tuple[str, str, str]]],
    offboard: int,
) -> tuple[list[tuple[str, str, str]], bool]:
    """STABLE なら指標を算出し cache 更新、非 STABLE なら前回 cache を凍結返却。

    Returns:
        (表示行リスト, is_frozen)。
    """
    board = side.confirmed_board
    is_stable = (
        side.state == BoardState.STABLE
        and board is not None
        and board.count_puyos() > 0
    )
    if is_stable:
        _update_game_idx(tracker, side.score)
        elapsed = ojama_tracker._elapsed(t_sec)
        tsumo = pipeline.tsumo_count(side_label)
        is_p1 = side_label == "1P"
        net = snap.net_balance_capped if is_p1 else -snap.net_balance_capped
        forecast = snap.forecast_p1 if is_p1 else snap.forecast_p2
        lines = _build_indicator_lines(
            board, tsumo, elapsed, net, forecast, offboard, side,
        )
        cache[side_label] = lines
        return lines, False
    # 非 STABLE: 前回 STABLE の cache を凍結表示
    if side_label in cache:
        return cache[side_label], True
    return [("note", "指標", "STABLE 待ち")], True


def _side_raw_or_frozen(
    side: SideResult,
    pipeline: RecognitionPipeline,
    side_label: str,
    ojama_tracker: OjamaAccountingTracker,
    t_sec: float,
    snap: OjamaAccountSnapshot,
    cache: dict[str, list[_RawLine]],
) -> tuple[list[_RawLine], bool]:
    """raw モード: STABLE なら原観測量を算出し cache 更新、非 STABLE は凍結返却。"""
    board = side.confirmed_board
    is_stable = (
        side.state == BoardState.STABLE
        and board is not None
        and board.count_puyos() > 0
    )
    if is_stable:
        elapsed = ojama_tracker._elapsed(t_sec)
        tsumo = pipeline.tsumo_count(side_label)
        is_p1 = side_label == "1P"
        forecast = snap.forecast_p1 if is_p1 else snap.forecast_p2
        visible_ojama = ojama_tracker._count_visible_ojama(board)
        lines = _build_raw_lines(
            board, side.state, side.score, tsumo, elapsed,
            forecast, visible_ojama, side,
        )
        cache[side_label] = lines
        return lines, False
    if side_label in cache:
        return cache[side_label], True
    return [("原観測量", "STABLE 待ち", _COL_NOTE)], True


def _side_verify_or_frozen(
    side: SideResult,
    side_label: str,
    ojama_tracker: OjamaAccountingTracker,
    t_sec: float,
    snap: OjamaAccountSnapshot,
    cache: dict[str, list[_RawLine]],
) -> tuple[list[_RawLine], bool]:
    """verify モード: STABLE なら選択指標の生値を算出し cache 更新、非 STABLE は凍結返却。

    net収支の符号: 1P は snap.net_balance_capped のまま、2P は符号反転 (既存 indicator 方式踏襲)。
    """
    board = side.confirmed_board
    is_stable = (
        side.state == BoardState.STABLE
        and board is not None
        and board.count_puyos() > 0
    )
    if is_stable:
        is_p1 = side_label == "1P"
        net = snap.net_balance_capped if is_p1 else -snap.net_balance_capped
        forecast = snap.forecast_p1 if is_p1 else snap.forecast_p2
        lines = _build_verify_lines(board, net, forecast)
        cache[side_label] = lines
        return lines, False
    if side_label in cache:
        return cache[side_label], True
    return [("選択指標", "STABLE 待ち", _COL_NOTE)], True


def generate(
    video_path: Path,
    out_path: Path,
    max_sec: float = 0.0,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    dump_png_dir: Path | None = None,
    png_interval_sec: float = 5.0,
    mode: str = "indicator",
) -> int:
    """指標 v2 レビュー動画を生成する。

    Returns:
        書き出したフレーム数。
    """
    fonts = _FontSet()
    if fonts.has_japanese:
        print(f"[viz_iv2] 日本語フォント: {fonts.path}")
    else:
        print("[viz_iv2] 警告: 日本語フォント未検出 → ローマ字 fallback で描画")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video_path}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_sec > 0:
        n_frames = min(n_frames, int(max_sec * fps))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[input] {video_path} {src_w}x{src_h} fps={fps:.1f} frames={n_frames}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))

    # visualize_recognition / collect_indicators_v2 と同じ load_default 経路。
    # 自動 HSV のみ (per-video 手調整 inject なし)。
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    _vid_match = __import__("re").search(r"(v\d+|video_\d+)", video_path.name)
    if _vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(_vid_match.group(1))
    if hasattr(pipeline._reader, "set_resolution_aware_s_min"):
        pipeline._reader.set_resolution_aware_s_min(src_h)

    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    tracker_p1 = _SideTracker()
    tracker_p2 = _SideTracker()

    is_raw = mode == "raw"
    is_verify = mode == "verify"
    # 凍結表示用の最新 STABLE 結果キャッシュ (indicator/raw/verify で行型が異なる)
    line_cache: dict[str, list] = {}
    last_p1_state = BoardState.MENU
    last_p2_state = BoardState.MENU
    last_p1_board: Board | None = None
    last_p2_board: Board | None = None
    last_p1_score: int | None = None
    last_p2_score: int | None = None
    if is_raw:
        _init_line: list = [("原観測量", "STABLE 待ち", _COL_NOTE)]
    elif is_verify:
        _init_line = [("選択指標", "STABLE 待ち", _COL_NOTE)]
    else:
        _init_line = [("note", "指標", "STABLE 待ち")]
    last_p1_lines: list = list(_init_line)
    last_p2_lines: list = list(_init_line)
    last_p1_frozen = True
    last_p2_frozen = True
    last_snap: OjamaAccountSnapshot | None = None

    if dump_png_dir is not None:
        dump_png_dir.mkdir(parents=True, exist_ok=True)
    png_saved = 0
    next_png_t = 0.0

    sample_interval_frames = max(1, int(round(sample_interval * fps)))
    written = 0

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(
                frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        if fi % sample_interval_frames == 0:
            result = pipeline.update(fi, t_sec, frame)
            last_p1_state = result.p1.state
            last_p2_state = result.p2.state
            if (result.p1.state == BoardState.STABLE
                    and result.p1.confirmed_board is not None):
                last_p1_board = result.p1.confirmed_board
            if (result.p2.state == BoardState.STABLE
                    and result.p2.confirmed_board is not None):
                last_p2_board = result.p2.confirmed_board
            if result.p1.score is not None:
                last_p1_score = result.p1.score
            if result.p2.score is not None:
                last_p2_score = result.p2.score
            # お邪魔会計駆動 (collect_indicators_v2._drive_ojama 流用)
            snap = _drive_ojama(
                ojama_tracker, result.p1, result.p2,
                prev_state_p1, prev_state_p2, t_sec,
            )
            prev_state_p1 = result.p1.state
            prev_state_p2 = result.p2.state
            last_snap = snap
            offboard_p1 = snap.offboard_p1
            offboard_p2 = snap.offboard_p2
            if is_raw:
                # raw モード: 原観測量を算出 (STABLE) / 凍結 (非 STABLE)
                last_p1_lines, last_p1_frozen = _side_raw_or_frozen(
                    result.p1, pipeline, "1P", ojama_tracker,
                    t_sec, snap, line_cache,
                )
                last_p2_lines, last_p2_frozen = _side_raw_or_frozen(
                    result.p2, pipeline, "2P", ojama_tracker,
                    t_sec, snap, line_cache,
                )
            elif is_verify:
                # verify モード: 選択指標の生値のみ (STABLE) / 凍結 (非 STABLE)
                last_p1_lines, last_p1_frozen = _side_verify_or_frozen(
                    result.p1, "1P", ojama_tracker, t_sec, snap, line_cache,
                )
                last_p2_lines, last_p2_frozen = _side_verify_or_frozen(
                    result.p2, "2P", ojama_tracker, t_sec, snap, line_cache,
                )
            else:
                # 指標算出 (STABLE) / 凍結 (非 STABLE)
                last_p1_lines, last_p1_frozen = _side_lines_or_frozen(
                    result.p1, pipeline, "1P", tracker_p1, ojama_tracker,
                    t_sec, snap, line_cache, offboard_p1,
                )
                last_p2_lines, last_p2_frozen = _side_lines_or_frozen(
                    result.p2, pipeline, "2P", tracker_p2, ojama_tracker,
                    t_sec, snap, line_cache, offboard_p2,
                )

        # --- 認識 overlay (盤面ぷよ色 + state 枠) ---
        draw_cell_overlay(frame, last_p1_board, P1_ROI_X, P1_ROI_Y)
        draw_cell_overlay(frame, last_p2_board, P2_ROI_X, P2_ROI_Y)
        draw_state_label(
            frame, last_p1_state, P1_ROI_X, P1_ROI_Y,
            score=last_p1_score or 0, label_prefix="1P:",
        )
        draw_state_label(
            frame, last_p2_state, P2_ROI_X, P2_ROI_Y,
            score=last_p2_score or 0, label_prefix="2P:",
        )
        draw_global_info(
            frame, fi, t_sec, last_p1_state, last_p2_state,
            p1_score=last_p1_score, p2_score=last_p2_score,
        )
        # --- パネル (日本語, PIL) ---
        if is_raw:
            raw_p2_x = OUT_W - _RAW_PANEL_W - 4
            draw_raw_panel(
                frame, fonts, _P1_PANEL_X, "1P", last_p1_state,
                last_p1_frozen, last_p1_lines,
            )
            draw_raw_panel(
                frame, fonts, raw_p2_x, "2P", last_p2_state,
                last_p2_frozen, last_p2_lines,
            )
        elif is_verify:
            vfy_p2_x = OUT_W - _VFY_PANEL_W - 4
            draw_verify_panel(
                frame, fonts, _P1_PANEL_X, "1P", last_p1_state,
                last_p1_frozen, last_p1_lines,
            )
            draw_verify_panel(
                frame, fonts, vfy_p2_x, "2P", last_p2_state,
                last_p2_frozen, last_p2_lines,
            )
        else:
            draw_indicator_panel(
                frame, fonts, _P1_PANEL_X, "1P", last_p1_state,
                last_p1_frozen, last_p1_lines,
            )
            draw_indicator_panel(
                frame, fonts, _P2_PANEL_X, "2P", last_p2_state,
                last_p2_frozen, last_p2_lines,
            )

        writer.write(frame)
        written += 1

        # smoke PNG 保存 (一定秒間隔で数枚)
        if dump_png_dir is not None and t_sec >= next_png_t:
            png_path = dump_png_dir / f"{video_path.stem}_t{t_sec:05.1f}s.png"
            cv2.imwrite(str(png_path), frame)
            png_saved += 1
            next_png_t += png_interval_sec

        if fi % 100 == 0:
            print(
                f"  [progress] {fi}/{n_frames} "
                f"({fi*100/max(n_frames,1):.1f}%) "
                f"1P={last_p1_state.value} 2P={last_p2_state.value}"
            )

    cap.release()
    writer.release()
    if last_snap is not None:
        print(
            f"[acct_final] net_capped={last_snap.net_balance_capped:+d} "
            f"forecast 1P={last_snap.forecast_p1} 2P={last_snap.forecast_p2}"
        )
    print(f"[done] {out_path}  ({written} frames written)")
    if dump_png_dir is not None:
        print(f"[smoke] PNG {png_saved} 枚保存 → {dump_png_dir}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="指標 v2 日本語レビュー動画生成")
    parser.add_argument("--video", type=Path, required=True, help="入力動画")
    parser.add_argument("--output", type=Path, required=True, help="出力 MP4 パス")
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="処理する最大秒数 (0 = 全長)",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL,
        help="認識処理する frame 間隔 (秒)。未サンプル frame は最後の認識結果を保持。",
    )
    parser.add_argument(
        "--dump-png-dir", type=Path, default=None,
        help="smoke 用 PNG を保存するディレクトリ (省略時は保存しない)。",
    )
    parser.add_argument(
        "--png-interval", type=float, default=5.0,
        help="PNG 保存間隔秒 (--dump-png-dir 指定時のみ有効)。",
    )
    parser.add_argument(
        "--mode", choices=("indicator", "raw", "verify"), default="indicator",
        help=(
            "indicator=計算指標21項目(既定) / "
            "raw=原観測量(生データ)のみ / "
            "verify=選択10指標の生値(目視照合用、大きめ表示)。"
        ),
    )
    args = parser.parse_args()
    n = generate(
        args.video, args.output, args.max_sec, args.sample_interval,
        args.dump_png_dir, args.png_interval, args.mode,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
