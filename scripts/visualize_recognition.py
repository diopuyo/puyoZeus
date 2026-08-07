"""認識+物理推論結果を盤面に重畳する可視化動画を生成する.

各セルに認識色 (赤/青/緑/黄/紫/お/空/?) を文字で描画し、
盤面外枠に state machine の現在状態 (STABLE/CHAIN/TSUMO_FALL/...) を描画する.

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \\
        --video data/evaluation_videos/v28_clip60s.mp4 \\
        --output data/evaluation_videos/v28_recognition_viz.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker, OjamaAccountSnapshot  # noqa: E402
from src.probabilistic_board import ProbabilisticBoard  # noqa: E402
from src.scoring import ojama_count_to_icons  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 描画定数
# ============================
# 盤面 ROI (calibration_video01.json より)
P1_ROI_X = 282
P1_ROI_Y = 160
P2_ROI_X = 1258
P2_ROI_Y = 160
ROI_W = 384
ROI_H = 720
N_VISIBLE_ROWS = 12
CELL_W = ROI_W // BOARD_COLS  # 64 px
CELL_H = ROI_H // N_VISIBLE_ROWS  # 60 px

# 色記号 (コンパクト表示)
COLOR_SYMBOLS = {
    COLOR_EMPTY: "",
    COLOR_RED: "R",
    COLOR_BLUE: "B",
    COLOR_GREEN: "G",
    COLOR_YELLOW: "Y",
    COLOR_PURPLE: "P",
    COLOR_OJAMA: "O",
    COLOR_UNKNOWN: "?",
}
# BGR 色 (cv2)
COLOR_BGR = {
    COLOR_EMPTY: (60, 60, 60),
    COLOR_RED: (50, 50, 240),
    COLOR_BLUE: (240, 100, 50),
    COLOR_GREEN: (50, 220, 50),
    COLOR_YELLOW: (50, 220, 240),
    COLOR_PURPLE: (200, 50, 200),
    COLOR_OJAMA: (180, 180, 180),
    COLOR_UNKNOWN: (255, 255, 255),
}

# State machine state ごとの枠色
STATE_COLOR = {
    BoardState.STABLE: (0, 255, 0),              # green = OK
    BoardState.TSUMO_FALL: (0, 200, 255),        # orange
    BoardState.CHAIN: (200, 100, 255),           # pink/purple
    BoardState.OJAMA_FALL: (255, 200, 0),        # cyan
    BoardState.MENU: (128, 128, 128),            # gray
    BoardState.EFFECT: (255, 0, 255),            # magenta (全消し等)
    BoardState.GRAVITY_SETTLE: (0, 165, 255),    # orange-yellow (重力 settle)
}

# 描画フォント
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_CELL = 0.7
FONT_SCALE_STATE = 0.9
FONT_THICKNESS = 2

# サンプリング間隔 (秒)
DEFAULT_SAMPLE_INTERVAL = 0.033  # 30 fps 認識 (= cycle 71p 2026-05-13 ユーザー要望)


def draw_cell_overlay(
    frame: np.ndarray, board: Board, roi_x: int, roi_y: int,
) -> None:
    """盤面 1 つに対し、各 cell の色 symbol を重畳する.

    可視 12 行のみ描画 (隠し段 row 0 は省略)。
    文字色は常に白、黒太縁で puyo 背景と同色化を回避。
    """
    if board is None:
        return
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            symbol = COLOR_SYMBOLS.get(color, "?")
            if not symbol:
                continue  # EMPTY は描画しない
            # cell 中心座標 (visible row index = row - HIDDEN_ROWS)
            display_row = row - HIDDEN_ROWS
            cx = roi_x + col * CELL_W + CELL_W // 2
            cy = roi_y + display_row * CELL_H + CELL_H // 2
            # 文字サイズ調整
            (tw, th), _ = cv2.getTextSize(
                symbol, FONT, FONT_SCALE_CELL, FONT_THICKNESS,
            )
            tx = cx - tw // 2
            ty = cy + th // 2
            # 黒太縁 (puyo 背景と同色化を回避するため)
            cv2.putText(
                frame, symbol, (tx, ty), FONT,
                FONT_SCALE_CELL, (0, 0, 0), FONT_THICKNESS + 4, cv2.LINE_AA,
            )
            # 白文字 (常に視認性確保)
            cv2.putText(
                frame, symbol, (tx, ty), FONT,
                FONT_SCALE_CELL, (255, 255, 255),
                FONT_THICKNESS, cv2.LINE_AA,
            )


def should_draw_cell_overlay(
    state: BoardState,
    overlay_stable_only: bool,
    overlay_show_states: frozenset[BoardState] | None,
) -> bool:
    """セル文字オーバーレイを描画すべきか判定する (2026-08-07 追加).

    --overlay-show-states と --overlay-stable-only は argparse 側で併用禁止のため
    ここでは overlay_show_states を優先判定するだけでよい。両方未指定なら常時描画
    (既定・後方互換)。
    """
    if overlay_show_states is not None:
        return state in overlay_show_states
    if overlay_stable_only:
        return state == BoardState.STABLE
    return True


def draw_state_label(
    frame: np.ndarray, state: BoardState, roi_x: int, roi_y: int,
    score: int = 0, label_prefix: str = "",
) -> None:
    """ROI 上方に state ラベルを描画する.

    隠し段帯 (HIDDEN_BAND_HEIGHT) の上方に配置し帯と被らないようオフセットを取る。
    """
    color = STATE_COLOR.get(state, (255, 255, 255))
    text = f"{label_prefix}{state.value}"
    if score > 0:
        text += f" score={score}"
    # 隠し段帯より上にラベルを配置
    label_y = roi_y - STATE_LABEL_OFFSET_Y
    # 影
    cv2.putText(
        frame, text, (roi_x + 4, label_y), FONT,
        FONT_SCALE_STATE, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (roi_x + 3, label_y - 1), FONT,
        FONT_SCALE_STATE, color, FONT_THICKNESS, cv2.LINE_AA,
    )
    # ROI 枠
    cv2.rectangle(
        frame, (roi_x, roi_y), (roi_x + ROI_W, roi_y + ROI_H),
        color, 2,
    )


def draw_next_overlay(
    frame: np.ndarray, next_pair: tuple[int, int] | None,
    dnext_pair: tuple[int, int] | None,
    roi_x: int, roi_y: int, label_prefix: str = "",
) -> None:
    """ネクスト / ダブルネクストの色を ROI 下部に描画する.

    next_pair = (上ぷよ色, 下ぷよ色)、dnext_pair = 同上。
    EMPTY (0) は "-" で表示し、None 全体は "next:?" で表示する。

    Args:
        frame: 描画対象フレーム。
        next_pair: (color_top, color_bot) または None。
        dnext_pair: (color_top, color_bot) または None。
        roi_x: 盤面 ROI 左端 X 座標。
        roi_y: 盤面 ROI 上端 Y 座標。
        label_prefix: 先頭に付けるラベル ("1P:" 等)。
    """
    def _pair_str(pair: tuple[int, int] | None) -> str:
        """ペアを記号 2 文字に変換する."""
        if pair is None:
            return "??"
        t, b = pair
        return f"{COLOR_SYMBOLS.get(t, '?') or '-'}{COLOR_SYMBOLS.get(b, '?') or '-'}"

    text = f"N:{_pair_str(next_pair)} D:{_pair_str(dnext_pair)}"
    # ROI 下端より少し下に描画
    tx = roi_x + 2
    ty = roi_y + ROI_H + 18
    cv2.putText(frame, text, (tx + 1, ty + 1), FONT,
                0.65, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), FONT,
                0.65, (200, 255, 200), FONT_THICKNESS, cv2.LINE_AA)


def draw_ojama_overlay(
    frame: np.ndarray, ojama_sent: int, recent_event_sec: float,
    roi_x: int, roi_y: int,
) -> None:
    """score OCR 差分由来の送出お邪魔数を ROI 下部 (next の下) に描画する.

    後方互換のため残す。新実装は draw_ojama_accounting_overlay を使うこと。

    Args:
        frame: 描画対象フレーム。
        ojama_sent: 最新の連鎖イベントで送出したお邪魔数 (score 差分由来)。
        recent_event_sec: 最後のイベントから経過した秒数 (表示フェードアウト用)。
                          負値 = イベント未発生。
        roi_x: 盤面 ROI 左端 X 座標。
        roi_y: 盤面 ROI 上端 Y 座標。
    """
    # 表示フェードアウト: 最後のイベントから OJAMA_DISPLAY_FADE_SEC 秒以上経過したら薄く
    OJAMA_DISPLAY_FADE_SEC: float = 3.0
    if ojama_sent <= 0 or recent_event_sec < 0:
        # イベント未発生または送出なし: 薄グレーで "OJ_sent:--" と表示
        count_label = "OJ_sent:--"
        color_bgr: tuple[int, int, int] = (80, 80, 80)
    else:
        count_label = f"OJ_sent:{ojama_sent}"
        # フェードアウト: 経過時間に応じて輝度を下げる
        fade = max(0.0, 1.0 - recent_event_sec / max(OJAMA_DISPLAY_FADE_SEC, 1e-9))
        r = int(80 + 175 * fade)
        g = int(80)
        b = int(80)
        color_bgr = (b, g, r)
    tx = roi_x + 2
    ty = roi_y + ROI_H + 36
    cv2.putText(frame, count_label, (tx + 1, ty + 1), FONT,
                0.60, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(frame, count_label, (tx, ty), FONT,
                0.60, color_bgr, FONT_THICKNESS, cv2.LINE_AA)


# 会計 overlay の縦オフセット定数 (ROI 下端からの px)
_ACCT_LINE1_OFFSET_Y: int = 36   # pending 行 (next の下)
_ACCT_LINE2_OFFSET_Y: int = 54   # drop/net 行
_ACCT_FONT_SCALE: float = 0.55


# 会計 overlay の3行目 Y オフセット (相殺表示用)
_ACCT_LINE3_OFFSET_Y: int = 72


def draw_ojama_accounting_overlay(
    frame: np.ndarray,
    snap: OjamaAccountSnapshot | None,
    side: str,
    roi_x: int,
    roi_y: int,
) -> None:
    """OjamaAccountSnapshot の会計値を ROI 下部 3 行に描画する.

    表示内容 (3 行):
        行1: pend:N  net(相殺後収支):±K  c:0.xx
        行2: drop:x  off:Y (累積相殺) [AC]
        行3: off-board:Z (画面外あふれ推定、空フィールド近似)

    side="1P" なら snap.pending_p1 / total_dropped_to_p1 等を使う。
    side="2P" なら snap.pending_p2 / total_dropped_to_p2 等を使う。

    net_balance_capped の符号: 正→1P有利(青), 負→2P有利(赤)。
    net は相殺後収支であることをラベルで明示する。

    Args:
        frame: 描画対象フレーム。
        snap: OjamaAccountSnapshot (None 時はグレー "--" 表示)。
        side: "1P" または "2P"。
        roi_x: 盤面 ROI 左端 X 座標。
        roi_y: 盤面 ROI 上端 Y 座標。
    """
    tx = roi_x + 2
    ty1 = roi_y + ROI_H + _ACCT_LINE1_OFFSET_Y
    ty2 = roi_y + ROI_H + _ACCT_LINE2_OFFSET_Y
    ty3 = roi_y + ROI_H + _ACCT_LINE3_OFFSET_Y

    if snap is None:
        # 未初期化: グレー "--" 表示
        _put_shadow(frame, "acct:--", tx, ty1, (70, 70, 70), _ACCT_FONT_SCALE)
        return

    # 1P/2P ごとに適切なフィールドを選択 (有界 capped 値を overlay 表示に使用)
    if side == "1P":
        pending_recv = snap.pending_p1_capped    # 自分が受ける pending (2P→1P、有界)
        dropped = snap.total_dropped_to_p1
        ac_flag = snap.all_clear_pending_p1
        # 相殺: 1P が相殺した分 (= 自分に向かう pending を自分の連鎖で消した量)
        offset_total = snap.total_offset_by_p1
        offboard = snap.offboard_p1              # 画面外あふれ推定 (空フィールド近似)
    else:
        pending_recv = snap.pending_p2_capped    # 自分が受ける pending (1P→2P、有界)
        dropped = snap.total_dropped_to_p2
        ac_flag = snap.all_clear_pending_p2
        offset_total = snap.total_offset_by_p2
        offboard = snap.offboard_p2

    net = snap.net_balance_capped         # 正→1P有利 (有界 -72..+72、相殺後収支)
    conf = snap.confidence

    # net の色: 正なら青(1P有利)、負なら赤(2P有利)、0 なら白
    if net > 0:
        net_bgr: tuple[int, int, int] = (220, 120, 40)   # 青寄り (1P有利)
    elif net < 0:
        net_bgr = (60, 60, 220)                           # 赤寄り (2P有利)
    else:
        net_bgr = (180, 180, 180)                         # 白 (均衡)

    # pending の色: 受け取りが多いほど赤く
    if pending_recv >= 6:
        pend_bgr: tuple[int, int, int] = (40, 40, 220)   # 赤 (危険)
    elif pending_recv >= 3:
        pend_bgr = (40, 160, 240)                         # オレンジ (注意)
    else:
        pend_bgr = (180, 220, 180)                        # 緑 (安全)

    # 行1: pending / net(相殺後収支) / conf
    net_sign = "+" if net >= 0 else ""

    # 行1 描画 (pending 部分のみ色付け、net はラベル付き)
    _put_shadow(frame, f"pend:{pending_recv}", tx, ty1, pend_bgr, _ACCT_FONT_SCALE)
    (w1, _), _ = cv2.getTextSize(f"pend:{pending_recv}", FONT, _ACCT_FONT_SCALE, FONT_THICKNESS)
    _put_shadow(frame, f"  net(off後):{net_sign}{net}", tx + w1, ty1, net_bgr, _ACCT_FONT_SCALE)
    (w2, _), _ = cv2.getTextSize(
        f"  net(off後):{net_sign}{net}", FONT, _ACCT_FONT_SCALE, FONT_THICKNESS,
    )
    _put_shadow(frame, f"  c:{conf:.2f}", tx + w1 + w2, ty1, (160, 160, 160), _ACCT_FONT_SCALE)

    # 行2: drop / 累積相殺(off) / AC
    ac_tag = " [AC]" if ac_flag else ""
    drop_bgr: tuple[int, int, int] = (200, 200, 200)
    off_bgr: tuple[int, int, int] = (160, 200, 255)   # 薄い黄色 (相殺は好材料)
    _put_shadow(frame, f"drop:{dropped}", tx, ty2, drop_bgr, _ACCT_FONT_SCALE)
    (wd, _), _ = cv2.getTextSize(f"drop:{dropped}", FONT, _ACCT_FONT_SCALE, FONT_THICKNESS)
    _put_shadow(frame, f"  off:{offset_total}{ac_tag}", tx + wd, ty2, off_bgr, _ACCT_FONT_SCALE)

    # 行3: 画面外あふれ推定 (off-board、空フィールド近似)
    if offboard > 0:
        ob_bgr: tuple[int, int, int] = (0, 80, 255)   # 橙色 (危険: 画面外あふれ)
        _put_shadow(frame, f"OB+{offboard}(approx)", tx, ty3, ob_bgr, _ACCT_FONT_SCALE)
    else:
        _put_shadow(frame, "OB:0", tx, ty3, (100, 100, 100), _ACCT_FONT_SCALE)


def _put_shadow(
    frame: np.ndarray,
    text: str,
    tx: int,
    ty: int,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    """黒縁付きテキストを描画するヘルパー関数 (影→本文の2パス)."""
    cv2.putText(frame, text, (tx + 1, ty + 1), FONT,
                scale, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), FONT,
                scale, color, FONT_THICKNESS, cv2.LINE_AA)


# ============================
# 予告お邪魔 直感UI (2026-06-10 刷新)
# ============================
#
# 目的: 算出した予告お邪魔個数を「人が一目で量を把握でき、画面の予告アイコン列と
#       見比べて検証できる」UI にする。小さいテキストではなく大きな数字 + ゲームと
#       同じお邪魔単位アイコン分解 (岩/連/小 等) + 左右優勢バーで表示する。

# 予告パネルの配置オフセット (ROI 下端からの px)
# 2026-06-10 大幅拡大: 旧デバッグ会計テキストを抑止し、盤面下の余白
# (ROI 下端 Y=880 〜 画面下端 Y=1080、約 200px) を専有して一目で読める大きさに。
_FC_PANEL_TOP_OFFSET_Y: int = 10       # パネル上端 (ROI 下端のすぐ下)
_FC_PANEL_HEIGHT: int = 188            # パネル高さ (拡大)
_FC_BIG_NUM_SCALE: float = 3.6         # 大きな予告数字のフォントスケール (拡大)
_FC_LABEL_SCALE: float = 1.1           # "OJAMA" ラベルのフォントスケール
_FC_ICON_SCALE: float = 1.05           # アイコン分解ラベルのフォントスケール (拡大)
# パネル内レイアウト (px、パネル左上 px/py 基準)
_FC_LABEL_DX: int = 14                 # "OJAMA" ラベル X オフセット
_FC_LABEL_DY: int = 42                 # "OJAMA" ラベル Y オフセット
_FC_NUM_DX: int = 12                   # 大きな数字 X オフセット
_FC_NUM_DY: int = 138                  # 大きな数字 ベースライン Y オフセット
_FC_ICON_DX: int = 160                 # アイコン分解 X オフセット (数字の右)
_FC_ICON_DY: int = 92                  # アイコン分解 ベースライン Y オフセット
_FC_OFF_DY: int = 138                  # 画面外あふれ ベースライン Y オフセット

# アイコン名 → ASCII 短縮ラベル (ゲーム表示と対応: 王冠/月/星/岩/連(line)/小)
# 注意: cv2 の FONT_HERSHEY は日本語グリフ非対応 (??? 化) のため ASCII 記号を使う。
#   CR=crown(王冠/720) MN=moon(月/360) ST=star(星/180)
#   RK=rock(岩/30) LN=line(連結/6) sm=small(小/1)
_OJAMA_ICON_LABELS: dict[str, str] = {
    "crown": "CR",   # 720 個 (王冠)
    "moon": "MN",    # 360 個 (月)
    "star": "ST",    # 180 個 (星)
    "rock": "RK",    # 30 個 (岩)
    "large": "LN",   # 6 個 (連結 line)
    "small": "sm",   # 1 個 (小)
}


def _format_ojama_icons(count: int) -> str:
    """forecast 個数をゲームと同じお邪魔単位に貪欲分解した文字列を返す.

    scoring.ojama_count_to_icons (2026-04-27 ユーザ確定仕様) を流用。
    例: 38 → "RKx1 LNx1 smx2" (= 岩x1 連x1 小x2)。0 のときは "-"。
    cv2 が日本語非対応のため ASCII ラベル (RK/LN/sm 等) を使う。

    Args:
        count: 予告お邪魔個数 (>= 0)。

    Returns:
        "RKx1 LNx1 smx2" 形式の文字列 (アイコン無しなら "-")。
    """
    icons = ojama_count_to_icons(int(count))
    if not icons:
        return "-"
    parts = [f"{_OJAMA_ICON_LABELS.get(name, name)}x{n}" for name, n in icons]
    return " ".join(parts)


def draw_ojama_forecast_panel(
    frame: np.ndarray,
    snap: OjamaAccountSnapshot | None,
    side: str,
    roi_x: int,
    roi_y: int,
) -> None:
    """予告お邪魔個数を大きな数字 + 単位アイコン分解で盤面下に描画する.

    構成 (盤面 ROI 下の専用パネル):
        - 半透明黒背景パネル (視認性確保)
        - 左: 大きな数字「予告 N」(危険度で色変化: 多いほど赤)
        - 下段: ゲーム同単位アイコン分解「岩x1 連x1 小x2」(画面の予告列と直接照合)
        - 画面外あふれ (offboard) があれば「画面外 +M」を赤で追記

    Args:
        frame: 描画対象フレーム (1920x1080 BGR)。
        snap: OjamaAccountSnapshot (None 時はグレー "予告 --")。
        side: "1P" または "2P"。
        roi_x: 盤面 ROI 左端 X 座標。
        roi_y: 盤面 ROI 上端 Y 座標。
    """
    px = roi_x
    py = roi_y + ROI_H + _FC_PANEL_TOP_OFFSET_Y
    # 半透明背景パネル
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + ROI_W, py + _FC_PANEL_HEIGHT),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (px, py), (px + ROI_W, py + _FC_PANEL_HEIGHT),
                  (110, 110, 110), 1)

    if snap is None:
        _put_shadow(frame, "OJAMA --", px + _FC_LABEL_DX, py + _FC_NUM_DY,
                    (120, 120, 120), _FC_BIG_NUM_SCALE)
        return

    forecast = snap.forecast_p1 if side == "1P" else snap.forecast_p2
    offboard = snap.offboard_p1 if side == "1P" else snap.offboard_p2

    # 危険度で大きな数字の色を変える (多いほど赤): 0=緑, 1-5=黄, 6-29=橙, 30+=赤
    if forecast >= 30:
        num_bgr: tuple[int, int, int] = (40, 40, 240)    # 赤 (致命)
    elif forecast >= 6:
        num_bgr = (40, 130, 250)                          # 橙 (危険)
    elif forecast >= 1:
        num_bgr = (40, 210, 240)                          # 黄 (注意)
    else:
        num_bgr = (150, 230, 150)                         # 緑 (安全)

    # 大きな数字「予告(OJAMA) N」(ラベルは小さめ、数字は大きく)
    _put_shadow(frame, f"OJAMA {side}", px + _FC_LABEL_DX, py + _FC_LABEL_DY,
                (210, 210, 210), _FC_LABEL_SCALE)
    _put_shadow(frame, str(int(forecast)), px + _FC_NUM_DX, py + _FC_NUM_DY,
                num_bgr, _FC_BIG_NUM_SCALE)

    # アイコン分解 (ゲーム表示と直接照合できるよう右側に)
    icon_text = _format_ojama_icons(forecast)
    _put_shadow(frame, icon_text, px + _FC_ICON_DX, py + _FC_ICON_DY,
                (230, 230, 255), _FC_ICON_SCALE)

    # 画面外あふれ
    if offboard > 0:
        _put_shadow(frame, f"OFF-FIELD +{offboard}", px + _FC_ICON_DX,
                    py + _FC_OFF_DY, (40, 80, 255), _FC_ICON_SCALE)


# 中央優勢バーの配置定数 (1920x1080)
# 2026-06-10 大幅拡大: 左右盤面の間 (X=666〜1258 の中央余白) に大きく配置。
_ADV_BAR_CX: int = 960          # 画面中央 X
_ADV_BAR_CY: int = 970          # バー中心 Y (盤面下部の余白帯)
_ADV_BAR_HALF_W: int = 290      # バー半幅 (px、拡大)
_ADV_BAR_H: int = 64            # バー高さ (拡大)
_ADV_BAR_FULL_NET: int = 30     # この net 差でバー満杯 (= 岩1個分)
_ADV_BAR_LABEL_SCALE: float = 1.6   # 優勢ラベルのフォントスケール (拡大)
_ADV_BAR_LABEL_GAP: int = 16        # ラベルをバー上端から離す px


def draw_ojama_advantage_bar(
    frame: np.ndarray,
    snap: OjamaAccountSnapshot | None,
) -> None:
    """画面中央下部に「どちらが何個多く送っているか」を左右バーで描画する.

    net_balance_capped > 0 = 1P有利 (相手2Pへ多く送出) → 青を左方向に伸ばす。
    net < 0 = 2P有利 (相手1Pへ多く送出) → 赤を右方向に伸ばす。
    中央に「1P有利 +N」等のラベルを大きく表示する。

    Args:
        frame: 描画対象フレーム。
        snap: OjamaAccountSnapshot (None 時は描画しない)。
    """
    if snap is None:
        return
    net = snap.net_balance_capped  # 正=1P有利
    cx, cy = _ADV_BAR_CX, _ADV_BAR_CY
    hw, h = _ADV_BAR_HALF_W, _ADV_BAR_H
    # バー枠 (グレー背景)
    cv2.rectangle(frame, (cx - hw, cy - h // 2), (cx + hw, cy + h // 2),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (cx - hw, cy - h // 2), (cx + hw, cy + h // 2),
                  (130, 130, 130), 1)
    cv2.line(frame, (cx, cy - h // 2), (cx, cy + h // 2), (200, 200, 200), 2)
    # 塗り幅 (net を正規化)
    ratio = min(1.0, abs(net) / max(_ADV_BAR_FULL_NET, 1))
    fill = int(hw * ratio)
    if net > 0:  # 1P有利 → 左半分を青で
        cv2.rectangle(frame, (cx - fill, cy - h // 2), (cx, cy + h // 2),
                      (230, 140, 40), -1)
        label, lab_bgr = f"1P LEAD +{net}", (240, 170, 80)
    elif net < 0:  # 2P有利 → 右半分を赤で
        cv2.rectangle(frame, (cx, cy - h // 2), (cx + fill, cy + h // 2),
                      (60, 60, 230), -1)
        label, lab_bgr = f"2P LEAD +{abs(net)}", (90, 90, 240)
    else:
        label, lab_bgr = "EVEN", (200, 200, 200)
    # ラベルをバー上に大きく (中央寄せ)
    (tw, _), _ = cv2.getTextSize(label, FONT, _ADV_BAR_LABEL_SCALE, FONT_THICKNESS)
    _put_shadow(frame, label, cx - tw // 2, cy - h // 2 - _ADV_BAR_LABEL_GAP,
                lab_bgr, _ADV_BAR_LABEL_SCALE)


# 隠し段 overlay: 確率閾値 (この値未満のセルは描画しない)
HIDDEN_ROW_MIN_PROB: float = 0.10
# 隠し段 overlay: 確率に応じた文字サイズ (最小・最大)
HIDDEN_ROW_FONT_SCALE_MIN: float = 0.50
HIDDEN_ROW_FONT_SCALE_MAX: float = 0.75
# 隠し段帯の高さ (ROI 上端より上の専用領域)
HIDDEN_BAND_HEIGHT: int = 52  # px; ROI_Y=160 で上方余白 160px 内に収まる

# 状態ラベルは ROI 上端 -HIDDEN_BAND_HEIGHT-18 に上げる (帯と被らないため)
STATE_LABEL_OFFSET_Y: int = HIDDEN_BAND_HEIGHT + 18


def draw_hidden_row_overlay(
    frame: np.ndarray,
    prob_board: ProbabilisticBoard | None,
    roi_x: int, roi_y: int,
    offboard_ojama: int = 0,
) -> None:
    """隠し段 (row 0) の各セルの色別確率を ROI 上端より上の専用帯に描画する.

    改善点 (2026-06-09):
    - state ラベルと被らないよう専用帯 (HIDDEN_BAND_HEIGHT px) を確保
    - 有効セルがある間は帯を半透明で強調
    - offboard_ojama > 0 のとき O個数を帯右端に赤で追記
    - O と通常ぷよを区別しやすいフォントサイズに拡大

    prob_board が None の場合も offboard_ojama があれば O個数は表示する。

    Args:
        frame: 描画対象フレーム (1920x1080 BGR)。
        prob_board: ProbabilisticBoard (SideResult.prob_board)。None 可。
        roi_x: 盤面 ROI 左端 X 座標。
        roi_y: 盤面 ROI 上端 Y 座標 (隠し段はこの上に描画)。
        offboard_ojama: 画面外 (隠し段以上) に積まれた推定 O 個数。
    """
    # 帯の Y 範囲: roi_y-HIDDEN_BAND_HEIGHT ~ roi_y
    band_y1 = roi_y - HIDDEN_BAND_HEIGHT
    band_y2 = roi_y
    has_content = False

    # prob_board から非 EMPTY セルの情報を収集
    cell_infos: list[tuple[int, int, int, float]] = []  # (col, cx, color, prob)
    if prob_board is not None:
        for col in range(BOARD_COLS):
            cell = prob_board.cell(0, col)
            color, prob = cell.most_likely()
            if color == COLOR_EMPTY or prob < HIDDEN_ROW_MIN_PROB:
                continue
            cx = roi_x + col * CELL_W + CELL_W // 2
            cell_infos.append((col, cx, color, prob))
        has_content = len(cell_infos) > 0

    # offboard_ojama があれば必ず帯を表示
    if offboard_ojama > 0:
        has_content = True

    # 帯を半透明で描画 (有効セルまたは O がある場合のみ)
    if has_content and band_y1 >= 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_x, band_y1), (roi_x + ROI_W, band_y2),
                      (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        # 帯の枠線
        cv2.rectangle(frame, (roi_x, band_y1), (roi_x + ROI_W, band_y2),
                      (120, 120, 120), 1)

    # 各セルのぷよ確率表示
    for _col, cx, color, prob in cell_infos:
        symbol = COLOR_SYMBOLS.get(color, "?")
        if not symbol:
            continue
        t = (prob - HIDDEN_ROW_MIN_PROB) / max(1.0 - HIDDEN_ROW_MIN_PROB, 1e-9)
        t = max(0.0, min(1.0, t))
        fscale = HIDDEN_ROW_FONT_SCALE_MIN + t * (
            HIDDEN_ROW_FONT_SCALE_MAX - HIDDEN_ROW_FONT_SCALE_MIN
        )
        pct_text = f"{symbol}{int(prob * 100)}%"
        bgr = COLOR_BGR.get(color, (200, 200, 200))
        (tw, th), _ = cv2.getTextSize(pct_text, FONT, fscale, FONT_THICKNESS)
        tx = cx - tw // 2
        ty = band_y1 + (HIDDEN_BAND_HEIGHT + th) // 2
        cv2.putText(frame, pct_text, (tx, ty), FONT,
                    fscale, (0, 0, 0), FONT_THICKNESS + 3, cv2.LINE_AA)
        cv2.putText(frame, pct_text, (tx, ty), FONT,
                    fscale, bgr, FONT_THICKNESS, cv2.LINE_AA)

    # 画面外お邪魔 (O) 個数: 帯の右端に赤字で描画
    if offboard_ojama > 0:
        oj_text = f"O+{offboard_ojama}"
        (ow, oh), _ = cv2.getTextSize(oj_text, FONT, 0.65, FONT_THICKNESS)
        ox = roi_x + ROI_W - ow - 4
        oy = band_y1 + (HIDDEN_BAND_HEIGHT + oh) // 2
        cv2.putText(frame, oj_text, (ox, oy), FONT,
                    0.65, (0, 0, 0), FONT_THICKNESS + 3, cv2.LINE_AA)
        cv2.putText(frame, oj_text, (ox, oy), FONT,
                    0.65, (60, 60, 255), FONT_THICKNESS, cv2.LINE_AA)


def draw_global_info(
    frame: np.ndarray, frame_idx: int, t_sec: float,
    p1_state: BoardState, p2_state: BoardState,
    p1_score: int | None = None, p2_score: int | None = None,
) -> None:
    """画面上部に時刻 + 状態 + スコアを描画."""
    s1 = f"{p1_score}" if p1_score is not None else "---"
    s2 = f"{p2_score}" if p2_score is not None else "---"
    text = (
        f"frame={frame_idx} t={t_sec:.2f}s "
        f"1P={p1_state.value}({s1}) 2P={p2_state.value}({s2})"
    )
    cv2.putText(
        frame, text, (20, 30), FONT, 0.8,
        (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (20, 30), FONT, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA,
    )


_HSV_DB_ROOT = Path("data/per_video_hsv_ranges")
_HSV_MERGED_DEFAULT = _HSV_DB_ROOT / "_merged_default.json"

# 2範囲以上で定義されている色 → per_video inject 後に DEFAULT の補完範囲を保証する
# (赤は H=0-13 と H=166-180 の循環2範囲。per_video は高側のみ学習しがちなので
#  低側 H=0-13 が失われると赤を系統的に miss する)
_CIRCULAR_GUARD_COLORS: tuple[int, ...] = (COLOR_RED,)


def _ensure_circular_ranges_guard(classifier: object) -> None:
    """per_video inject 後に DEFAULT の循環補完範囲が欠落していないか確認し補完する。

    赤 (COLOR_RED=1) は H=0-13 と H=166-180 の2範囲で循環 Hue をカバーする。
    per_video inject が H=166-180 側のみ学習した場合、append=True でも
    DEFAULT の H=0-13 側が存在する前提。ただし inject 経路のバグや
    将来的な変更で欠落するリスクを guard する。

    Args:
        classifier: ColorClassifier インスタンス (_ranges 属性を持つオブジェクト)。
    """
    from src.image_reader import DEFAULT_COLOR_RANGES, HsvRange
    if not hasattr(classifier, "_ranges"):
        return
    for color in _CIRCULAR_GUARD_COLORS:
        if color not in DEFAULT_COLOR_RANGES:
            continue
        default_rngs = DEFAULT_COLOR_RANGES[color]
        if len(default_rngs) < 2:
            # DEFAULT が1範囲なら循環問題なし
            continue
        current: list[HsvRange] = list(classifier._ranges.get(color, []))
        for dflt in default_rngs:
            already = any(
                r.h_min == dflt.h_min and r.h_max == dflt.h_max
                for r in current
            )
            if not already:
                current.append(dflt)
                print(
                    f"[viz] circular_guard: color={color} "
                    f"H=[{dflt.h_min},{dflt.h_max}] を補完"
                )
        classifier._ranges[color] = current


def resolve_hsv_path(video_path: Path) -> Path:
    """動画ファイル名から動画 ID を抽出し、 per-video HSV JSON を自動選択する。

    優先順位:
      1. video_path のファイル名先頭から "(v[0-9]+)" を抽出
      2. data/per_video_hsv_ranges/{video_id}.json が存在 → それを返す (per-video 直接 inject)
      3. 不在 → _merged_default.json を返す (fallback)

    案 K (2026-05-24): 38 動画 union の背景誤認問題を per-video 直接 inject で回避。
    """
    import re
    m = re.match(r"(v\d+)", video_path.name)
    if m:
        candidate = _HSV_DB_ROOT / f"{m.group(1)}.json"
        if candidate.exists():
            return candidate
    return _HSV_MERGED_DEFAULT


def _extract_raw_hsv_board(
    reader: object,
    frame: "np.ndarray",
    region: object,
) -> list[list[int]]:
    """ImageReader から HSV-only 経路の直出力 board を取得する。

    HybridClassifier の _hsv 属性、または ColorClassifier 直参照で
    全 visible cell に HSV 分類を適用して 13×6 grid を返す。
    bg_fp / CNN 経路をバイパスした純 HSV 結果。

    Args:
        reader: ImageReader インスタンス。
        frame: 1920x1080 BGR フレーム。
        region: BoardRegion インスタンス。

    Returns:
        13×6 の int grid (COLOR_* 定数)。
    """
    import cv2 as _cv2
    import numpy as _np
    from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS

    # HSV 分類器の取得 (HybridClassifier._hsv or ColorClassifier 直)
    classifier_raw = getattr(reader, "_classifier", None)
    hsv_cls = getattr(classifier_raw, "_hsv", classifier_raw)

    img_h, img_w = frame.shape[:2]
    grid: list[list[int]] = []
    for row in range(BOARD_ROWS):
        row_vals: list[int] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_vals.append(COLOR_EMPTY)
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(y1 + 1, min(y2, img_h))
            patch = frame[y1:y2, x1:x2]
            try:
                color = int(hsv_cls.classify(patch)) if hsv_cls is not None else COLOR_EMPTY
            except Exception:
                color = COLOR_EMPTY
            row_vals.append(color)
        grid.append(row_vals)
    return grid


def _extract_bg_fp_distance_grid(
    reader: object,
    frame: "np.ndarray",
    region: object,
    side: str,
) -> list[list[float]] | None:
    """各 cell の bg_fp 距離 (float) を 13×6 grid で返す。

    bg_fp が未採取の場合は None を返す。
    hidden rows (row < HIDDEN_ROWS) は -1.0 で埋める。

    Args:
        reader: ImageReader インスタンス。
        frame: 1920x1080 BGR フレーム。
        region: BoardRegion インスタンス。
        side: "1P" or "2P"。

    Returns:
        13×6 の float grid、または None (bg_fp 未採取)。
    """
    import cv2 as _cv2
    import numpy as _np
    from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS

    bg_fp = getattr(reader, "_bg_fp_p1" if side == "1P" else "_bg_fp_p2", None)
    if bg_fp is None:
        return None

    try:
        from src.background_fingerprint import CellFingerprint
    except Exception:
        return None

    hsv_full = _cv2.cvtColor(frame, _cv2.COLOR_BGR2HSV)
    img_h, img_w = frame.shape[:2]

    grid: list[list[float]] = []
    for row in range(BOARD_ROWS):
        row_vals: list[float] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_vals.append(-1.0)
                continue
            visible_row = row - HIDDEN_ROWS
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(y1 + 1, min(y2, img_h))
            hsv_patch = hsv_full[y1:y2, x1:x2]
            if hsv_patch.size == 0:
                row_vals.append(-1.0)
                continue
            h_med = int(_np.median(hsv_patch[:, :, 0]))
            s_med = int(_np.median(hsv_patch[:, :, 1]))
            v_med = int(_np.median(hsv_patch[:, :, 2]))
            cur_fp = CellFingerprint(h_med, s_med, v_med)
            try:
                bg_cell = bg_fp.cell_at(visible_row, col)
                dist = float(cur_fp.distance_to(bg_cell))
            except Exception:
                dist = -1.0
            row_vals.append(dist)
        grid.append(row_vals)
    return grid


def _extract_tier1_threshold_grid(
    reader: object,
    region: object,
) -> list[list[float]]:
    """各 cell の tier1 閾値 (float) を 13×6 grid で返す。

    _resolve_tier1_threshold() が存在しない場合は全 cell DEFAULT を使う。
    hidden rows は -1.0 で埋める。

    Args:
        reader: ImageReader インスタンス。
        region: 使用しない (visible_row/col のみで閾値は決まる)。

    Returns:
        13×6 の float grid。
    """
    from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
    from src.image_reader import BG_EXTREME_THRESHOLD_DEFAULT

    resolve_fn = getattr(reader, "_resolve_tier1_threshold", None)

    grid: list[list[float]] = []
    for row in range(BOARD_ROWS):
        row_vals: list[float] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_vals.append(-1.0)
                continue
            visible_row = row - HIDDEN_ROWS
            if resolve_fn is not None:
                try:
                    thresh = float(resolve_fn(visible_row, col))
                except Exception:
                    thresh = BG_EXTREME_THRESHOLD_DEFAULT
            else:
                thresh = BG_EXTREME_THRESHOLD_DEFAULT
            row_vals.append(thresh)
        grid.append(row_vals)
    return grid


def _board_diff_cells(
    board_before: "Board | None",
    board_after: "Board | None",
) -> list[list[int]]:
    """2 つの board の差分 cell を [[row, col, before, after], ...] で返す。

    None の場合は空リストを返す。

    Args:
        board_before: 変更前の Board。
        board_after: 変更後の Board。

    Returns:
        差分 cell リスト。各要素 = [row, col, color_before, color_after]。
    """
    from src.board import BOARD_COLS, BOARD_ROWS
    if board_before is None or board_after is None:
        return []
    diffs: list[list[int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            bv = int(board_before.get(r, c))
            av = int(board_after.get(r, c))
            if bv != av:
                diffs.append([r, c, bv, av])
    return diffs


def _build_detailed_log_entry(
    fi: int,
    t_sec: float,
    result: object,
    frame: "np.ndarray",
    pipeline: object,
    prev_p1_confirmed: "Board | None",
    prev_p2_confirmed: "Board | None",
) -> dict:
    """詳細 board log の 1 エントリを生成する。

    SideResult.cnn_board (= ImageReader 直出力)、raw_hsv_board、
    bg_fp_distance_grid、tier1_threshold_grid、constraint_fill_changed_cells
    (= cnn_board → confirmed_board の差分) を記録する。

    Args:
        fi: frame index。
        t_sec: 経過秒数。
        result: PipelineResult インスタンス。
        frame: 1920x1080 BGR フレーム。
        pipeline: RecognitionPipeline インスタンス。
        prev_p1_confirmed: 前フレームの 1P confirmed_board (差分計算用)。
        prev_p2_confirmed: 前フレームの 2P confirmed_board (差分計算用)。

    Returns:
        JSON シリアライズ可能な dict。
    """
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

    reader = getattr(pipeline, "_reader", None)

    # raw_cnn_board = SideResult.cnn_board (= ImageReader 直出力、pipeline 内で保持済)
    p1_cnn = getattr(result.p1, "cnn_board", None)
    p2_cnn = getattr(result.p2, "cnn_board", None)

    # raw_hsv_board: HSV-only 経路の全 cell 再分類
    p1_hsv_grid: list[list[int]] | None = None
    p2_hsv_grid: list[list[int]] | None = None
    if reader is not None:
        try:
            p1_hsv_grid = _extract_raw_hsv_board(reader, frame, DEFAULT_P1_REGION)
            p2_hsv_grid = _extract_raw_hsv_board(reader, frame, DEFAULT_P2_REGION)
        except Exception:
            pass

    # bg_fp_distance_grid
    p1_bg_dist: list[list[float]] | None = None
    p2_bg_dist: list[list[float]] | None = None
    if reader is not None:
        try:
            p1_bg_dist = _extract_bg_fp_distance_grid(reader, frame, DEFAULT_P1_REGION, "1P")
            p2_bg_dist = _extract_bg_fp_distance_grid(reader, frame, DEFAULT_P2_REGION, "2P")
        except Exception:
            pass

    # tier1_threshold_grid
    p1_tier1: list[list[float]] | None = None
    p2_tier1: list[list[float]] | None = None
    if reader is not None:
        try:
            p1_tier1 = _extract_tier1_threshold_grid(reader, DEFAULT_P1_REGION)
            p2_tier1 = _extract_tier1_threshold_grid(reader, DEFAULT_P2_REGION)
        except Exception:
            pass

    # pre_capture_mode
    pre_capture = bool(getattr(reader, "_pre_capture_mode", False)) if reader else False

    # constraint_fill_changed_cells: cnn_board → confirmed_board の差分
    # (constraint-fill / physics-fix 両方含む「何かが変えた」差分)
    p1_constraint_diff = _board_diff_cells(p1_cnn, result.p1.confirmed_board)
    p2_constraint_diff = _board_diff_cells(p2_cnn, result.p2.confirmed_board)

    # physics_fix_changed_cells: prev_confirmed → confirmed の差分
    p1_physics_diff = _board_diff_cells(prev_p1_confirmed, result.p1.confirmed_board)
    p2_physics_diff = _board_diff_cells(prev_p2_confirmed, result.p2.confirmed_board)

    return {
        "frame_idx": fi,
        "t_sec": t_sec,
        "p1_state": result.p1.state.value,
        "p2_state": result.p2.state.value,
        "p1_confirmed": (
            result.p1.confirmed_board.to_dict()["grid"]
            if result.p1.confirmed_board is not None else None
        ),
        "p2_confirmed": (
            result.p2.confirmed_board.to_dict()["grid"]
            if result.p2.confirmed_board is not None else None
        ),
        # 詳細フィールド
        "p1_raw_cnn_board": p1_cnn.to_dict()["grid"] if p1_cnn is not None else None,
        "p2_raw_cnn_board": p2_cnn.to_dict()["grid"] if p2_cnn is not None else None,
        "p1_raw_hsv_board": p1_hsv_grid,
        "p2_raw_hsv_board": p2_hsv_grid,
        "p1_bg_fp_distance_grid": p1_bg_dist,
        "p2_bg_fp_distance_grid": p2_bg_dist,
        "p1_tier1_threshold_grid": p1_tier1,
        "p2_tier1_threshold_grid": p2_tier1,
        "pre_capture_mode": pre_capture,
        # constraint_fill で変更された cell: [row, col, cnn_color, confirmed_color]
        "p1_constraint_fill_changed_cells": p1_constraint_diff,
        "p2_constraint_fill_changed_cells": p2_constraint_diff,
        # physics_fix で変更された cell: [row, col, prev_confirmed_color, new_confirmed_color]
        "p1_physics_fix_changed_cells": p1_physics_diff,
        "p2_physics_fix_changed_cells": p2_physics_diff,
        # 着地色診断 (2026-06-01 infer_placement 調査用)。
        # TSUMO_FALL→STABLE 着地フレームのみ非 null。
        # falling_pair_old: prev_next_queue[-2] 由来 (従来ロジック)
        # falling_pair_new: _landing_pending[1] 由来 (修正ロジック)
        # source: "landing_pending" | "next_queue_2" | "next_queue_1" | "none"
        "p1_landing_diag": getattr(result.p1, "landing_diag", None),
        "p2_landing_diag": getattr(result.p2, "landing_diag", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-interval", type=float,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="認識処理する frame 間隔 (秒)。出力は元動画の fps を維持し、未サンプル frame は最後の認識結果を保持。",
    )
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="入力動画の処理最大秒数 (0=全部)",
    )
    parser.add_argument("--cnn-model", type=str, default=None)
    parser.add_argument(
        "--hsv-state", type=Path,
        default=None,
        help="動画別 HSV ranges JSON (per_video_hsv_ranges DB)。 起動時に "
             "ColorClassifier.set_color_ranges_from_simple で注入。 "
             "省略時は resolve_hsv_path() が動画 ID から自動選択 "
             "(= 案 K: per-video JSON 優先、 不在時 _merged_default.json fallback)。 "
             "ファイル不在なら silent skip。 無効化したい場合は存在しないパスを渡す。",
    )
    parser.add_argument(
        "--vote-mode", action="store_true",
        help="ColorClassifier を per-pixel 投票方式に切替 (cycle 71)",
    )
    parser.add_argument(
        "--cnn-override-prob", type=float, default=None,
        help="HybridClassifier CNN 採用閾値 (None で default 0.70).",
    )
    parser.add_argument(
        "--mask-ojama-logit", action="store_true",
        help="cycle 32e: CNN の ojama logit を推論時 mask (= argmax 候補から除外)。 "
             "cycle 32e/32f model 等 ojama を学習対象外にした model で使用。",
    )
    parser.add_argument(
        "--no-online-hsv", action="store_true",
        help="OnlineHsvCalibrator を完全無効化 (= 動画中 HSV 学習なし)。 "
             "Step 0 OnlineHsv 効果定量評価用 (2026-05-24)。",
    )
    parser.add_argument(
        "--no-per-video-hsv",
        action="store_true",
        default=False,
        dest="disable_per_video_hsv",
        help="per-video 手調整 HSV inject をスキップする (汎用精度目視確認用)。 "
             "OnlineHsvCalibrator は引き続き動作するため自動 HSV 学習は生きる。 "
             "デフォルト False = 従来挙動完全一致 (backwards compat)。",
    )
    parser.add_argument(
        "--use-puyo-gate", action="store_true",
        help="cycle 32e: PuyoPresenceGate を HybridClassifier 前段に挟む。 "
             "gate=False の patch は HSV-only 経路に倒す (= 背景誤認対策)。",
    )
    parser.add_argument(
        "--use-circle-mask", action="store_true",
        help="cycle 32g: 推論時 patch に円形マスク適用 (学習時と必ず揃える)",
    )
    parser.add_argument(
        "--dump-board-log", type=Path, default=None,
        help="cycle 33 (2026-05-19): 各 frame の confirmed_board / state / "
             "chain_event を JSONL 形式で保存 (= 強化アナリスト用)。 "
             "evaluator が後処理で読み込んで自動評価する。",
    )
    parser.add_argument(
        "--dump-board-log-detailed", type=Path, default=None,
        help="Q1 全力施策: 各 frame の raw_cnn_board / raw_hsv_board / "
             "bg_fp_distance_grid / tier1_threshold_grid / pre_capture_mode / "
             "constraint_fill_changed_cells / physics_fix_changed_cells を含む "
             "詳細 JSONL を保存。 --dump-board-log の superset。 "
             "認識不能 frame の真因解明用 (= extract_unrecognized_frames.py の入力)。",
    )
    # B1 (warmup guard): STABLE 遷移直後 N frame confirmed 凍結 (= A/B 仮説 B1/B3 対応)
    # backwards compat: store_true で default=False、既存挙動は変わらない。
    parser.add_argument(
        "--enable-warmup-guard", action="store_true",
        help="B1 (M1 warmup guard): STABLE 遷移直後 N frame の confirmed board 更新を凍結。 "
             "遷移直後の CNN ノイズ (エフェクト残光・背景誤認) を吸収する。 "
             "A/B 仮説 B1/B3 用 (2026-05-27)。",
    )
    # B2 (bg_fp_force_max_puyo): 背景 FP 採取の緩和条件を絞る (= A/B 仮説 B2/B3 対応)
    # backwards compat: default=None で既存 class attribute 値 (=144) を維持。
    parser.add_argument(
        "--bg-fp-force-max-puyo", type=int, default=None,
        help="B2 (BG_FP_FORCE_MAX_PUYO): 背景 FP 採取緩和の puyo 上限数を上書き。 "
             "None (省略時) = class attribute 値 144 を維持。 "
             "4 に絞ると試合序盤の FP 採取機会が減り誤採取を抑制する。 "
             "A/B 仮説 B2/B3 用 (2026-05-27)。",
    )
    parser.add_argument(
        "--patch-ncc-threshold", type=float, default=None,
        help="PatchBackgroundFingerprint NCC 閾値を上書き (= default は "
             "background_fingerprint.py の PATCH_NCC_EMPTY_THRESHOLD = 0.92)。 "
             "NCC sweep 用 (候補: 0.85, 0.88, 0.90, 0.92)。",
    )
    parser.add_argument(
        "--enable-piece-persistence",
        action="store_true",
        default=False,
        help="B1 PiecePersistenceGuard を有効化 (= STABLE 中 cell 色保護、 散発色ブレ削減)。",
    )
    parser.add_argument(
        "--enable-tier1-warmup",
        action="store_true",
        default=False,
        help="tier1 warmup guard を有効化 (= NON-STABLE → STABLE 遷移直後 "
             f"TIER1_WARMUP_FRAMES={3} frame 間 tier1 を skip し、"
             "ツモ着地直後の cell を tier1 が誤 EMPTY 化するのを防ぐ)。",
    )
    parser.add_argument(
        "--ojama-tier1-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_tier1_warmup",
        help="経路 A': OJAMA_FALL → STABLE 遷移専用の tier1 warmup を制御する。"
             f" OJAMA_TIER1_WARMUP_FRAMES={8} frame 間 tier1 を skip し、"
             "お邪魔消滅後のセル背景化による誤 EMPTY 化 → 列崩壊を防ぐ (v70 対策)。"
             " ライブラリ default=True (有効)。 --no-ojama-tier1-warmup で無効化。",
    )
    parser.add_argument(
        "--constraint-fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_constraint_fill",
        help="案2: NEXT 累積制約による色 count 補正 (constraint_fill) を制御する。 "
             "--constraint-fill で有効化、 --no-constraint-fill で無効化。 "
             "ライブラリ default=False (無効)。 "
             "--constraint-fill で有効化して比較 (constraint_fill の net 効果測定用)。",
    )
    parser.add_argument(
        "--t2-highconf-yield",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_t2_highconf_yield",
        help="T2 高確信 yield を制御する。 "
             "STABLE → STABLE 遷移時の prev_stable 上書き (T2) において、 "
             "CNN が現在の confirmed 色を支持しているセルはスキップする。 "
             "infer_placement 誤推論 + T2 自己強化フリーズによる色破壊修正。 "
             "ライブラリ default=True (有効)。 --no-t2-highconf-yield で無効化。",
    )
    parser.add_argument(
        "--infer-empty-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_infer_empty_guard",
        help="infer_placement 空セル hallucination ガードを制御する。 "
             "pattern の非 diff セルが cnn_after で COLOR_EMPTY な候補をスキップし、 "
             "CNN が確信して空なセルへの NEXT 色書込 (hallucination) を防ぐ。 "
             "ライブラリ default=True (有効)。 --no-infer-empty-guard で無効化。",
    )
    parser.add_argument(
        "--game-event-chain-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_game_event_chain_exit",
        help="game-event ベース連鎖終了を制御する。 "
             "CHAIN 状態を timing hold だけでなく「次ツモ変化」または"
             "「連鎖側お邪魔降下」を検知するまで維持する。 "
             "安全弁として CHAIN_MAX_HOLD_SEC (5.0s) 超過で強制終了。 "
             "ライブラリ default=True (有効)。 --no-game-event-chain-exit で無効化。",
    )
    parser.add_argument(
        "--landing-color-fix",
        action="store_true",
        default=False,
        dest="enable_landing_color_fix",
        help="着地色修正 案1: TSUMO_FALL→STABLE 着地時の falling_pair を "
             "prev_next_queue[-2] から _landing_pending (消費済みツモ色) に切り替える。 "
             "slide_motion(R-7) 経由で 1 つ前のツモ色を指してしまう誤色問題の修正。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。 "
             "フラグ OFF でも --dump-board-log-detailed に landing_diag フィールドが記録される。",
    )
    parser.add_argument(
        "--chain-min-display",
        action="store_true",
        default=False,
        dest="enable_chain_min_display",
        help="X1/X4 短連鎖ちらつき対策を有効化。 "
             f"CHAIN 最小表示時間 (CHAIN_MIN_DISPLAY_SEC={RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC}s) + "
             f"短連鎖 game-event exit 抑止 (chain_count < {RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT})。 "
             "enable_game_event_chain_exit と独立フラグ (効果分解のため)。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    parser.add_argument(
        "--hsv-classify-fallback",
        action="store_true",
        default=False,
        dest="enable_hsv_classify_fallback",
        help="HSV 分類 fallback を有効化。 "
             "_classify_next_pair_by_hsv の 2 択強制確定を回避し、 "
             "両候補が拮抗・両候補とも遠い・低彩度 patch の場合は next_pair 素返しにする。 "
             "黄(H26)→赤(H7) 誤分類 (~900 件、 H 差 19) 発火点対策。 "
             "デフォルト OFF = 従来挙動不変 (2 択強制確定、 backwards compat)。",
    )
    parser.add_argument(
        "--landing-observed-color",
        action="store_true",
        default=False,
        dest="enable_landing_observed_color",
        help="真因 A 対処: 着地セルの CNN==HSV 一致色補正を有効化。 "
             "TSUMO_FALL→STABLE 着地時に 2 つの独立認識器 (CNN/HSV) が "
             "一致した着地色を優先し、 falling_pair タイミングずれによる誤色を断つ。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    parser.add_argument(
        "--red-hue-wrap-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_red_hue_wrap_fix",
        help="赤色相折り返し補正を制御する。 "
             "赤ぷよの H 画素が 0-4 と 166-179 に 2 峰分布するため単純 median が "
             "赤/黄境界 (H=13/14) に乗り毎フレームちらつく問題を修正する。 "
             "ライブラリ default=True (有効)。 --no-red-hue-wrap-fix で無効化。",
    )
    parser.add_argument(
        "--specular-robust-saturation",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_specular_robust_saturation",
        help="案D: 光沢ハイライト除外彩度計算を制御する。 "
             "ぷよ表面の白ハイライト画素 (V>=210 かつ S<=60) を彩度 median 計算から除外し、 "
             "光沢球混入による EMPTY 誤判定を防ぐ。 "
             "ライブラリ default=True (有効)。 --no-specular-robust-saturation で無効化。",
    )
    parser.add_argument(
        "--stable-recovery-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_stable_recovery_gate",
        help="設計C 事後復旧ゲートを制御する。 "
             "STABLE 中に confirmed==EMPTY なのに CNN==HSV が同一有効色で "
             "8 フレーム継続したセルを confirmed に復旧する。 "
             "ライブラリ default=True (有効)。 --no-stable-recovery-gate で無効化。",
    )
    # フェーズ A4 (2026-06-02): お邪魔ぷよ視覚的検出・連鎖終了・推論ガード・着地検出
    parser.add_argument(
        "--enable-ojama-visual-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_visual_detection",
        help="フェーズ A4: お邪魔ぷよ視覚的検出を制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-visual-detection で無効化。",
    )
    parser.add_argument(
        "--enable-ojama-visual-chain-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_visual_chain_exit",
        help="フェーズ A4: お邪魔ぷよ視覚的検出による連鎖終了判定を制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-visual-chain-exit で無効化。",
    )
    parser.add_argument(
        "--enable-ojama-infer-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_infer_guard",
        help="フェーズ A4: お邪魔ぷよ推論ガードを制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-infer-guard で無効化。",
    )
    parser.add_argument(
        "--enable-ojama-settle-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_settle_detection",
        help="フェーズ A4: お邪魔ぷよ着地検出を制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-settle-detection で無効化。",
    )
    parser.add_argument(
        "--chain-score-early-fire",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_score_early_fire",
        help="機能B: score 急増 CHAIN 早期発火を制御する。 "
             f"True にすると自 side の score_delta >= CHAIN_SCORE_EARLY_FIRE_DELTA={80} "
             "の frame で VideoChainTracker の puyo 減少検知を待たずに即 CHAIN state に突入する。 "
             "OCR 失敗 / score 取得不可時は従来の VideoChainTracker 経路を維持 (OR 追加)。 "
             "ライブラリ default=False (無効)。 --chain-score-early-fire で有効化。",
    )
    parser.add_argument(
        "--chain-exit-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_exit_warmup",
        help="機能C: CHAIN → STABLE 遷移直後の confirmed 凍結を制御する。 "
             f"True にすると CHAIN→STABLE 復帰から CHAIN_EXIT_WARMUP_SEC={0.1}s 間 confirmed "
             "更新を凍結しエフェクト残光色の混入を防ぐ。 "
             "時間ベース実装のため fps 非依存。 "
             "ライブラリ default=False (無効)。 --chain-exit-warmup で有効化。",
    )
    parser.add_argument(
        "--chain-formula-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_chain_formula_detection",
        help="機能D: 連鎖開始 掛け算式 検知を制御する。 "
             "True にすると score ROI の OCR が None (掛け算式表示で NCC conf 低下) かつ "
             "ink_ratio > CHAIN_FORMULA_INK_RATIO_MIN かつ last_score > 0 が "
             "CHAIN_FORMULA_CONSEC_FRAMES 連続で成立した frame で即 CHAIN state に突入する。 "
             "機能B (score 急増経路) と独立フラグ。 "
             "ライブラリ default=True (有効、 2026-06-03 採用)。 --no-chain-formula-detection で無効化。",
    )
    parser.add_argument(
        "--chain-formula-simulate-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_formula_simulate_verify",
        help="修正D (2026-07-24): 機能D 疑似発火の起点盤面を ChainSimulator で "
             "事前検証する。 真因診断で機能D 早期発火 77件中35件=45.5%が "
             "連鎖ゼロの起点盤面からの疑似発火 (偽イベント) と確定した対策。 "
             "True で連鎖ゼロの起点盤面での疑似発火を抑制し、 連鎖ありは "
             "固定 chain_count=1 でなく実測値で発火する。 "
             "ライブラリ default=False (無効、 bit-identical)。 "
             "--chain-formula-simulate-verify で有効化。",
    )
    parser.add_argument(
        "--hsv-deferred-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_hsv_deferred_consensus",
        help="案 Y-4: HSV-first commit + deferred consensus を制御する。 "
             "True にすると infer_placement が HSV 拮抗と判定した着地 2 候補を保留し、 "
             "後続フレームの CNN==HSV consensus 投票で確定させる (corruption 65% 起源対策)。 "
             "ライブラリ default=False (無効)。 --hsv-deferred-consensus で有効化。",
    )
    # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)
    # store_true を使う (BooleanOptionalAction の --no- 接頭辞反転バグ回避)
    parser.add_argument(
        "--ojama-warning-glow-guard",
        action="store_true",
        default=False,
        dest="enable_ojama_warning_glow_guard",
        help="不具合B 対処: 予告おじゃま発光ガードを有効化する。 "
             "相手連鎖の予告おじゃま演出による盤面上部多色発光を V_high_ratio で検知し、 "
             "STABLE 中の confirmed_board を frozen_board で保護する。 "
             "黄ぷよに発光が重なる黄(4)→おじゃま(9)誤認を防ぐ。 "
             "ライブラリ default=False (無効)。 --ojama-warning-glow-guard で有効化。",
    )
    parser.add_argument(
        "--chain-max-hold-override",
        action="store_true",
        default=False,
        dest="enable_chain_max_hold_override",
        help="案P3: CHAIN_MAX_HOLD_SEC 超過後の ojama 保留を無効化する。 "
             f"active_chain が CHAIN_MAX_HOLD_SEC={RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s "
             "超過で強制クリアされた frame では ojama_top_positive による STABLE 復帰保留を "
             "スキップして強制 STABLE に遷移させる (安全弁を本来機能させる)。 "
             "enable_ojama_visual_chain_exit=True と組み合わせて使用する。 "
             "ライブラリ default=False (無効)。 --chain-max-hold-override で有効化。",
    )
    # 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了 (2026-06-05)
    # store_true を使う (BooleanOptionalAction の --no- 接頭辞反転バグ回避)
    parser.add_argument(
        "--chain-exit-next-signal",
        action="store_true",
        default=False,
        dest="enable_chain_exit_next_signal",
        help="案X*: NextSlide signal による CHAIN 即終了を有効化する。 "
             "(A) 機能D 再点火抑制: 既に CHAIN 中なら 機能D (掛け算式) の発火をスキップし "
             "max_until 延長を止める。 "
             "(B) NextSlide signal (次ツモスライド) 検知で CHAIN を即終了させる。 "
             "warmup 連動: CHAIN_EXIT_WARMUP_SEC 秒間 confirmed 凍結を自動適用。 "
             "真因: ojama_top_positive 保留 + 機能D 再点火による 6.87 秒過剰保持 (v89 1P) を解消。 "
             "ライブラリ default=False (無効)。 --chain-exit-next-signal で有効化。",
    )
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化
    # 2026-06-06 採用: default=True。--no-gravity-settle-state で無効化可。
    parser.add_argument(
        "--gravity-settle-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_gravity_settle_state",
        help="GRAVITY_SETTLE 状態を有効化する (feat/gravity-settle-2026-06-05)。 "
             "連鎖終了直後の重力 settle/着地中を採点外・confirmed 凍結として扱う。 "
             "CHAIN → GRAVITY_SETTLE → STABLE の遷移経路を有効化する。 "
             "案X (--chain-exit-next-signal) との組み合わせを推奨 (内部で自動 ON)。 "
             "default=True (有効、2026-06-06 採用)。 --no-gravity-settle-state で無効化。",
    )
    parser.add_argument(
        "--slide-override-ojama-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_slide_override_ojama_hold",
        help="案γ: CHAIN 中 slide_motion=True (次ツモスライド) が来た場合に "
             "ojama_top_positive による CHAIN 過剰保持 (ojama-hold ガード) を上書きして終了。 "
             "v89 t35.2-39.67 の連鎖過剰保持修正。 "
             "default=True (有効、2026-06-06 採用)。 --no-slide-override-ojama-hold で無効化。",
    )
    parser.add_argument(
        "--dump-ojama-accounting",
        type=Path,
        default=None,
        dest="dump_ojama_accounting",
        help="OjamaAccountingTracker の各フレーム snapshot を JSONL で保存。 "
             "t_sec / p1_state / p2_state / score_p1/p2 / pending / net_balance / "
             "total_dropped / confidence を記録。 省略時は保存しない。",
    )
    # 全域無悪化ゲート (2026-08-07): バーストガード系フラグ6個の配線。
    # scripts/collect_boards_lean.py / scripts/measure_stable_cell_acc.py
    # (7335c24) と同一パターン (dest 名も同一、デフォルト全 OFF で
    # bit-identical、YouTubeデモ素材作成用)。
    parser.add_argument(
        "--enable-effect-gate", action="store_true", default=False,
        dest="enable_effect_gate",
        help=(
            "エフェクト時間ゲート (2026-08-03) を有効化する。満杯盤面 誤り根治用。 "
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-burst-guard-v2", action="store_true", default=False,
        dest="enable_burst_guard_v2",
        help=(
            "バーストガード再設計 Stage1 (2026-08-05) を有効化する。 "
            "Schmitt trigger 視覚トリガー + ハード凍結。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-transition-merge-guard", action="store_true", default=False,
        dest="enable_transition_merge_guard",
        help=(
            "バーストガード Stage1.5 (2026-08-05) を有効化する。 "
            "NON-STABLE→STABLE 遷移merge直前に物理的期待値フィルタを適用する。 "
            "--enable-burst-guard-v2 が無効の間は no-op。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--burst-gate-open-threshold", type=float, default=None,
        dest="burst_gate_open_threshold",
        help=(
            "バーストガード Schmitt trigger の開窓閾値を上書きする "
            "(CLOSE も同値運用)。既定 None = BURST_GATE_OPEN_THRESHOLD (0.97)。"
            "全域無悪化ゲート採用値は 0.954。"
        ),
    )
    parser.add_argument(
        "--enable-hidden-row-burst-guard", action="store_true", default=False,
        dest="enable_hidden_row_burst_guard",
        help=(
            "バーストガード Stage1.5b (2026-08-05、§11) を有効化する。 "
            "row1-3 凍結中/close直後クールダウン中の infer_hidden_row 呼び出しを "
            "スキップし row0 誤色書き込みを防ぐ。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-match-transition-debounce", action="store_true", default=False,
        dest="enable_match_transition_debounce",
        help=(
            "長時間劣化修正 A' (2026-08-06) を有効化する。 "
            "is_active の True/False遷移を対称デバウンスし、MATCH_TRANSITION_"
            "DEBOUNCE_SEC (1.0秒) 未満のフリッカーによる誤再アーム/リセットを"
            "防ぐ。既定は無効 (後方互換)。"
        ),
    )
    # YouTubeデモ素材用 (2026-08-07): STABLE 以外はセル文字オーバーレイを
    # 非表示にする表示モード。状態ラベル (1P=chain 等) や盤面枠色は維持する。
    parser.add_argument(
        "--overlay-stable-only", action="store_true", default=False,
        dest="overlay_stable_only",
        help=(
            "各プレイヤーの状態が STABLE でない間、そのプレイヤー側のセル文字"
            "オーバーレイ (R/G/B/Y/O/? 等) を描画しない。左右は独立判定 "
            "(1PがCHAINでもP2がSTABLEならP2側は描画する)。状態表示テキストは "
            "維持する。既定は無効 (後方互換、従来通り常時描画)。"
        ),
    )
    # YouTubeデモ素材用 (2026-08-07): 任意の state 集合でセル文字オーバーレイの
    # 表示/非表示を切り替える汎用モード。--overlay-stable-only の一般化。
    parser.add_argument(
        "--overlay-show-states", type=str, default=None,
        dest="overlay_show_states",
        help=(
            "カンマ区切りの BoardState 名 (小文字、例: stable,tsumo_fall) を指定する。"
            "各プレイヤーの現在状態がこの集合に含まれる場合のみそのプレイヤー側の"
            "セル文字オーバーレイを描画する (左右は独立判定)。状態表示テキストは"
            "維持する。--overlay-stable-only とは併用不可。未指定時は従来動作"
            "(後方互換)。"
        ),
    )
    # 幽霊セル対策 (2026-08-07): chain/ojama_fall から抜けた直後 N フレームは
    # セル文字オーバーレイを描画しない (一時汚染の自己修復待ち)。0=無効 (既定・後方互換)。
    parser.add_argument(
        "--overlay-transition-hold-frames", type=int, default=0,
        dest="overlay_transition_hold_frames",
        help=(
            "各プレイヤーの状態が CHAIN または OJAMA_FALL から抜けた時点を起点に、"
            "指定フレーム数の間はそのプレイヤー側のセル文字オーバーレイを描画しない "
            "(既存の --overlay-stable-only / --overlay-show-states の判定結果に AND "
            "する追加ガード)。ホールド中に再度 CHAIN/OJAMA_FALL に入るとカウンタは"
            "リセットされ、次に抜けた時点から改めてホールドが始まる。左右は独立判定。"
            "既定 0 = 無効 (後方互換)。"
        ),
    )
    args = parser.parse_args()
    # --overlay-show-states の検証・解決 (BoardState への変換、不正値は起動時エラー)
    overlay_show_states: frozenset[BoardState] | None = None
    if args.overlay_show_states is not None:
        if getattr(args, "overlay_stable_only", False):
            parser.error(
                "--overlay-show-states と --overlay-stable-only は併用不可"
            )
        _state_by_name = {s.name.lower(): s for s in BoardState}
        _requested_names = [
            n.strip().lower() for n in args.overlay_show_states.split(",") if n.strip()
        ]
        _invalid_names = [n for n in _requested_names if n not in _state_by_name]
        if _invalid_names:
            parser.error(
                f"--overlay-show-states に不正な状態名: {_invalid_names} "
                f"(有効値: {sorted(_state_by_name)})"
            )
        overlay_show_states = frozenset(_state_by_name[n] for n in _requested_names)
    # 案 K (2026-05-24): --hsv-state 省略時は動画 ID から自動選択
    if args.hsv_state is None:
        args.hsv_state = resolve_hsv_path(args.video)
        print(f"[viz] HSV auto-resolve: {args.hsv_state} (from {args.video.name})")
    # --no-per-video-hsv: per-video 手調整 HSV inject をスキップする (汎用精度目視確認用)
    # None にすることで下流の inject ブロックを無効化する
    if getattr(args, "disable_per_video_hsv", False):
        args.hsv_state = None
        print(
            "[viz] disable_per_video_hsv=ON "
            "(手調整 per-video HSV inject スキップ: 自動 HSV + merged レンジのみで動作)"
        )
    # cycle 32g: 円形マスクを推論前に有効化
    if args.use_circle_mask:
        from src.patch_classifier import set_circle_mask_enabled
        set_circle_mask_enabled(True)
        print("[viz] use_circle_mask=ON (cycle 32g)")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_sec > 0:
        n_frames = min(n_frames, int(args.max_sec * fps))
    # 認識処理は 1920x1080 前提、出力もそのサイズで揃える
    out_w, out_h = 1920, 1080
    print(f"[input] {args.video} {width}x{height} fps={fps:.1f} frames={n_frames}")
    print(f"[output] {out_w}x{out_h} (resize から書き出し)")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), fourcc, fps, (out_w, out_h),
    )

    # Pipeline (force_in_match=True で MENU 判定スキップ)
    pipeline = RecognitionPipeline.load_default(
        # 2 だと一時的 empty 観測で confirmed_board が誤確定する問題あり
        # (= 26s 1P 盤面 fully filled なのに empty 確定)。 6 で慎重に。
        # サイクル70: 6→3 で遅延 50% 短縮 (= 60fps で 50ms)
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        vote_mode=args.vote_mode,
        cnn_override_prob=args.cnn_override_prob,
        mask_ojama_logit=args.mask_ojama_logit,
        use_puyo_gate=args.use_puyo_gate,
        # B1/B2/B3 仮説対応 (2026-05-27): --enable-warmup-guard / --bg-fp-force-max-puyo
        enable_warmup_guard=args.enable_warmup_guard,
        bg_fp_force_max_puyo=args.bg_fp_force_max_puyo,
        # NCC sweep 用 (2026-05-28): --patch-ncc-threshold で閾値上書き
        patch_ncc_threshold=args.patch_ncc_threshold,
        # B1 PiecePersistenceGuard (2026-05-28): --enable-piece-persistence
        enable_piece_persistence=args.enable_piece_persistence,
        # tier1 warmup guard (2026-05-28): --enable-tier1-warmup
        enable_tier1_warmup=args.enable_tier1_warmup,
        # 経路 A' (2026-05-30): --ojama-tier1-warmup で OJAMA 専用 warmup 有効化
        enable_ojama_tier1_warmup=args.enable_ojama_tier1_warmup,
        # 案2 (2026-05-30 / BooleanOptionalAction 整合 2026-06-02):
        # --constraint-fill / --no-constraint-fill で直接制御、反転ロジック除去済み
        enable_constraint_fill=args.enable_constraint_fill,
        # T2 高確信 yield (2026-05-31): --t2-highconf-yield で T2 フリーズ修正を有効化
        enable_t2_highconf_yield=args.enable_t2_highconf_yield,
        # 空セル hallucination ガード (2026-06-01): --infer-empty-guard で有効化
        enable_infer_empty_guard=args.enable_infer_empty_guard,
        # game-event ベース連鎖終了 (2026-06-01): --game-event-chain-exit で有効化
        enable_game_event_chain_exit=args.enable_game_event_chain_exit,
        # 着地色修正 案1 (2026-06-01): --landing-color-fix で有効化
        enable_landing_color_fix=args.enable_landing_color_fix,
        # X1/X4 短連鎖ちらつき対策 (2026-06-01): --chain-min-display で有効化
        enable_chain_min_display=args.enable_chain_min_display,
        # HSV 分類 fallback (fix/v70-zeropatch-redyellow, 2026-06-01): --hsv-classify-fallback で有効化
        enable_hsv_classify_fallback=args.enable_hsv_classify_fallback,
        # 真因 A 対処 (2026-06-01): --landing-observed-color で有効化
        enable_landing_observed_color=args.enable_landing_observed_color,
        # 赤色相折り返し補正 (fix/v70-zeropatch-redyellow, 2026-06-02): --red-hue-wrap-fix で有効化
        enable_red_hue_wrap_fix=args.enable_red_hue_wrap_fix,
        # 案D 光沢ハイライト除外彩度計算 (fix/v70-zeropatch-redyellow): --specular-robust-saturation で有効化
        enable_specular_robust_saturation=args.enable_specular_robust_saturation,
        # 設計C 事後復旧ゲート (2026-06-02): --stable-recovery-gate で有効化
        enable_stable_recovery_gate=args.enable_stable_recovery_gate,
        # フェーズ A4 (2026-06-02): --enable-ojama-visual-detection で有効化
        enable_ojama_visual_detection=args.enable_ojama_visual_detection,
        # フェーズ A4 (2026-06-02): --enable-ojama-visual-chain-exit で有効化
        enable_ojama_visual_chain_exit=args.enable_ojama_visual_chain_exit,
        # フェーズ A4 (2026-06-02): --enable-ojama-infer-guard で有効化
        enable_ojama_infer_guard=args.enable_ojama_infer_guard,
        # フェーズ A4 (2026-06-02): --enable-ojama-settle-detection で有効化
        enable_ojama_settle_detection=args.enable_ojama_settle_detection,
        # 機能B (2026-06-02): --chain-score-early-fire で有効化
        enable_chain_score_early_fire=args.enable_chain_score_early_fire,
        # 機能C (2026-06-02): --chain-exit-warmup で有効化
        enable_chain_exit_warmup=args.enable_chain_exit_warmup,
        # 機能D (2026-06-02): --chain-formula-detection で有効化
        enable_chain_formula_detection=args.enable_chain_formula_detection,
        enable_chain_formula_simulate_verify=(
            args.enable_chain_formula_simulate_verify
        ),
        # 案 Y-4 (2026-06-03): --hsv-deferred-consensus で有効化
        enable_hsv_deferred_consensus=args.enable_hsv_deferred_consensus,
        # 不具合B 対処 (2026-06-04): --ojama-warning-glow-guard で有効化
        enable_ojama_warning_glow_guard=args.enable_ojama_warning_glow_guard,
        # 案P3 (2026-06-05): --chain-max-hold-override で有効化
        enable_chain_max_hold_override=args.enable_chain_max_hold_override,
        # 案X*(A)(B)+warmup (2026-06-05): --chain-exit-next-signal で有効化
        enable_chain_exit_next_signal=args.enable_chain_exit_next_signal,
        # feat/gravity-settle-2026-06-05: --gravity-settle-state で有効化
        enable_gravity_settle_state=args.enable_gravity_settle_state,
        # 案γ (2026-06-06): --slide-override-ojama-hold で有効化
        enable_slide_override_ojama_hold=args.enable_slide_override_ojama_hold,
        # 全域無悪化ゲート (2026-08-07): バーストガード系フラグ6個。
        # scripts/collect_boards_lean.py と同一パターン (末尾追加)。
        enable_effect_gate=args.enable_effect_gate,
        enable_burst_guard_v2=args.enable_burst_guard_v2,
        enable_transition_merge_guard=args.enable_transition_merge_guard,
        burst_gate_open_threshold=args.burst_gate_open_threshold,
        enable_hidden_row_burst_guard=args.enable_hidden_row_burst_guard,
        enable_match_transition_debounce=args.enable_match_transition_debounce,
    )
    if args.patch_ncc_threshold is not None:
        print(f"[viz] patch_ncc_threshold={args.patch_ncc_threshold} (NCC sweep)")
    # 案 R3 改 (2026-05-28): pipeline に video_id を設定
    # bg_fp 採取完了後に per-video PuyoColorProfileDB が自動ロードされる
    _vid_match = __import__("re").search(r"(v\d+)", args.video.name)
    if _vid_match and hasattr(pipeline, "set_video_id"):
        _vid_id = _vid_match.group(1)
        pipeline.set_video_id(_vid_id)
        print(f"[viz] puyo_profile video_id={_vid_id} (R3改: per-video profile 自動ロード)")
    if args.enable_warmup_guard:
        print("[viz] enable_warmup_guard=ON (B1: STABLE 直後 confirmed 凍結)")
    if args.enable_piece_persistence:
        print("[viz] enable_piece_persistence=ON (B1: STABLE 中 cell 色保護 / 散発色ブレ削減)")
    if args.enable_tier1_warmup:
        print(
            "[viz] enable_tier1_warmup=ON "
            "(NON-STABLE→STABLE 遷移直後 3 frame tier1 skip / 着地直後誤 EMPTY 化防止)"
        )
    if args.enable_ojama_tier1_warmup:
        print(
            "[viz] enable_ojama_tier1_warmup=ON "
            "(経路 A': OJAMA_FALL→STABLE 遷移直後 8 frame tier1 skip / v70 列崩壊対策)"
        )
    if not args.enable_constraint_fill:
        print("[viz] constraint_fill=OFF (案2: constraint_fill 無効 / CNN 高確信セル保護)")
    elif args.enable_constraint_fill:
        print("[viz] constraint_fill=ON (明示指定: constraint_fill 有効化)")
    if args.enable_t2_highconf_yield:
        print(
            "[viz] t2_highconf_yield=ON "
            "(T2 高確信 yield: CNN 支持セルは prev_stable 上書きスキップ / "
            "infer_placement 誤推論 + T2 自己強化フリーズ修正)"
        )
    if args.enable_infer_empty_guard:
        print(
            "[viz] infer_empty_guard=ON "
            "(空セル hallucination ガード: 非 diff セルが CNN EMPTY なら候補スキップ)"
        )
    if args.enable_game_event_chain_exit:
        print(
            "[viz] game_event_chain_exit=ON "
            "(game-event ベース連鎖終了: 次ツモ変化 / お邪魔降下で CHAIN 終了 / "
            f"安全弁 max={RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s)"
        )
    if args.enable_landing_color_fix:
        print(
            "[viz] landing_color_fix=ON "
            "(着地色修正 案1: falling_pair を _landing_pending 消費色に切り替え / "
            "slide_motion 経由の 1 つ前ツモ色誤書き修正)"
        )
    if args.enable_chain_min_display:
        print(
            "[viz] chain_min_display=ON "
            f"(X1: 最小{RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC}s 表示保証 / "
            f"X4: chain_count < {RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT} で exit 抑止)"
        )
    if args.enable_hsv_classify_fallback:
        print(
            "[viz] hsv_classify_fallback=ON "
            "(2 択強制確定回避: 両候補拮抗/遠い/低彩度で next_pair 素返し / "
            "黄→赤誤分類 ~900 件発火点対策)"
        )
    if args.enable_landing_observed_color:
        print(
            "[viz] landing_observed_color=ON "
            "(真因 A 対処: 着地 2 cell の CNN==HSV 一致色で falling_pair ズレを補正)"
        )
    # フェーズ A4 (2026-06-02): お邪魔ぷよ視覚的検出関連ログ
    if args.enable_ojama_visual_detection:
        print("[viz] enable_ojama_visual_detection=ON (フェーズ A4: お邪魔ぷよ視覚的検出)")
    if args.enable_ojama_visual_chain_exit:
        print("[viz] enable_ojama_visual_chain_exit=ON (フェーズ A4: お邪魔ぷよ視覚的連鎖終了判定)")
    if args.enable_ojama_infer_guard:
        print("[viz] enable_ojama_infer_guard=ON (フェーズ A4: お邪魔ぷよ推論ガード)")
    if args.enable_ojama_settle_detection:
        print("[viz] enable_ojama_settle_detection=ON (フェーズ A4: お邪魔ぷよ着地検出)")
    if args.enable_chain_score_early_fire:
        print(
            "[viz] chain_score_early_fire=ON "
            f"(機能B: score >= {80} で即 CHAIN 突入 / VideoChainTracker フォールバック維持)"
        )
    if args.enable_chain_exit_warmup:
        print(
            "[viz] chain_exit_warmup=ON "
            f"(機能C: CHAIN→STABLE 後 {0.1}s confirmed 凍結 / エフェクト残光混入防止)"
        )
    if args.enable_chain_max_hold_override:
        print(
            "[viz] chain_max_hold_override=ON "
            f"(案P3: CHAIN_MAX_HOLD_SEC={RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s 超過後 "
            "ojama 保留を強制解除 / 連鎖過剰保持 v89 t34-40.87 修正)"
        )
    if args.enable_chain_exit_next_signal:
        print(
            "[viz] chain_exit_next_signal=ON "
            "(案X*: (A) 機能D CHAIN 中再点火抑制 + (B) NextSlide で CHAIN 即終了 "
            f"+ warmup {0.1}s confirmed 凍結 / v89 1P 6.87s 過剰保持根本修正)"
        )
    if args.enable_gravity_settle_state:
        from src.board_state_machine import (
            GRAVITY_SETTLE_MIN_FRAMES, GRAVITY_SETTLE_MAX_SEC,
            GRAVITY_SETTLE_PHYSICS_CLEAR_MIN,
        )
        print(
            "[viz] gravity_settle_state=ON "
            f"(CHAIN → GRAVITY_SETTLE → STABLE: "
            f"min={GRAVITY_SETTLE_MIN_FRAMES}f physics_clear={GRAVITY_SETTLE_PHYSICS_CLEAR_MIN}f "
            f"timeout={GRAVITY_SETTLE_MAX_SEC}s / 連鎖後 settle 採点外)"
        )
    if args.enable_slide_override_ojama_hold:
        print(
            "[viz] slide_override_ojama_hold=ON "
            "(案γ: CHAIN 中 slide_motion=True が ojama-hold ガードを上書きして CHAIN 終了 "
            "/ v89 t35.2-39.67 連鎖過剰保持修正、2026-06-06 採用)"
        )
    else:
        print("[viz] slide_override_ojama_hold=OFF (--no-slide-override-ojama-hold 指定)")
    if args.bg_fp_force_max_puyo is not None:
        print(f"[viz] bg_fp_force_max_puyo={args.bg_fp_force_max_puyo} (B2: FP 採取制限)")
    # Step 0 (2026-05-24): --no-online-hsv で OnlineHsvCalibrator を無効化
    if args.no_online_hsv:
        pipeline._online_hsv = None
        print("[viz] online_hsv DISABLED (= Step 0 比較用)")
    if args.vote_mode:
        print("[viz] vote_mode=ON (per-pixel HSV voting)")
    if args.cnn_override_prob is not None:
        print(f"[viz] cnn_override_prob={args.cnn_override_prob}")
    if args.mask_ojama_logit:
        print("[viz] mask_ojama_logit=ON (cycle 32e)")
    if args.use_puyo_gate:
        print("[viz] use_puyo_gate=ON (cycle 32e)")
    # 2026-05-11 サイクル63: 元動画解像度を image_reader に通知
    # (image_reader.read_both_boards で 1920x1080 にリサイズされるため、
    # pipeline.update に渡る frame からは元解像度が分からない)。
    if hasattr(pipeline._reader, "set_resolution_aware_s_min"):
        pipeline._reader.set_resolution_aware_s_min(height)
        print(f"[viz] resolution-aware S_min applied for source height={height}")
    # cycle 71r (案 A, 2026-05-13): BoardRegion 自動 calibration.
    # cycle 71u (2026-05-13 副作用対策): 案 A を撤回. 誤った座標補正で
    # ベース認識精度が悪化 (= 「双方とも認識悪化」 ユーザー報告) のため.
    # 必要なら --auto-calibrate 引数で明示的に有効化する.
    # 巻き戻し不要 (= cap は最初から再開).
    pass
    # サイクル4: 動画別 HSV ranges DB を起動時に inject
    if args.hsv_state is not None:
        try:
            import json as _json
            with args.hsv_state.open("r", encoding="utf-8") as _f:
                _state = _json.load(_f)
            _ranges = _state.get("per_video_ranges", {})
            _ranges_int = {
                int(k): tuple(int(x) for x in v) for k, v in _ranges.items()
            }
            from src.hybrid_classifier import HybridClassifier
            _hc = pipeline._reader._classifier
            if (
                isinstance(_hc, HybridClassifier)
                and hasattr(_hc._hsv, "set_color_ranges_from_simple")
                and _ranges_int
            ):
                _hc._hsv.set_color_ranges_from_simple(_ranges_int)
                # 循環 Hue 補完 guard: 赤等の2範囲定義色で per_video inject が
                # 片側 (H=0-13) を欠落させていないか確認し不足分を追加する。
                _ensure_circular_ranges_guard(_hc._hsv)
                # 2026-05-11 サイクル63 #6: 低解像度では pre-inject 後も
                # OnlineHsv の学習を継続させる (= merged_default は generic
                # で動画固有の調整が必要なため). 720p+ は DB が動画別で
                # tight なので従来通り suppress.
                if pipeline._online_hsv is not None and height >= 720:
                    pipeline._online_hsv_injected = True
                print(
                    f"[viz] HSV pre-inject from {args.hsv_state}: "
                    f"{len(_ranges_int)} colors "
                    f"(online_hsv {'suppressed' if height >= 720 else 'continues'})",
                )
        except Exception as _e:
            print(f"[viz] HSV pre-inject failed: {_e}", file=sys.stderr)

    # OjamaAccountingTracker: アーキ案A (連鎖終了イベント駆動) でお邪魔管理
    # on_state_transition + on_tsumo_settled + get_snapshot が新 API
    print("[viz] ojama overlay: OjamaAccountingTracker アーキ案A (on_state_transition 駆動)")
    print("[viz] OjamaWarningDetector は使用しない (誤読源として排除)")

    _ojama_tracker = OjamaAccountingTracker()
    _ojama_tracker.reset()
    _last_snap: OjamaAccountSnapshot | None = None
    # 前フレームの state: on_state_transition の prev 引数用
    _prev_acct_state_p1: BoardState = BoardState.MENU
    _prev_acct_state_p2: BoardState = BoardState.MENU
    # ojama_accounting JSONL 出力
    _ojama_accounting_fp = None
    if args.dump_ojama_accounting is not None:
        args.dump_ojama_accounting.parent.mkdir(parents=True, exist_ok=True)
        _ojama_accounting_fp = open(args.dump_ojama_accounting, "w", encoding="utf-8")
        print(f"[viz] ojama_accounting log → {args.dump_ojama_accounting}")

    sample_interval_frames = max(1, int(round(args.sample_interval * fps)))
    last_p1_state = BoardState.MENU
    last_p2_state = BoardState.MENU
    # YouTubeデモ素材用 (2026-08-07): STABLE 以外はセル文字を隠す表示モード。
    overlay_stable_only = bool(getattr(args, "overlay_stable_only", False))
    # 幽霊セル対策 (2026-08-07): chain/ojama_fall 脱出直後の描画ホールド (フレーム数)。
    overlay_transition_hold_frames = max(
        0, int(getattr(args, "overlay_transition_hold_frames", 0) or 0)
    )
    # ホールド判定対象の state (連鎖中・おじゃま落下中): 抜けた瞬間がホールド起点
    _TRANSITION_HOLD_STATES = frozenset({BoardState.CHAIN, BoardState.OJAMA_FALL})
    # 各プレイヤーが最後に _TRANSITION_HOLD_STATES から抜けた frame index (未発生時 None)
    last_p1_transition_exit_frame: int | None = None
    last_p2_transition_exit_frame: int | None = None
    # 評価で使う盤面 = STABLE 時の confirmed_board を凍結保持
    # NON-STABLE (chain/tsumo_fall/ojama_fall/effect) では更新せず、前回 STABLE 値維持
    last_p1_eval_board: Board | None = None
    last_p2_eval_board: Board | None = None
    # 総合 overlay 用: score / next の最新値を保持
    last_p1_score: int | None = None
    last_p2_score: int | None = None
    last_p1_next: tuple[int, int] | None = None
    last_p2_next: tuple[int, int] | None = None
    last_p1_dnext: tuple[int, int] | None = None
    last_p2_dnext: tuple[int, int] | None = None
    # score OCR 差分由来の ojama 送出量トラッカー (サマリ出力用・フェード表示用)
    # OjamaAccountingTracker が本体。last_p*_ojama_sent はイベント単位のサマリ用に残す。
    last_p1_ojama_sent: int = 0   # 1P が受けたojama (最新イベント)
    last_p2_ojama_sent: int = 0   # 2P が受けたojama (最新イベント)
    last_p1_ojama_event_sec: float = -1.0   # 1P の最後のojama受け取り時刻
    last_p2_ojama_event_sec: float = -1.0   # 2P の最後のojama受け取り時刻
    # 隠し段確率 overlay 用: 最新の prob_board を保持 (STABLE 時のみ更新)
    last_p1_prob_board: ProbabilisticBoard | None = None
    last_p2_prob_board: ProbabilisticBoard | None = None
    _hidden_row_nonnull_count: int = 0  # smoke 確認用: non-None 取得回数
    # cycle 33: board log JSONL 出力 (= 強化アナリスト用)
    board_log_fp = None
    if args.dump_board_log is not None:
        args.dump_board_log.parent.mkdir(parents=True, exist_ok=True)
        board_log_fp = open(args.dump_board_log, "w", encoding="utf-8")
        print(f"[viz] board log → {args.dump_board_log}")
    # Q1 全力施策: 詳細 board log JSONL (raw_hsv / bg_fp_distance / tier1 等)
    board_log_detail_fp = None
    if args.dump_board_log_detailed is not None:
        args.dump_board_log_detailed.parent.mkdir(parents=True, exist_ok=True)
        board_log_detail_fp = open(args.dump_board_log_detailed, "w", encoding="utf-8")
        print(f"[viz] detailed board log → {args.dump_board_log_detailed}")
    # 詳細 dump 用: prev frame の confirmed_board (physics_fix diff 計算用)
    prev_p1_confirmed: Board | None = None
    prev_p2_confirmed: Board | None = None

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        # 認識実行 (sample_interval_frames ごと)
        if fi % sample_interval_frames == 0:
            result = pipeline.update(fi, t_sec, frame)
            # 幽霊セル対策: 更新前の state (= 遷移元) を保持してから上書きする
            _prev_hold_state_p1 = last_p1_state
            _prev_hold_state_p2 = last_p2_state
            last_p1_state = result.p1.state
            last_p2_state = result.p2.state
            # chain/ojama_fall から抜けた瞬間を記録 (ホールド起点)。
            # 再突入時はカウンタをリセット (次に抜けた時点から改めてホールド)。
            if (_prev_hold_state_p1 in _TRANSITION_HOLD_STATES
                    and last_p1_state not in _TRANSITION_HOLD_STATES):
                last_p1_transition_exit_frame = fi
            elif last_p1_state in _TRANSITION_HOLD_STATES:
                last_p1_transition_exit_frame = None
            if (_prev_hold_state_p2 in _TRANSITION_HOLD_STATES
                    and last_p2_state not in _TRANSITION_HOLD_STATES):
                last_p2_transition_exit_frame = fi
            elif last_p2_state in _TRANSITION_HOLD_STATES:
                last_p2_transition_exit_frame = None
            # STABLE 時のみ確定盤面を取得 (= indicator 評価で使うのと同じ条件)
            if (result.p1.state == BoardState.STABLE
                    and result.p1.confirmed_board is not None):
                last_p1_eval_board = result.p1.confirmed_board
            if (result.p2.state == BoardState.STABLE
                    and result.p2.confirmed_board is not None):
                last_p2_eval_board = result.p2.confirmed_board
            # cycle 33: 各 frame の認識結果を JSONL に保存
            if board_log_fp is not None:
                import json as _json
                # erasure_alerts: SideResult に乗った「当該 frame での新規 alert」
                # backwards compat: キーなし既存 JSONL を読む側は .get('p1_erasure_alerts', []) で対処。
                p1_ea = result.p1.erasure_alerts if result.p1.erasure_alerts is not None else []
                p2_ea = result.p2.erasure_alerts if result.p2.erasure_alerts is not None else []
                entry = {
                    "frame_idx": fi,
                    "t_sec": t_sec,
                    "p1_state": result.p1.state.value,
                    "p2_state": result.p2.state.value,
                    "p1_confirmed": (
                        result.p1.confirmed_board.to_dict()["grid"]
                        if result.p1.confirmed_board is not None else None
                    ),
                    "p2_confirmed": (
                        result.p2.confirmed_board.to_dict()["grid"]
                        if result.p2.confirmed_board is not None else None
                    ),
                    # T4 PuyoErasureMonitor: 当該 frame で検出した新規 alert [(row, col), ...]
                    "p1_erasure_alerts": [list(a) for a in p1_ea],
                    "p2_erasure_alerts": [list(a) for a in p2_ea],
                }
                board_log_fp.write(_json.dumps(entry, ensure_ascii=False) + "\n")
            # Q1 全力施策: 詳細 JSONL 出力 (raw_hsv / bg_fp_distance 等)
            if board_log_detail_fp is not None:
                import json as _json2
                try:
                    detail_entry = _build_detailed_log_entry(
                        fi, t_sec, result, frame, pipeline,
                        prev_p1_confirmed, prev_p2_confirmed,
                    )
                except Exception as _detail_err:
                    # 詳細 dump 失敗は本流に影響させない
                    detail_entry = {
                        "frame_idx": fi, "t_sec": t_sec,
                        "error": str(_detail_err),
                    }
                board_log_detail_fp.write(
                    _json2.dumps(detail_entry, ensure_ascii=False) + "\n"
                )
            # prev_confirmed を更新 (次 frame の physics_fix diff 計算用)
            prev_p1_confirmed = (
                result.p1.confirmed_board.copy()
                if result.p1.confirmed_board is not None else prev_p1_confirmed
            )
            prev_p2_confirmed = (
                result.p2.confirmed_board.copy()
                if result.p2.confirmed_board is not None else prev_p2_confirmed
            )
            # 総合 overlay 用: score / next / dnext を毎サンプルフレームで更新
            if result.p1.score is not None:
                last_p1_score = result.p1.score
            if result.p2.score is not None:
                last_p2_score = result.p2.score
            if result.p1.next_pair is not None:
                last_p1_next = result.p1.next_pair
                last_p1_dnext = result.p1.dnext_pair
            if result.p2.next_pair is not None:
                last_p2_next = result.p2.next_pair
                last_p2_dnext = result.p2.dnext_pair
            # 隠し段確率 overlay: STABLE 時のみ prob_board を更新
            if result.p1.prob_board is not None:
                last_p1_prob_board = result.p1.prob_board
                _hidden_row_nonnull_count += 1
            if result.p2.prob_board is not None:
                last_p2_prob_board = result.p2.prob_board
            # =============================================
            # OjamaAccountingTracker 駆動 (アーキ案A: on_state_transition)
            # =============================================
            # on_state_transition: state 遷移を通知。
            #   連鎖開始(STABLE→CHAIN): score スナップ
            #   連鎖終了(CHAIN/GRAVITY_SETTLE→STABLE): 一括換算 + 相殺
            #   MENU 遷移 / score 大幅減少: 自動 reset
            _curr_p1_state = result.p1.state
            _curr_p2_state = result.p2.state
            # 連鎖終了 drain 用: 非STABLE → STABLE 立ち上がりエッジ
            _tsumo_settled_p1 = (
                _prev_acct_state_p1 != BoardState.STABLE
                and _curr_p1_state == BoardState.STABLE
                and _prev_acct_state_p1 == BoardState.TSUMO_FALL
            )
            _tsumo_settled_p2 = (
                _prev_acct_state_p2 != BoardState.STABLE
                and _curr_p2_state == BoardState.STABLE
                and _prev_acct_state_p2 == BoardState.TSUMO_FALL
            )
            # on_state_transition で 1P/2P それぞれ通知
            _ojama_tracker.on_state_transition(
                "p1", _prev_acct_state_p1, _curr_p1_state,
                result.p1.score, t_sec,
            )
            _ojama_tracker.on_state_transition(
                "p2", _prev_acct_state_p2, _curr_p2_state,
                result.p2.score, t_sec,
            )
            # TSUMO_FALL → STABLE で予告 drain
            if _tsumo_settled_p1:
                _ojama_tracker.on_tsumo_settled("p1", t_sec)
            if _tsumo_settled_p2:
                _ojama_tracker.on_tsumo_settled("p2", t_sec)
            _prev_acct_state_p1 = _curr_p1_state
            _prev_acct_state_p2 = _curr_p2_state
            # スナップショット取得
            _last_snap = _ojama_tracker.get_snapshot(t_sec)

            # 会計 JSONL 出力 (アーキ案A 検証フィールド追加)
            if _ojama_accounting_fp is not None:
                import json as _jacct
                _acct_entry = {
                    "t_sec": t_sec,
                    "p1_state": result.p1.state.value,
                    "p2_state": result.p2.state.value,
                    "score_p1": result.p1.score,
                    "score_p2": result.p2.score,
                    # 予告個数(画面と一致させる目標値)
                    "forecast_p1": _last_snap.forecast_p1,
                    "forecast_p2": _last_snap.forecast_p2,
                    # 後方互換フィールド(= forecast と同値)
                    "pending_p1": _last_snap.pending_p1,
                    "pending_p2": _last_snap.pending_p2,
                    "pending_p1_capped": _last_snap.pending_p1_capped,
                    "pending_p2_capped": _last_snap.pending_p2_capped,
                    "offboard_p1": _last_snap.offboard_p1,
                    "offboard_p2": _last_snap.offboard_p2,
                    "net_balance": _last_snap.net_ojama_balance,
                    "net_balance_capped": _last_snap.net_balance_capped,
                    "total_dropped_p1": _last_snap.total_dropped_to_p1,
                    "total_dropped_p2": _last_snap.total_dropped_to_p2,
                    "total_generated_p1": _last_snap.total_generated_by_p1,
                    "total_generated_p2": _last_snap.total_generated_by_p2,
                    "total_offset_p1": _last_snap.total_offset_by_p1,
                    "total_offset_p2": _last_snap.total_offset_by_p2,
                    "confidence": _last_snap.confidence,
                    # 検証フィールド (アーキ案A)
                    "chain_total_score_p1": _last_snap.chain_total_score_p1,
                    "chain_total_score_p2": _last_snap.chain_total_score_p2,
                    "chain_end_triggered_p1": _last_snap.chain_end_triggered_p1,
                    "chain_end_triggered_p2": _last_snap.chain_end_triggered_p2,
                    "score_at_chain_start_p1": _last_snap.score_at_chain_start_p1,
                    "score_at_chain_start_p2": _last_snap.score_at_chain_start_p2,
                    "tsumo_settled_p1": _tsumo_settled_p1,
                    "tsumo_settled_p2": _tsumo_settled_p2,
                }
                _ojama_accounting_fp.write(
                    _jacct.dumps(_acct_entry, ensure_ascii=False) + "\n"
                )

        # 描画用エイリアス
        last_p1_board = last_p1_eval_board
        last_p2_board = last_p2_eval_board

        # 描画: 6 要素 (フィールド状態・ぷよ色・score・next・OJ送出・隠し段)
        # --overlay-stable-only / --overlay-show-states: 各プレイヤー独立判定で
        # 対象外 state のセル文字を隠す (状態ラベル・盤面枠色は維持、2026-08-07 追加)。
        # --overlay-transition-hold-frames: chain/ojama_fall 脱出直後 N フレームは
        # 上記判定が True でも AND で追加抑制する (幽霊セル対策、2026-08-07 追加)。
        _p1_in_hold = (
            overlay_transition_hold_frames > 0
            and last_p1_transition_exit_frame is not None
            and (fi - last_p1_transition_exit_frame) < overlay_transition_hold_frames
        )
        _p2_in_hold = (
            overlay_transition_hold_frames > 0
            and last_p2_transition_exit_frame is not None
            and (fi - last_p2_transition_exit_frame) < overlay_transition_hold_frames
        )
        if (should_draw_cell_overlay(last_p1_state, overlay_stable_only, overlay_show_states)
                and not _p1_in_hold):
            draw_cell_overlay(frame, last_p1_board, P1_ROI_X, P1_ROI_Y)
        if (should_draw_cell_overlay(last_p2_state, overlay_stable_only, overlay_show_states)
                and not _p2_in_hold):
            draw_cell_overlay(frame, last_p2_board, P2_ROI_X, P2_ROI_Y)
        draw_state_label(
            frame, last_p1_state, P1_ROI_X, P1_ROI_Y,
            score=last_p1_score or 0, label_prefix="1P:",
        )
        draw_state_label(
            frame, last_p2_state, P2_ROI_X, P2_ROI_Y,
            score=last_p2_score or 0, label_prefix="2P:",
        )
        draw_next_overlay(
            frame, last_p1_next, last_p1_dnext, P1_ROI_X, P1_ROI_Y, label_prefix="1P:",
        )
        draw_next_overlay(
            frame, last_p2_next, last_p2_dnext, P2_ROI_X, P2_ROI_Y, label_prefix="2P:",
        )
        # 会計モデルの小さなデバッグテキスト (pend/net/drop/OB) は 2026-06-10 で
        # 本番 overlay から除去 (日本語?化・雑然) し、下の大きな新パネルに一本化。
        # 関数 draw_ojama_accounting_overlay は後方互換のため定義のみ残置。
        # 予告お邪魔 直感UI (2026-06-10 刷新・拡大): 大きな数字 + 単位アイコン分解 + 優勢バー
        draw_ojama_forecast_panel(
            frame, _last_snap, "1P", P1_ROI_X, P1_ROI_Y,
        )
        draw_ojama_forecast_panel(
            frame, _last_snap, "2P", P2_ROI_X, P2_ROI_Y,
        )
        draw_ojama_advantage_bar(frame, _last_snap)
        # 第6要素: 隠し段 (row 0) 確率 overlay + 画面外 O 個数
        # offboard は OjamaAccountingTracker の会計値 (pending - 72) から取得
        # 試合境界で pending がresetされるため O 表示も自動的に 0 に戻る
        _offboard_p1 = _last_snap.offboard_p1 if _last_snap is not None else 0
        _offboard_p2 = _last_snap.offboard_p2 if _last_snap is not None else 0
        draw_hidden_row_overlay(
            frame, last_p1_prob_board, P1_ROI_X, P1_ROI_Y,
            offboard_ojama=_offboard_p1,
        )
        draw_hidden_row_overlay(
            frame, last_p2_prob_board, P2_ROI_X, P2_ROI_Y,
            offboard_ojama=_offboard_p2,
        )
        draw_global_info(
            frame, fi, t_sec, last_p1_state, last_p2_state,
            p1_score=last_p1_score, p2_score=last_p2_score,
        )

        writer.write(frame)
        if fi % 100 == 0:
            print(f"  [progress] {fi}/{n_frames} ({fi*100/max(n_frames,1):.1f}%) "
                  f"1P={last_p1_state.value} 2P={last_p2_state.value}")

    cap.release()
    writer.release()
    # 隠し段確率 overlay: 発火回数をログ出力 (smoke 確認用)
    print(
        f"[hidden_row] prob_board non-None 取得回数: {_hidden_row_nonnull_count} frames "
        f"({'発火あり' if _hidden_row_nonnull_count > 0 else '発火なし (隠し段推論が起動しなかった)'})"
    )
    # ojama_sent サマリ (OjamaAccountingTracker 由来)
    if _last_snap is not None:
        print(
            f"[ojama_score] 1P が受けた OJ累積(generated by 2P)={_last_snap.total_generated_by_p2}個 "
            f"(最終 event={last_p1_ojama_sent}個, t={last_p1_ojama_event_sec:.2f}s)"
        )
        print(
            f"[ojama_score] 2P が受けた OJ累積(generated by 1P)={_last_snap.total_generated_by_p1}個 "
            f"(最終 event={last_p2_ojama_sent}個, t={last_p2_ojama_event_sec:.2f}s)"
        )
    else:
        print(
            f"[ojama_score] 1P が受けた OJ: 最終 event={last_p1_ojama_sent}個, "
            f"t={last_p1_ojama_event_sec:.2f}s"
        )
        print(
            f"[ojama_score] 2P が受けた OJ: 最終 event={last_p2_ojama_sent}個, "
            f"t={last_p2_ojama_event_sec:.2f}s"
        )
    if board_log_fp is not None:
        board_log_fp.close()
        print(f"[done] board log saved")
    if board_log_detail_fp is not None:
        board_log_detail_fp.close()
        print(f"[done] detailed board log saved → {args.dump_board_log_detailed}")
    if _ojama_accounting_fp is not None:
        _ojama_accounting_fp.close()
        print(f"[done] ojama_accounting log saved → {args.dump_ojama_accounting}")
    # 会計最終 snapshot サマリ
    if _last_snap is not None:
        print(
            f"[acct_final] pending 1P={_last_snap.pending_p1} 2P={_last_snap.pending_p2} "
            f"net={_last_snap.net_ojama_balance:+d} conf={_last_snap.confidence:.2f}"
        )
        print(
            f"[acct_final] capped  1P={_last_snap.pending_p1_capped} "
            f"2P={_last_snap.pending_p2_capped} "
            f"net_capped={_last_snap.net_balance_capped:+d}"
        )
        print(
            f"[acct_final] generated 1P={_last_snap.total_generated_by_p1} "
            f"2P={_last_snap.total_generated_by_p2} "
            f"drop 1P={_last_snap.total_dropped_to_p1} 2P={_last_snap.total_dropped_to_p2}"
        )
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
