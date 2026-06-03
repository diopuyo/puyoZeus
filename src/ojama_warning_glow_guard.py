"""予告おじゃま発光ガード (OjamaWarningGlowGuard).

相手の連鎖による予告おじゃま演出で盤面上部に発生する多色高輝度アニメが
黄ぷよに重なり黄(4)→おじゃま(9)誤認を引き起こす問題を防ぐ。

実測 calibration (2026-06-04, main 測定):
  通常 STABLE:       V_high_ratio の median = 0.048
  予告発光中 (v89 t=68-72s): V_high_ratio = 0.22〜0.44
  → ratio ≥ 0.20 で綺麗に分離

設計方針:
  - stateless 本体 (compute_glow_score, update_glow_state, apply_glow_guard)
  - state は GlowGuardState dataclass で外部 wrapper (pipeline) が保持する
  - STABLE 中発光を主戦場とし、CHAIN 中の保護は既存機構に委ねる
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA, Board,
)
from src.image_reader import BoardRegion

# ============================
# 定数 (マジックナンバー禁止)
# ============================

# 盤面上部 ROI の行数 (予告おじゃま演出が影響する行)
GLOW_ROI_ROW_COUNT: int = 5

# V チャンネル (明度) 高輝度画素の閾値 (V ≥ この値を「発光画素」とみなす)
# 実画素: 通常 STABLE では V_high_ratio ≈ 0.05、発光中 0.22〜0.44
V_HIGH_THRESHOLD: int = 220

# glow_score 正規化の下限: ratio がこの値以下なら glow_score=0
GLOW_RATIO_LOW: float = 0.12

# glow_score 正規化の上限: ratio がこの値以上なら glow_score=1
GLOW_RATIO_HIGH: float = 0.28

# 発光検知の閾値: glow_score ≥ この値を「発光中」と判定
# = ratio ≥ 0.20 相当 (実測分離点)
GLOW_DETECTION_THRESHOLD: float = 0.5

# 発光 ON 判定に必要な連続フレーム数 (単発ノイズ除外)
GLOW_CONSEC_MIN: int = 2

# 発光 OFF 判定に必要な連続フレーム数 (即解除)
GLOW_RELEASE_CONSEC: int = 2

# 発光保護の最大保持フレーム数 (強制解除上限 @ 60fps = 1秒)
GLOW_MAX_HOLD_FRAMES: int = 60


# ============================
# state dataclass
# ============================


@dataclass
class GlowGuardState:
    """予告おじゃま発光ガードの状態.

    pipeline wrapper が 1P/2P 別に保持し、
    update_glow_state に渡してフレーム毎に更新する。
    """

    # 発光保護中か (True = confirmed を frozen_board で保護)
    glow_active: bool = False
    # glow_score ≥ GLOW_DETECTION_THRESHOLD が連続した回数
    consec_on: int = 0
    # glow_score < GLOW_DETECTION_THRESHOLD が連続した回数
    consec_off: int = 0
    # 発光 ON になってからの保持フレーム数 (MAX_HOLD 超で強制解除)
    hold_frame_count: int = 0
    # 発光 ON になる直前の confirmed_board (保護の基準盤面)
    # 発光 OFF 中のみ現 confirmed で更新する
    frozen_board: Board | None = None


# ============================
# stateless 本体
# ============================


def compute_glow_score(frame_bgr: np.ndarray, region: BoardRegion) -> float:
    """盤面上部 ROI の発光スコア (0〜1) を計算する.

    1P/2P の BoardRegion から上部 GLOW_ROI_ROW_COUNT 行に相当する
    ピクセル領域を切り出し、HSV の V チャンネルで高輝度画素比率を
    計算して 0〜1 に正規化した glow_score を返す。

    Args:
        frame_bgr: 1920×1080 BGR フレーム画像。
        region: 対象サイドの BoardRegion (DEFAULT_P1/P2_REGION)。

    Returns:
        glow_score: 0.0 (通常) 〜 1.0 (最大発光)。
        計算失敗・ROI 空の場合は 0.0 を返す。
    """
    try:
        # 上部 GLOW_ROI_ROW_COUNT 行のピクセル高さを算出
        roi_h = int(region.cell_height * GLOW_ROI_ROW_COUNT)
        roi_h = max(1, roi_h)
        x1 = region.x
        y1 = region.y
        x2 = x1 + region.width
        y2 = y1 + roi_h
        # フレーム境界クリップ
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(y1 + 1, min(y2, h_img))
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
        total_px = v_channel.size
        if total_px == 0:
            return 0.0
        high_count = float(np.sum(v_channel >= V_HIGH_THRESHOLD))
        ratio = high_count / total_px
        # 0〜1 に正規化
        if ratio <= GLOW_RATIO_LOW:
            return 0.0
        if ratio >= GLOW_RATIO_HIGH:
            return 1.0
        return float((ratio - GLOW_RATIO_LOW) / (GLOW_RATIO_HIGH - GLOW_RATIO_LOW))
    except Exception:
        return 0.0


def update_glow_state(
    state: GlowGuardState,
    glow_score: float,
    frame_idx: int,
) -> bool:
    """発光 state を 1 フレーム分更新し、「発光保護中か」を返す.

    ON/OFF の連続フレーム要件と最大保持フレーム数による強制解除を処理する。
    state は in-place で更新される (stateless 処理本体、state 保持は外側)。

    Args:
        state: GlowGuardState (in-place 更新)。
        glow_score: 現フレームの glow_score (0〜1)。
        frame_idx: 現在のフレームインデックス (デバッグ用、内部では未使用)。

    Returns:
        is_glow_active: True = 発光保護中。
    """
    is_high = glow_score >= GLOW_DETECTION_THRESHOLD

    if is_high:
        state.consec_on += 1
        state.consec_off = 0
    else:
        state.consec_off += 1
        state.consec_on = 0

    if not state.glow_active:
        # OFF → ON 遷移: GLOW_CONSEC_MIN 連続で発火
        if state.consec_on >= GLOW_CONSEC_MIN:
            state.glow_active = True
            state.hold_frame_count = 0
    else:
        # ON 中: カウンタ更新
        state.hold_frame_count += 1
        # OFF 方向: GLOW_RELEASE_CONSEC 連続で即解除
        if state.consec_off >= GLOW_RELEASE_CONSEC:
            state.glow_active = False
            state.consec_on = 0
            state.hold_frame_count = 0
        # 上限フレーム超: 強制解除
        elif state.hold_frame_count >= GLOW_MAX_HOLD_FRAMES:
            state.glow_active = False
            state.consec_on = 0
            state.hold_frame_count = 0

    return state.glow_active


def apply_glow_guard(
    confirmed: Board,
    state: GlowGuardState,
    is_glow_active: bool,
) -> Board:
    """発光保護を適用した confirmed_board を返す (v2: ターゲット型).

    v2 の適用ルール (正常セルには一切触れない):
      発光中 (is_glow_active=True) かつ frozen_board が存在する場合:
        - confirmed が おじゃま(COLOR_OJAMA=9) かつ
          frozen が有色 (非空・非おじゃま・非UNKNOWN) のセルのみ → frozen 色に復元。
          (= 発光演出が色ぷよをおじゃまに誤らせた場合のみピンポイント修正)
        - それ以外のセルは confirmed をそのまま使用 (不触)。

    v1 との差分:
      - 「frozen 有色なら全部上書き」を廃止。おじゃま誤認セルのみ復元する。
      - 「frozen 空 + confirmed 有色 → UNKNOWN 留保」を廃止。
        過剰発火による corruption 増を防ぐ。

    発光 OFF 中: confirmed をそのまま返す (変更なし)。

    Args:
        confirmed: 現フレームの確定盤面 (上書き元)。
        state: GlowGuardState (frozen_board 参照用、更新なし)。
        is_glow_active: update_glow_state の戻り値。

    Returns:
        保護を適用した Board (変更なし or おじゃま誤認セルのみ復元済み)。
    """
    if not is_glow_active or state.frozen_board is None:
        return confirmed

    result = confirmed.copy()
    frozen = state.frozen_board
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            frozen_v = int(frozen.get(r, c))
            conf_v = int(confirmed.get(r, c))
            # 「confirmed=おじゃま かつ frozen=有色(非空・非おじゃま・非UNKNOWN)」のみ復元
            # = 発光で色ぷよがおじゃまに誤認された場合だけをピンポイントで打ち消す
            conf_is_ojama = conf_v == COLOR_OJAMA
            frozen_is_colored = frozen_v not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN)
            if conf_is_ojama and frozen_is_colored:
                result.set(r, c, frozen_v)
            # それ以外 (正常色・空・元々おじゃま・新規ぷよ): 不触 (confirmed のまま)
    return result
