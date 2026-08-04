"""視覚エフェクトグロー検出器 (案B, 2026-08-04)。

エフェクト時間ゲート (enable_effect_gate) の4条件AND拡張
(enable_effect_visual_gate) 用の視覚判定コンポーネント。相手の連鎖1リンク
毎に約0.2秒発生する「予告おじゃま送付エフェクト」は自盤面上段 (row1-3) を
高輝度画素で覆う (memory `project_full_board_error_taxonomy_2026-08-02`)。
これをセル単位の HSV V チャンネル統計で検出する。

較正出典 (本モジュールの定数の根拠):
    data/verify/effect_detector_calibration_v3_2026-08-04/calibration_report_v3.md §3
    bright_ratio_max: n_pos=17 / n_neg=87、AUC=0.811、zero_fp_threshold=0.97
    (v1+v3統合ラベル136枚較正、誤発火0件を保証する動作点)。
    ロジック自体は scripts/study_effect_signature_2026-08-03.py の
    compute_cell_features / scripts/calibrate_effect_detector_v3.py の
    band_max を stateless に移植したもの (調査専用スクリプトからの本実装化)。

設計: 完全 stateless (観測指標は state を持たない、CLAUDE.md 規約)。
呼び出し側 (RecognitionPipeline) が enable_effect_visual_gate 経由で
既存3条件 (時間窓 AND not自連鎖中 AND not全消しラッチ) が真の場合のみ
本モジュールを評価する (視覚判定はコスト高いため遅延評価する設計)。
"""
from __future__ import annotations

import cv2
import numpy as np

from src.board import BOARD_COLS
from src.board_state_machine import EFFECT_GATE_TOP_ROWS
from src.image_reader import BoardRegion

# 高輝度率の V チャンネル閾値 (出典: scripts/study_effect_signature_2026-08-03.py
# の BRIGHT_V_THRESHOLD、_diag_effect_glow_hsv_2026-07-31.py の「高輝度率」指標と
# 同一定義)。V >= この値の画素比率を bright_ratio とする。
BRIGHT_V_THRESHOLD: int = 230

# 較正で確定したゼロFP動作点 (出典: 上記 docstring 記載の calibration_report_v3.md §3)。
# 窓内 (行帯) の bright_ratio 最大値がこれを超えたらエフェクトグローありと判定する。
EFFECT_BRIGHT_RATIO_MAX_THRESHOLD: float = 0.97


def compute_cell_bright_ratio(patch_bgr: np.ndarray) -> float:
    """1セル切り出し画像 (BGR) の高輝度画素比率を計算する (stateless純関数)。

    Args:
        patch_bgr: セル領域の切り出し画像 (BGR, uint8)。

    Returns:
        V チャンネルが BRIGHT_V_THRESHOLD 以上の画素の比率 (0.0〜1.0)。
    """
    v_channel = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(np.mean(v_channel >= BRIGHT_V_THRESHOLD))


def compute_effect_glow_score(
    frame_bgr: np.ndarray,
    region: BoardRegion,
    rows: "frozenset[int]" = EFFECT_GATE_TOP_ROWS,
) -> float:
    """指定行帯 (既定 row1-3) 内の bright_ratio 最大値を返す (stateless純関数)。

    2026-08-05 バーストガード再設計 (docs/BURST_GUARD_DESIGN_2026-08-05.md §2.1):
    `is_effect_glow_active` の閾値判定ロジックをスコア計算部分から分離した
    もの。ロジック・数値は完全に同一 (bright_ratio_max 較正済み方式、
    calibration_report_v3.md §3)。Schmitt trigger (`_update_burst_visual_gate`)
    はこの連続値スコアを入力として使う。

    Args:
        frame_bgr: 1920x1080 リサイズ済みフルフレーム (BGR)。
        region: 判定対象 side の BoardRegion (DEFAULT_P1_REGION/P2_REGION)。
        rows: 判定対象の abs_row 集合 (既定 EFFECT_GATE_TOP_ROWS={1,2,3})。

    Returns:
        窓内の全セル (rows × BOARD_COLS) の bright_ratio 最大値 (0.0〜1.0)。
    """
    max_ratio = 0.0
    for row in rows:
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            patch = frame_bgr[y1:y2, x1:x2]
            ratio = compute_cell_bright_ratio(patch)
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio


def is_effect_glow_active(
    frame_bgr: np.ndarray,
    region: BoardRegion,
    rows: "frozenset[int]" = EFFECT_GATE_TOP_ROWS,
    threshold: float = EFFECT_BRIGHT_RATIO_MAX_THRESHOLD,
) -> bool:
    """指定行帯 (既定 row1-3) 内で高輝度バーストが観測されているか判定する。

    窓内の全セル (rows × BOARD_COLS) の bright_ratio 最大値が threshold を
    超えたら True。単セル最大値方式 (較正レポートの `bright_ratio_max` と
    同一集約方法、AUC=0.811・zero_fp_threshold=0.97 の較正済み動作点)。

    2026-08-05: 内部実装を `compute_effect_glow_score` 呼び出しに変更した
    (バーストガード再設計 §2.1、薄いラッパー化)。戻り値・数値は
    完全に bit-identical (score計算ロジックを1文字も変えていない)。

    Args:
        frame_bgr: 1920x1080 リサイズ済みフルフレーム (BGR)。
        region: 判定対象 side の BoardRegion (DEFAULT_P1_REGION/P2_REGION)。
        rows: 判定対象の abs_row 集合 (既定 EFFECT_GATE_TOP_ROWS={1,2,3})。
        threshold: bright_ratio_max の判定閾値。

    Returns:
        True ならエフェクトグロー (予告おじゃま送付バースト) を検出。
    """
    return compute_effect_glow_score(frame_bgr, region, rows) > threshold


__all__ = [
    "BRIGHT_V_THRESHOLD",
    "EFFECT_BRIGHT_RATIO_MAX_THRESHOLD",
    "compute_cell_bright_ratio",
    "compute_effect_glow_score",
    "is_effect_glow_active",
]
