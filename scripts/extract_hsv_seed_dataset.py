"""HSV-only 信頼判定で seed dataset を抽出する (Phase I CNN mode collapse 突破用、 cycle_14)。

設計:
    visualize_recognition.py と同じ pipeline を回しつつ、
    STABLE 中の confirmed_board と HSV-only 判定 (HybridClassifier.hsv_grid) が
    一致 + S/V 高い cell の patch を PseudoLabelSample 形式で保存する。

    CNN を信用せず HSV 単独で確信度高い cell のみ採用するため、
    mode collapse model でも anchor seed を集められる。

出力:
    data/pseudo_labels_hsv_seed/<vid>/cell.jsonl (PseudoLabelSample 形式)
    → phase_i_fine_tune.py --component cell_color --store-root で fine-tune 可能
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

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    HIDDEN_ROWS,
)
from src.board_state_machine import BoardState, NON_STABLE_STATES
from src.hybrid_classifier import HybridClassifier
from src.patch_classifier import OjamaShapeGate
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.recognition_pipeline import RecognitionPipeline
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL, PseudoLabelSample

TRAINABLE_COLORS: tuple[int, ...] = (
    COLOR_EMPTY,  # cycle 32e (2026-05-19): EMPTY 採取追加
    COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)

# cycle 32g (2026-05-19): EMPTY 採取条件拡張。 cycle 32e で「キャラ背景紫誤認」 が
# 発生したため、 試合中のキャラ背景も EMPTY として採取できるよう緩和。
# - S 上限: 90→70 に厳格化 (= 紫/青の高彩度 puyo を誤採取しない)
# - V 上限: 170→200 に緩和 (= 明るいキャラ背景も採取)
# - bg_fp 距離上限: 35→90 に緩和 (= 試合開始時の盤面 bg から離れた cell も採取)
EMPTY_S_MAX: float = 70.0
EMPTY_V_MAX: float = 200.0
EMPTY_BG_FP_MAX_DISTANCE: float = 90.0

# 品質閾値 (HSV-only 信頼判定後の最終フィルタ)
MIN_S_MEDIAN: float = 80.0   # 鮮やかさ閾値
MIN_V_MEDIAN: float = 50.0   # 明度閾値
# cycle 32 (2026-05-19): G1 = bg_fp 距離 check。 採取候補 cell が背景指紋と
# 近すぎる場合 (= 距離 < この閾値) は採取拒否。 青背景バイアス元凶対策。
# ユーザー目視で blue 6/20 が背景、 red 11/20、 yellow 20/20、 ojama 19/20 が
# 背景由来と判明。 確実に排除するため strict 閾値 70。
BG_FP_REJECT_THRESHOLD: float = 70.0
# cycle 32 G10 (2026-05-19): ojama 採取時の V 上限。 灰色 (V 中) と白 (V 高) を
# 区別する。 試合終了時の勝敗表示 ("WIN"/"LOSE" 白文字) が ojama 採取に混入する
# 問題への対策。 V > 180 は白系として ojama 採取拒否。
OJAMA_V_MAX: float = 180.0
# cycle 32 G9 (2026-05-19): 動画末尾 N frame skip。 試合切り出し済動画でも
# 末尾に勝敗表示 (~3 秒) が含まれるため、 これを seed 採取から除外する。
# cycle 50b (2026-05-21): v86m17 で「やった」 勝利 telop 混入確認 → 6 秒に拡張。
DEFAULT_END_SKIP_FRAMES: int = 360  # = 6 秒 @ 60fps

# cycle 50 改修 2 (2026-05-21): 両側 STABLE 復帰直後 N frame skip。
# 片側 CHAIN 中は画面全体に effect mask がかかるため、 もう片側 STABLE でも
# seed cell が effect 色に染まる。 両側 STABLE 復帰から N frame 経過してから
# 採取することで chain 残響を排除する。 12 frame ~= 0.2 秒 @ 60fps。
STABLE_RECOVERY_SKIP_FRAMES: int = 12

# cycle 50 改修 4 (2026-05-21): 非 STABLE state 復帰直後 N frame skip。
# CHAIN / EFFECT / OJAMA_FALL から STABLE 復帰した直後は、 全消し telop /
# 連鎖 telop の残響が cell に重なる。 30 frame ~= 0.5 秒 @ 60fps で待機。
EFFECT_RECOVERY_SKIP_FRAMES: int = 30

# cycle 50 改修 3 (2026-05-21): 色別 H レンジ厳格化。 yellow に red 混入 60%、
# red に green 混入 30%、 blue に背景混入 20% (= ユーザー目視 10 動画 22 PNG)
# への直接対策。 各色の典型 H 中央値から ±DELTA 以内のみ採用。
SEED_H_CORE_DELTA: int = 8  # ±8° (= 16° 幅、 HSV.H 0-180 OpenCV scale)
SEED_COLOR_H_CENTER: dict[int, int] = {
    # COLOR_RED は wrap-around (H~0 / H~180) を別判定
    COLOR_YELLOW: 25,
    COLOR_GREEN: 60,
    COLOR_BLUE: 110,
    COLOR_PURPLE: 140,
}
# red は H wrap-around: H < SEED_RED_H_LOW_MAX or H > SEED_RED_H_HIGH_MIN
SEED_RED_H_LOW_MAX: int = 8
SEED_RED_H_HIGH_MIN: int = 172

# cycle 50 改修 1 (2026-05-21): ojama 採取復活 (= cycle 32 撤回連動)
# OjamaShapeGate で文字エフェクト / 勝敗 telop と区別。 デフォルト OFF、
# --include-ojama 指定時のみ ON で seed 採取 path に組み込まれる。
# cycle 58 (2026-05-23): --ojama-relaxed で閾値緩和版 gate 使用可能。
_OJAMA_SHAPE_GATE = OjamaShapeGate()
# cycle 58 (2026-05-23): _OJAMA_NO_GATE=True で gate skip、 V/S check のみで品質担保。
_OJAMA_NO_GATE: bool = False


def _get_ojama_gate(relaxed: bool) -> OjamaShapeGate:
    """relaxed=True なら閾値緩和版を返す (= cycle 58 案 A)."""
    return OjamaShapeGate.relaxed() if relaxed else OjamaShapeGate()


def _pre_inject_hsv(pipeline: RecognitionPipeline, hsv_state: Path) -> None:
    """visualize_recognition.py と同じ HSV pre-inject ロジック."""
    import json as _json
    with hsv_state.open("r", encoding="utf-8") as f:
        state = _json.load(f)
    ranges = state.get("per_video_ranges", {})
    if not ranges:
        return
    ranges_int = {
        int(k): tuple(int(x) for x in v) for k, v in ranges.items()
    }
    hc = pipeline._reader._classifier
    if (
        isinstance(hc, HybridClassifier)
        and hasattr(hc._hsv, "set_color_ranges_from_simple")
    ):
        hc._hsv.set_color_ranges_from_simple(ranges_int)
        print(f"[seed] HSV pre-inject: {len(ranges_int)} colors")


def _extract_patch(
    frame: np.ndarray, region, row: int, col: int,
) -> np.ndarray | None:
    """指定 cell の BGR patch を抽出 (画像外 clip)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(y1 + 1, min(int(y2), h))
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


def _is_high_quality(patch: np.ndarray) -> bool:
    """S/V median が閾値以上 = 高信頼な puyo パッチ."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s_med = float(np.median(hsv[:, :, 1]))
    v_med = float(np.median(hsv[:, :, 2]))
    return s_med >= MIN_S_MEDIAN and v_med >= MIN_V_MEDIAN


def _is_color_h_core(patch: np.ndarray, color: int) -> bool:
    """cycle 50 改修 3: 色別 H 中央値が core レンジ内か (= 隣接色混入排除).

    ユーザー目視 10 動画レビューで yellow に red 混入 60% 等が判明。
    HSV grid 一致 check (= L210) だけでは隣接色 (= red↔yellow / red↔green
    等) を排除しきれないため、 patch 自体の H 中央値が想定範囲内か追加 check。
    OJAMA / EMPTY は対象外 (= 色 H 概念がない)。
    """
    if color == COLOR_OJAMA or color == COLOR_EMPTY:
        return True
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_med = int(np.median(hsv[:, :, 0]))
    if color == COLOR_RED:
        return h_med < SEED_RED_H_LOW_MAX or h_med > SEED_RED_H_HIGH_MIN
    center = SEED_COLOR_H_CENTER.get(color)
    if center is None:
        return True
    return abs(h_med - center) <= SEED_H_CORE_DELTA


def _is_empty_quality(
    patch: np.ndarray, bg_fp: object | None, vrow: int, col: int,
) -> bool:
    """cycle 32e: EMPTY 採取条件 = 低彩度 + 中明度 + bg_fp に近い."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s_med = float(np.median(hsv[:, :, 1]))
    v_med = float(np.median(hsv[:, :, 2]))
    if s_med > EMPTY_S_MAX:
        return False
    if v_med > EMPTY_V_MAX:
        return False
    # bg_fp 無し frame は信頼性確保のため EMPTY 採取拒否
    if bg_fp is None:
        return False
    try:
        bg_cell = bg_fp.cell_at(vrow, col)
        dist = _bg_fp_distance(patch, bg_cell)
    except Exception:
        return False
    return dist < EMPTY_BG_FP_MAX_DISTANCE


def _bg_fp_distance(patch: np.ndarray, bg_cell: object) -> float:
    """patch HSV 中央値と bg_fp cell の重み付き HSV 距離 (= background_fingerprint.py 準拠)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ph = int(np.median(hsv[:, :, 0]))
    ps = int(np.median(hsv[:, :, 1]))
    pv = int(np.median(hsv[:, :, 2]))
    bh = int(getattr(bg_cell, "h"))
    bs = int(getattr(bg_cell, "s"))
    bv = int(getattr(bg_cell, "v"))
    dh = abs(ph - bh)
    dh = min(dh, 180 - dh)
    ds = abs(ps - bs)
    dv = abs(pv - bv)
    # H_WEIGHT=0.5, S_WEIGHT=1.0, V_WEIGHT=1.0 (= background_fingerprint.py)
    return 0.5 * dh + 1.0 * ds + 1.0 * dv


def _collect_side_samples(
    frame: np.ndarray, side_result, region, side: str,
    hsv_grid: np.ndarray, video_id: str, fi: int, t_sec: float,
    counts: dict[int, int], max_per_color: int,
    bg_fp: object = None,  # cycle 32 G1: bg_fp による背景採取拒否
    skip_ojama: bool = True,  # cycle 32 I1: ojama 採取スキップ (構造的に破綻のため)
    max_empty: int = 0,  # cycle 32e: EMPTY 採取上限 (= 0 で EMPTY 採取無効)
) -> list[PseudoLabelSample]:
    """STABLE 中 1 side の高信頼 cell を抽出して PseudoLabelSample 化."""
    if side_result.state != BoardState.STABLE:
        return []
    if side_result.confirmed_board is None:
        return []
    out: list[PseudoLabelSample] = []
    rejected_by_bg = 0
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(side_result.confirmed_board.get(row, col))
            if color not in TRAINABLE_COLORS:
                continue
            # cycle 32 I1: ojama 採取スキップ
            if skip_ojama and color == COLOR_OJAMA:
                continue
            # cycle 32e: EMPTY 採取 (= 別 path、 HSV 一致 check 不要)
            if color == COLOR_EMPTY:
                if max_empty <= 0:
                    continue
                if counts.get(COLOR_EMPTY, 0) >= max_empty:
                    continue
                patch = _extract_patch(frame, region, row, col)
                if patch is None:
                    continue
                if not _is_empty_quality(patch, bg_fp, vrow, col):
                    continue
                out.append(PseudoLabelSample(
                    component=COMPONENT_CELL,
                    timestamp=t_sec,
                    input_data={"patch": patch},
                    label=COLOR_EMPTY,
                    confidence=1.0,
                    metadata={
                        "video_id": video_id, "frame_idx": fi,
                        "row": row, "col": col, "side": side,
                        "hsv_seed": True,
                    },
                ))
                counts[COLOR_EMPTY] = counts.get(COLOR_EMPTY, 0) + 1
                continue
            if counts[color] >= max_per_color:
                continue
            # cycle 58 (2026-05-23): ojama は HSV 5 色判定と一致不能のため
            # check を skip。 代わりに後段の OjamaShapeGate + V check で品質担保。
            if color != COLOR_OJAMA:
                if int(hsv_grid[vrow, col]) != color:
                    continue
            patch = _extract_patch(frame, region, row, col)
            if patch is None:
                continue
            # cycle 58 (2026-05-23): ojama は灰色 (= 低彩度) のため _is_high_quality
            # の鮮やかさ check (S>=80) を skip。 ojama 専用の品質 check は後段の
            # OjamaShapeGate + V check で実施。 これで「cycle 56_v4 で ojama 0 件」
            # の真因 (= MIN_S_MEDIAN=80 で灰色を弾く) を回避。
            if color != COLOR_OJAMA:
                if not _is_high_quality(patch):
                    continue
            # cycle 50 改修 3: 色別 H core レンジ check (= 隣接色混入排除)
            if not _is_color_h_core(patch, color):
                continue
            # cycle 32 G10: ojama 採取時の V 上限 check。 白 (= 勝敗表示 "WIN"/
            # "LOSE" 等) を ojama と誤採取する問題への対策。 V > OJAMA_V_MAX は
            # 白系として ojama 採取拒否。
            if color == COLOR_OJAMA:
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                v_med = float(np.median(hsv[:, :, 2]))
                s_med = float(np.median(hsv[:, :, 1]))
                # cycle 58 (2026-05-23): V/S 二重 check で文字 (= V 高) と
                # 色付き patch (= S 高) を排除。 gate なしでも品質確保。
                if v_med > OJAMA_V_MAX:  # 白文字排除 (V > 180)
                    continue
                if s_med > 60:  # 色付き patch 排除 (= ojama は灰色 = S < 60)
                    continue
                # cycle 58: OjamaShapeGate は厳しすぎる (= 387 中 1 のみ pass)
                # ため _ojama_no_gate=True なら skip。 V/S check で品質担保。
                if not _OJAMA_NO_GATE:
                    if not _OJAMA_SHAPE_GATE.is_ojama(patch):
                        continue
            # cycle 32 G1: bg_fp 距離 check。 背景指紋に近すぎる cell は採取拒否
            # = CNN が「青背景 = blue」 等の誤分類で confirmed/HSV 両方一致する
            # ケースを seed から排除する (= 青背景バイアスの元凶対策)。
            if bg_fp is not None:
                try:
                    bg_cell = bg_fp.cell_at(vrow, col)
                    dist = _bg_fp_distance(patch, bg_cell)
                    if dist < BG_FP_REJECT_THRESHOLD:
                        rejected_by_bg += 1
                        continue
                except Exception:
                    pass
            out.append(PseudoLabelSample(
                component=COMPONENT_CELL,
                timestamp=t_sec,
                input_data={"patch": patch},
                label=color,
                confidence=1.0,
                metadata={
                    "video_id": video_id, "frame_idx": fi,
                    "row": row, "col": col, "side": side,
                    "hsv_seed": True,
                },
            ))
            counts[color] += 1
    if rejected_by_bg > 0 and fi % 300 == 0:
        print(
            f"  [{side} frame={fi}] bg_fp rejected {rejected_by_bg} cells",
        )
    return out


def extract(
    video: Path, video_id: str, out_root: Path,
    max_per_color: int, cnn_model: Path | None,
    hsv_state: Path | None, cnn_override_prob: float | None,
    end_skip_frames: int = DEFAULT_END_SKIP_FRAMES,
    skip_ojama: bool = True,  # cycle 32 I1: ojama 採取スキップ
    max_empty: int = 0,  # cycle 32e: EMPTY 採取上限 (= 0 で無効)
) -> dict[int, int]:
    """1 動画から HSV-seed dataset を抽出。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cycle 32 G9: 動画末尾 N frame skip。 試合切り出し済動画でも末尾に
    # 勝敗表示 (~3 秒) が含まれるため、 これを採取から除外。
    end_frame = max(0, total_frames - end_skip_frames)
    print(
        f"[seed] {video.name}: total={total_frames} end_skip={end_skip_frames} "
        f"→ extract until frame {end_frame}",
    )
    pipeline = RecognitionPipeline.load_default(
        cnn_model_path=cnn_model,
        cnn_override_prob=cnn_override_prob,
    )
    if hsv_state is not None and hsv_state.exists():
        try:
            _pre_inject_hsv(pipeline, hsv_state)
        except Exception as e:
            print(f"[seed] HSV pre-inject failed: {e}", file=sys.stderr)
    store = LabelStore(video_id=video_id, root=out_root)
    counts: dict[int, int] = {c: 0 for c in TRAINABLE_COLORS}
    # cycle 50 改修 4: 非 STABLE 復帰直後 N frame skip のための tracking。
    # 各 side が最後に非 STABLE だった frame index を持ち、 経過 frame で gate。
    last_nonstable_fi: dict[str, int] = {"1P": -10**9, "2P": -10**9}
    fi = 0
    skip_stats = {
        "single_stable_skip": 0,  # 片側 only STABLE 時 skip (= 改修 2)
        "effect_recovery_skip": 0,  # 非 STABLE 復帰直後 skip (= 改修 4)
    }
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # cycle 32 G9: 動画末尾 skip
        if fi >= end_frame:
            print(f"[{video_id}] reached end_frame={end_frame}, stop")
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        # cycle 50 改修 4: 非 STABLE state を tracking (= chain / effect 復帰直後 skip)
        if result.p1.state in NON_STABLE_STATES:
            last_nonstable_fi["1P"] = fi
        if result.p2.state in NON_STABLE_STATES:
            last_nonstable_fi["2P"] = fi
        # cycle 50 改修 2: 両側 STABLE 要求 (= 片側 CHAIN 中の effect 混入対策)
        both_stable = (
            result.p1.state == BoardState.STABLE
            and result.p2.state == BoardState.STABLE
        )
        # cycle 50 改修 4: 両側とも非 STABLE 復帰から N frame 経過したか
        both_effect_recovered = (
            (fi - last_nonstable_fi["1P"]) >= EFFECT_RECOVERY_SKIP_FRAMES
            and (fi - last_nonstable_fi["2P"]) >= EFFECT_RECOVERY_SKIP_FRAMES
        )
        if not both_stable:
            skip_stats["single_stable_skip"] += 1
        elif not both_effect_recovered:
            skip_stats["effect_recovery_skip"] += 1
        if both_stable and both_effect_recovered:
            hc = pipeline._reader._classifier
            if isinstance(hc, HybridClassifier):
                _, hsv_grid_p1 = hc.predict_proba_and_hsv_grid(
                    frame, DEFAULT_P1_REGION,
                )
                _, hsv_grid_p2 = hc.predict_proba_and_hsv_grid(
                    frame, DEFAULT_P2_REGION,
                )
                # cycle 32 G1: pipeline 内で採取済の bg_fp を取得して
                # 採取候補 cell の背景距離 check に使う。 採取前の frame は
                # bg_fp=None → check skip (= 通常採取、 ただし最初の数 frame
                # のみで影響軽微)。
                bg_fp_p1 = getattr(pipeline._reader, "_bg_fp_p1", None)
                bg_fp_p2 = getattr(pipeline._reader, "_bg_fp_p2", None)
                samples = _collect_side_samples(
                    frame, result.p1, DEFAULT_P1_REGION, "1P",
                    hsv_grid_p1, video_id, fi, t_sec,
                    counts, max_per_color,
                    bg_fp=bg_fp_p1, skip_ojama=skip_ojama,
                    max_empty=max_empty,
                )
                samples.extend(_collect_side_samples(
                    frame, result.p2, DEFAULT_P2_REGION, "2P",
                    hsv_grid_p2, video_id, fi, t_sec,
                    counts, max_per_color,
                    bg_fp=bg_fp_p2, skip_ojama=skip_ojama,
                    max_empty=max_empty,
                ))
                if samples:
                    store.append(samples)
        # cycle 32 I1/32e: ojama スキップ + EMPTY 別 max (= EMPTY は max_empty で判定)
        target_colors = [
            c for c in counts
            if not (skip_ojama and c == COLOR_OJAMA)
            and c != COLOR_EMPTY
        ]
        all_reached = all(counts[c] >= max_per_color for c in target_colors)
        empty_reached = (max_empty <= 0 or counts.get(COLOR_EMPTY, 0) >= max_empty)
        if all_reached and empty_reached:
            print(
                f"[{video_id}] all target colors reached {max_per_color} "
                f"@frame={fi}",
            )
            break
        if fi % 300 == 0:
            print(f"[{video_id}] frame={fi} counts={counts}")
        fi += 1
    cap.release()
    print(
        f"[{video_id}] DONE counts={counts} "
        f"skip_stats={skip_stats}",
    )
    return counts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--video-id", type=str, required=True)
    p.add_argument(
        "--out-root", type=Path,
        default=Path("data/pseudo_labels_hsv_seed"),
    )
    p.add_argument("--max-per-color", type=int, default=500)
    p.add_argument("--cnn-model", type=Path, default=None)
    p.add_argument(
        "--hsv-state", type=Path,
        default=Path("data/per_video_hsv_ranges/_merged_default.json"),
    )
    p.add_argument("--cnn-override-prob", type=float, default=None)
    p.add_argument(
        "--end-skip-frames", type=int, default=DEFAULT_END_SKIP_FRAMES,
        help="動画末尾 N frame を採取から除外 (= 勝敗表示対策、 default 180=3秒@60fps)",
    )
    p.add_argument(
        "--include-ojama", action="store_true",
        help="ojama を採取対象に含める (default: False、 構造的破綻のため除外)",
    )
    p.add_argument(
        "--ojama-relaxed", action="store_true",
        help="cycle 58 (= 2026-05-23 案 A): OjamaShapeGate 閾値緩和版を使用。 "
             "cycle 56_v4 で ojama 採取 0 件問題への対策、 文字エフェクト混入リスクと"
             "引換に採取数増加。 --include-ojama 併用時のみ有効",
    )
    p.add_argument(
        "--ojama-no-gate", action="store_true",
        help="cycle 58: OjamaShapeGate を完全 disable し、 V/S check のみで "
             "ojama 採取。 strict/relaxed 共に 387 中 1 件しか pass しないため必須。",
    )
    p.add_argument(
        "--max-empty", type=int, default=0,
        help="cycle 32e: EMPTY 採取上限 (default: 0 で EMPTY 採取無効)。 "
             "推奨値 500 (= puyo 5 色 × 1500 の 1/3 程度、 empty bias 回避)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    # cycle 58 (= 2026-05-23 案 A): --ojama-relaxed で gate 緩和版を global 上書き
    if args.ojama_relaxed and args.include_ojama:
        global _OJAMA_SHAPE_GATE
        _OJAMA_SHAPE_GATE = OjamaShapeGate.relaxed()
        print(f"[ojama-gate] relaxed mode: GRAY_S_MAX={_OJAMA_SHAPE_GATE.GRAY_S_MAX} "
              f"GRAY_V_MAX={_OJAMA_SHAPE_GATE.GRAY_V_MAX} "
              f"EDGE>={_OJAMA_SHAPE_GATE.EDGE_DENSITY_MIN} "
              f"CIRC>={_OJAMA_SHAPE_GATE.CIRCULARITY_MIN}")
    if args.ojama_no_gate and args.include_ojama:
        global _OJAMA_NO_GATE
        _OJAMA_NO_GATE = True
        print("[ojama-gate] DISABLED (= V<=180 + S<=60 のみで採取)")
    extract(
        args.video, args.video_id, args.out_root,
        args.max_per_color, args.cnn_model,
        args.hsv_state, args.cnn_override_prob,
        end_skip_frames=args.end_skip_frames,
        skip_ojama=not args.include_ojama,
        max_empty=args.max_empty,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
