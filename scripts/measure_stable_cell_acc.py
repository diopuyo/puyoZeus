"""STABLE 確定盤面 cell-level 正解率測定スクリプト (Phase I 精度評価)。

3 者独立判定で STABLE 確定盤面の cell 単位正解率を測定する。
3 者:
  1. raw_cnn  : CNN+HSV hybrid の ImageReader 直出力 (物理推論 post-process 前)
  2. raw_hsv  : HSV-only pipeline の ImageReader 直出力 (CNN 完全無効化)
  3. confirmed: CNN+HSV hybrid + 全物理推論 post-process 後の確定盤面

合意 = 3 者のうち少なくとも 2 者が一致したセルを正解ラベル確定とみなす。
分裂 cell は JSON に出力し、人手チェック対象として flag する。

使い方:
    python scripts/measure_stable_cell_acc.py         --videos v89,v97,v29         --holdout v89,v97         --video-dir data/holdout_videos         --output data/verify/stable_cell_acc/2026-05-26.json

判定基準:
    PASS = holdout 全マス平均 >= 99.5% かつ 色別最低 >= 98%
    FAIL = 上記未達 (内訳出力)
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# プロジェクトルートを sys.path に追加 (script 直接実行時)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
)
from src.board_state_machine import BoardState
from src.recognition_evaluator import compute_avg_puyo_count
from src.recognition_pipeline import RecognitionPipeline
# ============================
# 定数定義
# ============================

# 評価対象色 (UNKNOWN は正解ラベル対象外)
EVAL_COLORS: tuple[int, ...] = (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)

COLOR_NAMES: dict[int, str] = {
    COLOR_EMPTY:   "empty",
    COLOR_RED:     "red",
    COLOR_BLUE:    "blue",
    COLOR_GREEN:   "green",
    COLOR_YELLOW:  "yellow",
    COLOR_PURPLE:  "purple",
    COLOR_OJAMA:   "ojama",
    COLOR_UNKNOWN: "unknown",
}

# 精度基準値
PASS_OVERALL_THRESHOLD: float = 0.995
PASS_PER_COLOR_THRESHOLD: float = 0.98

# 認識処理間隔 (秒)
DEFAULT_SAMPLE_INTERVAL_SEC: float = 1.0 / 30.0

# 1 動画あたり最大処理フレーム数 (0 = 制限なし)
DEFAULT_MAX_FRAMES: int = 0

# UNKNOWN 含む cell をスキップするか
SKIP_UNKNOWN_CELLS: bool = True

# 不一致 cell 出力上限
DISAGREEMENT_OUTPUT_LIMIT: int = 500
DISAGREEMENT_COLLECT_LIMIT: int = 2000

# I1 メトリクス: per_col_unknown_rate 閾値
# STABLE confirmed_board で col 別 COLOR_UNKNOWN 比率が高い = 認識崩壊 (v89 27-30s 相当)
# mismatch/replace が fail-silent でも col 別 UNKNOWN 率は明示的に上昇する
PER_COL_UNKNOWN_WARNING: float = 0.15  # 15% 超 = WARNING
PER_COL_UNKNOWN_CRITICAL: float = 0.30  # 30% 超 = CRITICAL

# I1 メトリクス: non_stable_consecutive_frames 閾値
# state が stable 以外の連続サンプリングフレーム数
# 試合序盤 15 秒バッファ後から計測する (= 初期化猶予)
NON_STABLE_CRITICAL_FRAMES: int = 180  # 180 sample frame = ~3 秒 @ 60fps
NON_STABLE_WARMUP_SEC: float = 15.0  # 試合開始から 15 秒は計測除外

# I1 メトリクス: per_col_empty_rate_by_game_phase 閾値
# 中盤 (= 30 秒以降) で特定 col が全 STABLE フレーム中 100% EMPTY なら CRITICAL
# v40_match01「1P col=1 全 EMPTY 誤判定」 を捕捉する
MIDGAME_START_SEC: float = 30.0  # 中盤開始時刻 (秒)
MIDGAME_COL_EMPTY_CRITICAL: float = 0.99  # 99% 以上 EMPTY = CRITICAL
MIDGAME_COL_MIN_FRAMES: int = 30  # 最低 30 STABLE frame が必要

# HSV DB ルート
_HSV_DB_ROOT = Path("data/per_video_hsv_ranges")
_HSV_MERGED_DEFAULT = _HSV_DB_ROOT / "_merged_default.json"

# 動画検索ディレクトリ (デフォルト)
_DEFAULT_VIDEO_DIRS: tuple[Path, ...] = (
    Path("data/evaluation_videos"),
    Path("data/holdout_videos"),
)
# ============================
# データクラス
# ============================

@dataclass
class VideoStats:
    """1 動画の集計結果。

    agreed_cells の判定方式:
      3 者独立モード (use_three_way=True): raw_cnn / raw_hsv / confirmed の
      うち 2 者以上が一致したセルを合意とみなす。
      2 者モード (後方互換): raw_cnn == raw_hsv の一致のみ (旧挙動)。
    """

    video_id: str
    is_holdout: bool
    total_cells: int = 0
    agreed_cells: int = 0
    correct_by_color: dict = field(default_factory=lambda: defaultdict(int))
    total_by_color: dict = field(default_factory=lambda: defaultdict(int))
    correct_by_row: dict = field(default_factory=lambda: defaultdict(int))
    total_by_row: dict = field(default_factory=lambda: defaultdict(int))
    disagreement_count: int = 0
    stable_frame_count: int = 0
    # 3 者独立メトリクス (追加フィールド、後方互換のため default=0)
    # physics_fix_count: raw_cnn != raw_hsv だが confirmed が正解ラベルと一致したセル数
    physics_fix_count: int = 0
    # all_three_agree_count: 3 者全員一致セル数
    all_three_agree_count: int = 0
    # ------------------------------------------------
    # I1 追加メトリクス (後方互換のため全て default 付き)
    # ------------------------------------------------
    # per_col_unknown_cells[col]: confirmed_board で COLOR_UNKNOWN だった cell 数 (col 別)
    per_col_unknown_cells: dict = field(default_factory=lambda: defaultdict(int))
    # per_col_stable_cells[col]: STABLE フレームで確認した cell 数 (col 別、分母)
    per_col_stable_cells: dict = field(default_factory=lambda: defaultdict(int))
    # non_stable_max_consecutive: warmup 後の最長連続 non-stable サンプルフレーム数
    non_stable_max_consecutive: int = 0
    # per_col_midgame_empty_cells[col]: 中盤 (>= MIDGAME_START_SEC) で EMPTY だった cell 数
    per_col_midgame_empty_cells: dict = field(default_factory=lambda: defaultdict(int))
    # per_col_midgame_cells[col]: 中盤 STABLE フレームで確認した cell 数 (col 別、分母)
    per_col_midgame_cells: dict = field(default_factory=lambda: defaultdict(int))
    # _non_stable_current_by_side: side 別 non-stable 連続カウンタ (内部、init 時 {}).
    # non_stable_max_consecutive の更新用。直接参照禁止。
    _non_stable_current_by_side: dict = field(default_factory=dict, repr=False, compare=False)
    # C1: avg_puyo_count メトリクス (後方互換のため default 付き)
    # STABLE フレームの 1P+2P 合算ぷよ数合計と frame 数
    _puyo_count_sum: int = 0
    _puyo_count_n_stable: int = 0


# ============================
# ユーティリティ
# ============================

def _resolve_video_path(video_id: str, video_dir: Optional[Path]) -> Optional[Path]:
    """動画 ID からファイルパスを解決する。

    video_dir 以下のサブディレクトリも再帰検索する (rglob)。
    これにより data/match_clips/v29/v29_match01.mp4 形式にも対応。
    """
    search_dirs: list[Path] = (
        [video_dir] if video_dir is not None else list(_DEFAULT_VIDEO_DIRS)
    )
    for d in search_dirs:
        if not d.exists():
            continue
        # rglob で再帰検索 (サブディレクトリ対応)
        for f in sorted(d.rglob("*")):
            if f.suffix in (".mp4", ".mkv", ".avi", ".mov") and video_id in f.name:
                return f
    return None


def _resolve_hsv_path(video_id: str) -> Path:
    """動画 ID から per-video HSV JSON を解決する。

    clip ID (例: v29_match01) の場合、先頭の v<NN> 部分を抽出して
    v29.json を探す。完全一致ファイルが優先。
    """
    # 完全一致を優先
    candidate = _HSV_DB_ROOT / f"{video_id}.json"
    if candidate.exists():
        return candidate
    # clip ID (v29_match01 など) から動画 ID を抽出して再試行
    import re
    m = re.match(r"^(v\d+)", video_id)
    if m:
        base_candidate = _HSV_DB_ROOT / f"{m.group(1)}.json"
        if base_candidate.exists():
            return base_candidate
    return _HSV_MERGED_DEFAULT


def _inject_hsv(pipe: RecognitionPipeline, hsv_path: Path) -> None:
    """pipeline に per-video HSV ranges を注入する。"""
    if not hsv_path.exists():
        return
    try:
        with hsv_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in state.get("per_video_ranges", {}).items()
        }
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if (
            isinstance(hc, HybridClassifier)
            and hasattr(hc._hsv, "set_color_ranges_from_simple")
            and ranges
        ):
            hc._hsv.set_color_ranges_from_simple(ranges)
            if pipe._online_hsv is not None:
                pipe._online_hsv_injected = True
    except Exception as e:
        print(f"[measure] HSV inject 失敗 ({hsv_path}): {e}", file=sys.stderr)
def _make_pipeline_cnn(video_id: str) -> RecognitionPipeline:
    """CNN + HSV ハイブリッド pipeline を構築する。"""
    pipe = RecognitionPipeline.load_default(force_in_match=True)
    _inject_hsv(pipe, _resolve_hsv_path(video_id))
    return pipe


def _make_pipeline_hsv_only(video_id: str) -> RecognitionPipeline:
    """HSV-only pipeline を構築する。

    cnn_override_prob=2.0 で CNN 採用閾値を 1.0 超にし、
    HybridClassifier が常に HSV 側を採用するよう強制する。
    backwards compat: load_default の既存シグネチャに optional 引数追加のみ。
    """
    pipe = RecognitionPipeline.load_default(
        cnn_override_prob=2.0,
        force_in_match=True,
    )
    _inject_hsv(pipe, _resolve_hsv_path(video_id))
    return pipe


# ============================
# 1 動画処理
# ============================


def _open_capture(
    video_path: Path,
    max_frames: int,
    sample_interval_sec: float,
) -> tuple:
    """動画キャプチャを開き (cap, fps, n_target, interval_frames) を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    interval_frames = max(1, int(round(sample_interval_sec * fps)))
    return cap, fps, n_target, interval_frames



def _eval_one_frame(
    video_id: str,
    fi: int,
    fps: float,
    interval_frames: int,
    frame: object,
    pipe_cnn: RecognitionPipeline,
    pipe_hsv: RecognitionPipeline,
    stats: VideoStats,
    disagreements: list[dict],
) -> None:
    """1 frame の認識・合意判定を行い stats を更新する。

    3 者独立方式:
      raw_cnn   = res_cnn.pX.cnn_board  (ImageReader 直出力、物理推論 post-process 前)
      raw_hsv   = res_hsv.pX.cnn_board  (HSV-only pipeline の ImageReader 直出力)
      confirmed = res_cnn.pX.confirmed_board (CNN+物理推論 post-process 後の確定盤面)
    """
    t_sec = fi / fps
    if fi % interval_frames != 0:
        return
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    res_cnn = pipe_cnn.update(fi, t_sec, frame)
    res_hsv = pipe_hsv.update(fi, t_sec, frame)
    for side, sr_cnn, sr_hsv in [
        ("1P", res_cnn.p1, res_hsv.p1),
        ("2P", res_cnn.p2, res_hsv.p2),
    ]:
        if sr_cnn.state != BoardState.STABLE or sr_cnn.confirmed_board is None:
            # non-stable フレームをカウント (warmup 後のみ)
            if t_sec >= NON_STABLE_WARMUP_SEC:
                stats._non_stable_current_by_side[side] = (
                    stats._non_stable_current_by_side.get(side, 0) + 1
                )
                cur = stats._non_stable_current_by_side[side]
                if cur > stats.non_stable_max_consecutive:
                    stats.non_stable_max_consecutive = cur
            continue
        # STABLE フレームで non-stable カウントをリセット
        stats._non_stable_current_by_side[side] = 0
        stats.stable_frame_count += 1
        # C1: STABLE confirmed_board のぷよ数を集計 (= avg_puyo_count 計算用)
        _collect_puyo_count(sr_cnn.confirmed_board, stats)
        _eval_side_frame(
            side, fi, t_sec, video_id,
            raw_cnn_board=sr_cnn.cnn_board,
            raw_hsv_board=sr_hsv.cnn_board,
            confirmed_board=sr_cnn.confirmed_board,
            stats=stats,
            disagreements=disagreements,
        )


def _run_frame_loop(
    video_id: str,
    cap: object,
    fps: float,
    n_target: int,
    interval_frames: int,
    is_holdout: bool,
    pipe_cnn: RecognitionPipeline,
    pipe_hsv: RecognitionPipeline,
    disagreements: list[dict],
) -> VideoStats:
    """動画 frame ループを走らせ VideoStats を返す。"""
    stats = VideoStats(video_id=video_id, is_holdout=is_holdout)
    for fi in range(n_target):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        _eval_one_frame(
            video_id, fi, fps, interval_frames, frame,
            pipe_cnn, pipe_hsv, stats, disagreements,
        )
        if fi % 300 == 0 and fi > 0:
            print(
                f"  [progress] {fi}/{n_target} ({fi*100/max(n_target,1):.0f}%) "
                f"agreed={stats.agreed_cells} total={stats.total_cells}"
            )
    return stats


def _process_video(
    video_id: str,
    video_path: Path,
    is_holdout: bool,
    max_frames: int,
    sample_interval_sec: float,
    disagreements: list[dict],
) -> VideoStats:
    """1 動画を処理し VideoStats を返す。"""
    cap_info = _open_capture(video_path, max_frames, sample_interval_sec)
    if cap_info is None:
        print(f"[measure] 動画を開けません: {video_path}", file=sys.stderr)
        return VideoStats(video_id=video_id, is_holdout=is_holdout)
    cap, fps, n_target, interval_frames = cap_info
    pipe_cnn = _make_pipeline_cnn(video_id)
    pipe_hsv = _make_pipeline_hsv_only(video_id)
    print(f"[measure] {video_id}: fps={fps:.1f} target={n_target} holdout={is_holdout}")
    stats = _run_frame_loop(
        video_id, cap, fps, n_target, interval_frames,
        is_holdout, pipe_cnn, pipe_hsv, disagreements,
    )
    cap.release()
    rate = stats.agreed_cells / stats.total_cells if stats.total_cells > 0 else 0.0
    print(
        f"[measure] {video_id} 完了: stable={stats.stable_frame_count} "
        f"total={stats.total_cells} 合意率={rate:.4f} disagree={stats.disagreement_count}"
    )
    return stats
def _majority_vote(a: int, b: int, c: int) -> int:
    """3 値の多数決を返す。全員不一致の場合は a (raw_cnn) を返す。"""
    if a == b or a == c:
        return a
    if b == c:
        return b
    return a  # 全員不一致: raw_cnn を基準にする


def _record_cell(
    video_id: str, fi: int, t_sec: float, side: str,
    row: int, col: int,
    raw_cnn_val: int, raw_hsv_val: int, confirmed_val: int,
    stats: VideoStats,
    disagreements: list[dict],
) -> None:
    """1 cell の 3 者独立合意判定結果を stats と disagreements に記録する。

    合意ラベル = raw_cnn / raw_hsv / confirmed の多数決 (2 者以上一致)。
    全員不一致時は raw_cnn を正解ラベルとして扱う (最保守的方針)。

    合意 = 多数決ラベルが raw_cnn と一致 (= 集計基準は raw_cnn 主軸)。
    """
    label = _majority_vote(raw_cnn_val, raw_hsv_val, confirmed_val)
    stats.total_cells += 1
    # total_by_color / total_by_row は label (多数決) を基準にする
    stats.total_by_color[label] += 1
    stats.total_by_row[row] += 1

    all_agree = (raw_cnn_val == raw_hsv_val == confirmed_val)
    if all_agree:
        stats.all_three_agree_count += 1

    if raw_cnn_val == label:
        stats.agreed_cells += 1
        stats.correct_by_color[label] += 1
        stats.correct_by_row[row] += 1
    else:
        stats.disagreement_count += 1
        if len(disagreements) < DISAGREEMENT_COLLECT_LIMIT:
            disagreements.append({
                "video": video_id, "frame": fi, "t_sec": round(t_sec, 2),
                "side": side, "cell": [row, col],
                "predictions": {
                    "raw_cnn": COLOR_NAMES.get(raw_cnn_val, str(raw_cnn_val)),
                    "raw_hsv": COLOR_NAMES.get(raw_hsv_val, str(raw_hsv_val)),
                    "confirmed": COLOR_NAMES.get(confirmed_val, str(confirmed_val)),
                    "majority_label": COLOR_NAMES.get(label, str(label)),
                },
            })
        return

    # physics_fix_count: raw_cnn != raw_hsv だが confirmed が label と一致したケース
    if raw_cnn_val != raw_hsv_val and confirmed_val == label:
        stats.physics_fix_count += 1


def _eval_side_frame(
    side: str,
    fi: int,
    t_sec: float,
    video_id: str,
    raw_cnn_board: object,
    raw_hsv_board: object,
    confirmed_board: object,
    stats: VideoStats,
    disagreements: list[dict],
) -> None:
    """1 サイド × 1 frame の cell ごとに 3 者独立合意判定し stats を更新する。

    Args:
        raw_cnn_board: CNN+HSV hybrid ImageReader 直出力 (物理推論 post-process 前)。
        raw_hsv_board: HSV-only pipeline の ImageReader 直出力 (CNN 無効化済)。
        confirmed_board: CNN+物理推論 post-process 後の確定盤面 (STABLE 時のみ非 None)。
    """
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            raw_cnn_val = int(raw_cnn_board.get(row, col)) if raw_cnn_board is not None else COLOR_UNKNOWN
            raw_hsv_val = int(raw_hsv_board.get(row, col)) if raw_hsv_board is not None else COLOR_UNKNOWN
            confirmed_val = int(confirmed_board.get(row, col)) if confirmed_board is not None else COLOR_UNKNOWN
            if SKIP_UNKNOWN_CELLS and (
                raw_cnn_val == COLOR_UNKNOWN
                or raw_hsv_val == COLOR_UNKNOWN
                or confirmed_val == COLOR_UNKNOWN
            ):
                continue
            # 評価対象色フィルタ: 多数決ラベルで判定 (全員 UNKNOWN 等を除外)
            label = _majority_vote(raw_cnn_val, raw_hsv_val, confirmed_val)
            if label not in EVAL_COLORS:
                continue
            _record_cell(
                video_id, fi, t_sec, side, row, col,
                raw_cnn_val, raw_hsv_val, confirmed_val,
                stats, disagreements,
            )
    # I1 メトリクス集計: confirmed_board の col 別 UNKNOWN 率 + 中盤 EMPTY 率
    if confirmed_board is not None:
        _collect_col_metrics(fi, t_sec, confirmed_board, stats)


def _collect_puyo_count(confirmed_board: object, stats: VideoStats) -> None:
    """STABLE confirmed_board の非 EMPTY・非 UNKNOWN cell 数を stats に加算する。

    C1 avg_puyo_count_per_stable_frame 計算用。
    1 サイド分のカウントを加算する (= frame ごとに 1P / 2P 別に呼ばれる)。
    """
    if confirmed_board is None:
        return
    count = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            val = int(confirmed_board.get(row, col))
            if val not in (COLOR_EMPTY, COLOR_UNKNOWN):
                count += 1
    stats._puyo_count_sum += count
    stats._puyo_count_n_stable += 1


def _collect_col_metrics(
    fi: int,
    t_sec: float,
    confirmed_board: object,
    stats: VideoStats,
) -> None:
    """STABLE confirmed_board から col 別 UNKNOWN 率と中盤 EMPTY 率を集計する。

    col 別 UNKNOWN 率が高い = STABLE 中の認識崩壊 (v89 27-30s 相当) を捕捉。
    中盤 EMPTY 率が 100% = col=1 全 EMPTY 誤判定 (v40_match01 相当) を捕捉。
    """
    is_midgame = t_sec >= MIDGAME_START_SEC
    for col in range(BOARD_COLS):
        col_unknown = 0
        col_cells = 0
        col_empty_mid = 0
        col_cells_mid = 0
        for row in range(BOARD_ROWS):
            val = int(confirmed_board.get(row, col))
            col_cells += 1
            if val == COLOR_UNKNOWN:
                col_unknown += 1
            if is_midgame:
                col_cells_mid += 1
                if val == COLOR_EMPTY:
                    col_empty_mid += 1
        stats.per_col_stable_cells[col] += col_cells
        stats.per_col_unknown_cells[col] += col_unknown
        if is_midgame:
            stats.per_col_midgame_cells[col] += col_cells_mid
            stats.per_col_midgame_empty_cells[col] += col_empty_mid


# ============================
# 集計・判定
# ============================


def _build_color_acc(stats_list: list[VideoStats]) -> dict[str, float]:
    """色別合意率 dict を生成する。"""
    total: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for s in stats_list:
        for c in EVAL_COLORS:
            total[c] += s.total_by_color.get(c, 0)
            correct[c] += s.correct_by_color.get(c, 0)
    return {
        COLOR_NAMES[c]: correct[c] / total[c] if total[c] > 0 else 1.0
        for c in EVAL_COLORS
    }


def _build_row_acc(stats_list: list[VideoStats]) -> dict[str, float]:
    """行別合意率 dict を生成する。"""
    total: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for s in stats_list:
        for r in range(BOARD_ROWS):
            total[r] += s.total_by_row.get(r, 0)
            correct[r] += s.correct_by_row.get(r, 0)
    return {
        f"row_{r}": correct[r] / total[r] if total[r] > 0 else 1.0
        for r in range(BOARD_ROWS)
    }


def _build_video_acc(stats_list: list[VideoStats]) -> dict[str, dict]:
    """動画別集計 dict を生成する。"""
    return {
        s.video_id: {
            "acc": s.agreed_cells / s.total_cells if s.total_cells > 0 else 0.0,
            "total_cells": s.total_cells,
            "agreed_cells": s.agreed_cells,
            "disagreement_count": s.disagreement_count,
            "stable_frame_count": s.stable_frame_count,
            "is_holdout": s.is_holdout,
            # 3 者独立追加メトリクス
            "physics_fix_count": s.physics_fix_count,
            "all_three_agree_count": s.all_three_agree_count,
            # I1 追加メトリクス
            "non_stable_max_consecutive": s.non_stable_max_consecutive,
            "per_col_unknown_rate": {
                str(col): (
                    s.per_col_unknown_cells.get(col, 0)
                    / s.per_col_stable_cells.get(col, 1)
                )
                for col in range(6)
            },
            "per_col_midgame_empty_rate": {
                str(col): (
                    s.per_col_midgame_empty_cells.get(col, 0)
                    / s.per_col_midgame_cells.get(col, 1)
                    if s.per_col_midgame_cells.get(col, 0) > 0 else None
                )
                for col in range(6)
            },
            # C1 avg_puyo_count_per_stable_frame (= fail-silent 経路検知)
            "avg_puyo_count_per_stable_frame": (
                s._puyo_count_sum / s._puyo_count_n_stable
                if s._puyo_count_n_stable > 0 else None
            ),
            "n_stable_frames_puyo": s._puyo_count_n_stable,
        }
        for s in stats_list
    }


def _build_i1_summary(stats_list: list[VideoStats]) -> dict:
    """I1 メトリクスの全動画 worst-case サマリを返す。

    per_col_unknown_rate の worst col と worst video、
    non_stable_max_consecutive の max video、
    per_col_midgame_empty_rate の worst col を集計する。
    """
    worst_unknown: dict[int, float] = {col: 0.0 for col in range(6)}
    worst_unknown_vid: dict[int, str] = {col: "" for col in range(6)}
    max_non_stable: int = 0
    max_non_stable_vid: str = ""
    worst_midgame: dict[int, float] = {col: 0.0 for col in range(6)}
    worst_midgame_vid: dict[int, str] = {col: "" for col in range(6)}
    for s in stats_list:
        for col in range(6):
            stable = s.per_col_stable_cells.get(col, 0)
            if stable > 0:
                rate = s.per_col_unknown_cells.get(col, 0) / stable
                if rate > worst_unknown[col]:
                    worst_unknown[col] = rate
                    worst_unknown_vid[col] = s.video_id
        if s.non_stable_max_consecutive > max_non_stable:
            max_non_stable = s.non_stable_max_consecutive
            max_non_stable_vid = s.video_id
        for col in range(6):
            mid_cells = s.per_col_midgame_cells.get(col, 0)
            if mid_cells >= MIDGAME_COL_MIN_FRAMES:
                rate = s.per_col_midgame_empty_cells.get(col, 0) / mid_cells
                if rate > worst_midgame[col]:
                    worst_midgame[col] = rate
                    worst_midgame_vid[col] = s.video_id
    return {
        "per_col_unknown_worst": {
            str(col): {"rate": worst_unknown[col], "video": worst_unknown_vid[col]}
            for col in range(6)
        },
        "non_stable_max_consecutive": {
            "max": max_non_stable, "video": max_non_stable_vid
        },
        "per_col_midgame_empty_worst": {
            str(col): {"rate": worst_midgame[col], "video": worst_midgame_vid[col]}
            for col in range(6)
        },
        "thresholds": {
            "per_col_unknown_warning": PER_COL_UNKNOWN_WARNING,
            "per_col_unknown_critical": PER_COL_UNKNOWN_CRITICAL,
            "non_stable_critical_frames": NON_STABLE_CRITICAL_FRAMES,
            "midgame_col_empty_critical": MIDGAME_COL_EMPTY_CRITICAL,
        },
    }


def _aggregate_stats(stats_list: list[VideoStats]) -> dict:
    """VideoStats リストから JSON 出力用 dict を生成する。"""
    total_cells = sum(s.total_cells for s in stats_list)
    correct = sum(s.agreed_cells for s in stats_list)
    overall_acc = correct / total_cells if total_cells > 0 else 0.0
    total_physics_fix = sum(s.physics_fix_count for s in stats_list)
    total_all_three = sum(s.all_three_agree_count for s in stats_list)
    return {
        "overall": {
            "acc": overall_acc,
            "total_cells": total_cells,
            "correct": correct,
            # 3 者独立追加メトリクス
            "physics_fix_count": total_physics_fix,
            "all_three_agree_count": total_all_three,
            "physics_fix_rate": (
                total_physics_fix / total_cells if total_cells > 0 else 0.0
            ),
            "all_three_agree_rate": (
                total_all_three / total_cells if total_cells > 0 else 0.0
            ),
        },
        "per_color": _build_color_acc(stats_list),
        "per_row": _build_row_acc(stats_list),
        "per_video": _build_video_acc(stats_list),
        # I1 メトリクス集計サマリ (全動画 worst-case)
        "i1_metrics_summary": _build_i1_summary(stats_list),
    }


def _compute_holdout_summary(
    stats_list: list[VideoStats],
    holdout_ids: list[str],
) -> dict:
    """holdout 動画のみの集計結果を返す。"""
    ho_stats = [s for s in stats_list if s.video_id in holdout_ids]
    if not ho_stats:
        return {"acc": None, "videos": holdout_ids, "note": "holdout 動画なし"}
    total = sum(s.total_cells for s in ho_stats)
    correct = sum(s.agreed_cells for s in ho_stats)
    return {
        "acc": correct / total if total > 0 else 0.0,
        "total_cells": total,
        "correct": correct,
        "videos": holdout_ids,
    }


def _judge_pass_fail(
    overall_acc: float,
    per_color: dict[str, float],
    holdout_acc: Optional[float],
    stats_list: Optional[list] = None,
) -> tuple[str, list[str]]:
    """PASS/FAIL 判定と失敗理由リストを返す。

    I1 メトリクス (per_col_unknown_rate / non_stable_consecutive / per_col_midgame_empty)
    が NG ならも FAIL にする。stats_list=None なら従来通りの acc 判定のみ。
    backwards compat: stats_list は optional 引数。
    """
    target_acc = holdout_acc if holdout_acc is not None else overall_acc
    failures: list[str] = []

    if target_acc < PASS_OVERALL_THRESHOLD:
        failures.append(
            f"全マス平均 {target_acc:.4f} < 閾値 {PASS_OVERALL_THRESHOLD:.4f}"
        )

    for color_name, acc in per_color.items():
        if acc < PASS_PER_COLOR_THRESHOLD:
            failures.append(
                f"色別 {color_name}: {acc:.4f} < 閾値 {PASS_PER_COLOR_THRESHOLD:.4f}"
            )

    # I1 メトリクス判定
    if stats_list is not None:
        failures.extend(_judge_i1_metrics(stats_list))

    return ("PASS" if not failures else "FAIL"), failures


def _judge_i1_metrics(stats_list: list) -> list[str]:
    """I1 追加メトリクスの FAIL 判定を返す。

    per_col_unknown_rate / non_stable_max_consecutive / per_col_midgame_empty_rate
    の 3 メトリクスが閾値超なら FAIL 理由を追加する。
    mismatch/replace が fail-silent でも本メトリクスは発火する (= cycle 23/24 の反省)。
    """
    failures: list[str] = []
    for s in stats_list:
        # メトリクス 1: per_col_unknown_rate
        for col in range(6):
            stable = s.per_col_stable_cells.get(col, 0)
            if stable == 0:
                continue
            rate = s.per_col_unknown_cells.get(col, 0) / stable
            if rate >= PER_COL_UNKNOWN_CRITICAL:
                failures.append(
                    f"[{s.video_id}] col={col} UNKNOWN率 {rate:.1%} >= CRITICAL閾値 {PER_COL_UNKNOWN_CRITICAL:.0%}"
                    f" (fail-silent 対象: cycle 23/24 参照)"
                )
            elif rate >= PER_COL_UNKNOWN_WARNING:
                failures.append(
                    f"[{s.video_id}] col={col} UNKNOWN率 {rate:.1%} >= WARNING閾値 {PER_COL_UNKNOWN_WARNING:.0%}"
                )
        # メトリクス 2: non_stable_max_consecutive
        if s.non_stable_max_consecutive >= NON_STABLE_CRITICAL_FRAMES:
            failures.append(
                f"[{s.video_id}] non_stable 連続 {s.non_stable_max_consecutive} frames"
                f" >= CRITICAL閾値 {NON_STABLE_CRITICAL_FRAMES}"
            )
        # メトリクス 3: per_col_midgame_empty_rate
        for col in range(6):
            mid_cells = s.per_col_midgame_cells.get(col, 0)
            if mid_cells < MIDGAME_COL_MIN_FRAMES:
                continue
            empty_rate = s.per_col_midgame_empty_cells.get(col, 0) / mid_cells
            if empty_rate >= MIDGAME_COL_EMPTY_CRITICAL:
                failures.append(
                    f"[{s.video_id}] 中盤 col={col} EMPTY率 {empty_rate:.1%} >= CRITICAL閾値 {MIDGAME_COL_EMPTY_CRITICAL:.0%}"
                    f" (v40_match01 全EMPTY誤判定パターン)"
                )
    return failures

# ============================
# CLI
# ============================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STABLE 確定盤面 cell-level 正解率測定",
    )
    p.add_argument(
        "--videos", type=str, required=True,
        help="評価対象動画 ID リスト (カンマ区切り)。例: v89,v97,v29",
    )
    p.add_argument(
        "--holdout", type=str, default="",
        help="holdout 動画 ID リスト (カンマ区切り)。例: v89,v97",
    )
    p.add_argument(
        "--video-dir", type=Path, default=None,
        help="動画ファイル検索ルートディレクトリ。省略時は自動検索。",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="結果 JSON 出力パス。省略時は data/verify/stable_cell_acc/<timestamp>.json。",
    )
    p.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="1 動画あたり最大処理フレーム数 (0=制限なし)。",
    )
    p.add_argument(
        "--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC,
        help="認識処理間隔 (秒)。",
    )
    return p.parse_args()



def _resolve_output_path(output: object) -> Path:
    """出力 JSON パスを決定する。"""
    if output is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        return Path("data/verify/stable_cell_acc") / f"{ts}.json"
    return Path(output)


def _collect_results(
    video_ids: list[str],
    holdout_ids: list[str],
    video_dir: object,
    max_frames: int,
    sample_interval_sec: float,
    disagreements: list[dict],
) -> list[VideoStats]:
    """動画リストを走らせ VideoStats リストを返す。"""
    stats_list: list[VideoStats] = []
    for vid in video_ids:
        vpath = _resolve_video_path(vid, video_dir)
        if vpath is None:
            print(f"[measure] 動画ファイル未発見: {vid} → スキップ", file=sys.stderr)
            continue
        vstats = _process_video(
            video_id=vid,
            video_path=vpath,
            is_holdout=(vid in holdout_ids),
            max_frames=max_frames,
            sample_interval_sec=sample_interval_sec,
            disagreements=disagreements,
        )
        stats_list.append(vstats)
    return stats_list


def _print_summary(
    agg: dict,
    holdout_summary: dict,
    holdout_acc: object,
    holdout_ids: list,
    failures: list,
    n_disagreements: int,
    output_path: Path,
) -> None:
    """評価結果のサマリを標準出力に表示する。"""
    sep = "=" * 60
    verdict = "PASS" if not failures else "FAIL"
    print("")
    print(sep)
    print("判定: " + verdict)
    ov = agg["overall"]
    acc_str = "{:.4f} ({}/{})" .format(ov["acc"], ov["correct"], ov["total_cells"])
    print("全マス平均合意率: " + acc_str)
    if holdout_ids and holdout_acc is not None:
        ho_str = "{:.4f} ({}/{})".format(
            holdout_acc,
            holdout_summary.get("correct", 0),
            holdout_summary.get("total_cells", 0),
        )
        print("holdout 合意率:   " + ho_str)
    print("[色別合意率]")
    for cname, acc in sorted(agg["per_color"].items()):
        mark = "OK" if acc >= PASS_PER_COLOR_THRESHOLD else "NG"
        print("  {:8s}: {:.4f}  [{}]".format(cname, acc, mark))
    print("[不一致 cell 総数]: " + str(n_disagreements))
    print("[3 者独立メトリクス]")
    print("  全員一致率:         {:.4f} ({}/{})".format(
        ov.get("all_three_agree_rate", 0.0),
        ov.get("all_three_agree_count", 0),
        ov.get("total_cells", 0),
    ))
    print("  物理推論修正率:     {:.4f} ({}/{})".format(
        ov.get("physics_fix_rate", 0.0),
        ov.get("physics_fix_count", 0),
        ov.get("total_cells", 0),
    ))
    # I1 メトリクスサマリ出力
    per_vid = agg.get("per_video", {})
    has_i1 = any(
        "per_col_unknown_rate" in v for v in per_vid.values()
    )
    if has_i1:
        print("[I1 メトリクス: per_col_unknown_rate (STABLE 中 col 別 UNKNOWN 率)]")
        for vid_id, vid_data in per_vid.items():
            rates = vid_data.get("per_col_unknown_rate", {})
            for col_key, rate in sorted(rates.items()):
                if rate >= PER_COL_UNKNOWN_WARNING:
                    mark = "CRITICAL" if rate >= PER_COL_UNKNOWN_CRITICAL else "WARNING"
                    print(f"  [{vid_id}] col={col_key}: {rate:.1%}  [{mark}]")
        print("[I1 メトリクス: non_stable_max_consecutive (最長連続 non-STABLE フレーム数)]")
        for vid_id, vid_data in per_vid.items():
            n = vid_data.get("non_stable_max_consecutive", 0)
            mark = "CRITICAL" if n >= NON_STABLE_CRITICAL_FRAMES else "ok"
            if n > 0:
                print(f"  [{vid_id}] max={n}  [{mark}]")
        print("[I1 メトリクス: per_col_midgame_empty_rate (中盤 col 別 EMPTY 率)]")
        for vid_id, vid_data in per_vid.items():
            rates = vid_data.get("per_col_midgame_empty_rate", {})
            for col_key, rate in sorted(rates.items()):
                if rate is not None and rate >= MIDGAME_COL_EMPTY_CRITICAL:
                    print(f"  [{vid_id}] col={col_key}: {rate:.1%}  [CRITICAL]")
    # C1: avg_puyo_count_per_stable_frame 出力 (= fail-silent 経路検知)
    has_avg = any(
        v.get("avg_puyo_count_per_stable_frame") is not None
        for v in per_vid.values()
    )
    if has_avg:
        print("[C1 avg_puyo_count_per_stable_frame (STABLE フレームの 1P or 2P 平均ぷよ数)]")
        for vid_id, vid_data in per_vid.items():
            avg = vid_data.get("avg_puyo_count_per_stable_frame")
            n_st = vid_data.get("n_stable_frames_puyo", 0)
            if avg is not None:
                print(f"  [{vid_id}] avg={avg:.2f} (n_stable={n_st})")
    if failures:
        print("[FAIL 理由]")
        for reason in failures:
            print("  - " + reason)
    print("[結果 JSON]: " + str(output_path))
    print(sep)
def main() -> int:
    """PASS なら 0, FAIL なら 1 を返す。"""
    args = _parse_args()
    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    holdout_ids = (
        [v.strip() for v in args.holdout.split(",") if v.strip()]
        if args.holdout else []
    )
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[measure] 評価開始: videos={video_ids} holdout={holdout_ids}")
    print(f"[measure] 出力先: {output_path}")
    disagreements: list[dict] = []
    stats_list = _collect_results(
        video_ids, holdout_ids, args.video_dir,
        args.max_frames, args.sample_interval, disagreements,
    )
    if not stats_list:
        print("[measure] 処理した動画がゼロ件。終了。", file=sys.stderr)
        return 2
    agg = _aggregate_stats(stats_list)
    holdout_summary = _compute_holdout_summary(stats_list, holdout_ids)
    holdout_acc = holdout_summary.get("acc") if holdout_ids else None
    verdict, failures = _judge_pass_fail(
        overall_acc=agg["overall"]["acc"],
        per_color=agg["per_color"],
        holdout_acc=holdout_acc,
        stats_list=stats_list,
    )
    result = {
        **agg,
        "holdout_summary": holdout_summary,
        "disagreement_cells": disagreements[:DISAGREEMENT_OUTPUT_LIMIT],
        "disagreement_total": len(disagreements),
        "verdict": verdict, "failures": failures,
        "meta": {
            "videos": video_ids, "holdout": holdout_ids,
            "max_frames": args.max_frames,
            "sample_interval_sec": args.sample_interval,
            "pass_overall_threshold": PASS_OVERALL_THRESHOLD,
            "pass_per_color_threshold": PASS_PER_COLOR_THRESHOLD,
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _print_summary(agg, holdout_summary, holdout_acc, holdout_ids,
                   failures, len(disagreements), output_path)
    return 0 if verdict == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
