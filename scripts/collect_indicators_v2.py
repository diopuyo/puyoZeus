"""指標 v2 (第1バッチ) 算出パイプライン — 1 動画 → dataset CSV。

`docs/INDICATOR_V2_MEASUREMENT_SPEC_2026-06-17.md` のパイプライン仕様に従う。

処理概要:
    - RecognitionPipeline.load_default (visualize_recognition と同じ load_default 経路)
      で 1 動画を frame 単位処理。--no-per-video-hsv 相当で自動 HSV のみ動作。
    - 両者 STABLE snapshot で指標を算出し dataset 行 (CSV) を出力。
    - OjamaAccountingTracker を viz 統合と同様に駆動して net収支/forecast snapshot を得る。

前処理 (仕様書 4):
    - STABLE 時のみ算出 (両者個別。各 side が STABLE のフレームのみ行を出力)。
    - 試合境界 (score 大幅減少) で game_idx を進め、手数をリセット (pipeline 内部で自動)。
    - 連続フレーム間引き: 同一 STABLE 区間 (盤面が変わらない連続フレーム) は 1 回のみ出力。
    - 全消し直後フレーム除外: 盤面ぷよ数 0 (= 全消し / 試合開始直後) の STABLE は除外。

各行メタ: video_id / game_idx / t_sec / frame / 手数(tsumo) / side(1P/2P)。

使い方 (短尺検証):
    python -m scripts.collect_indicators_v2 \
        --video data/frames/video_124_4min.mp4 \
        --out data/indicators_v2/video_124_4min.csv --max-sec 60
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    OjamaAccountingTracker,
    OjamaAccountSnapshot,
)
from src.recognition_pipeline import RecognitionPipeline, SideResult  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# 出力解像度 (認識は 1920x1080 前提)
TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0
# 試合境界検知: score がこの値以上減少したら新しい試合とみなす (会計と同基準)
SCORE_RESET_THRESHOLD: int = 500


# ============================
# CSV 列定義 (順序固定)
# ============================

# メタ列
META_COLUMNS: tuple[str, ...] = (
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side",
)
# 指標列 (順序保持。新指標は末尾に追加)
INDICATOR_COLUMNS: tuple[str, ...] = (
    # ① 進行度
    "tsumo_count_rate", "tsumo_count_raw",
    "board_puyo_total", "board_puyo_total_raw",
    "board_color_puyo_total", "board_color_puyo_total_raw",
    "margin_time_rate", "margin_time_rate_raw",
    # ② 占有・危険
    "max_column_height", "max_column_height_raw",
    "column_bumpiness", "column_bumpiness_raw",
    "death_margin", "death_margin_raw",
    "death_margin_neighbor", "death_margin_neighbor_raw",
    # ③ 火力・潜在
    "current_max_chain", "current_max_chain_raw",
    "immediate_fire_power", "immediate_fire_power_raw",
    "chain_efficiency", "chain_efficiency_raw",
    "min_puyos_to_ignite", "min_puyos_to_ignite_raw",
    "conn_pair_count", "conn_triple_count", "conn_max_group_size",
    "second_chain_potential", "second_chain_potential_raw",
    # ④ お邪魔
    "ojama_net_balance", "ojama_net_balance_raw",
    "ojama_forecast", "ojama_forecast_raw",
    "board_ojama_count", "board_ojama_count_raw",
    # ⑤ テンポ
    "chain_duration_sec", "chain_duration_source",
    # ⑥ 受け力
    "dig_resistance", "dig_resistance_raw",
    "absorption_capacity", "absorption_capacity_raw",
)
ALL_COLUMNS: tuple[str, ...] = META_COLUMNS + INDICATOR_COLUMNS


@dataclass
class _SideTracker:
    """1 side の前処理状態 (間引き・全消し検知用)。"""
    game_idx: int = 0
    prev_score: int | None = None
    last_emitted_grid: bytes | None = None  # 直前に出力した盤面 (間引き)


def _compute_row(
    video_id: str,
    side_label: str,
    side: SideResult,
    board: Board,
    t_sec: float,
    frame_idx: int,
    tsumo: int,
    elapsed_sec: float,
    snap: OjamaAccountSnapshot,
) -> dict[str, object]:
    """1 STABLE snapshot から指標を算出し CSV 行 dict を返す。"""
    is_p1 = side_label == "1P"
    net = snap.net_balance_capped if is_p1 else -snap.net_balance_capped
    forecast = snap.forecast_p1 if is_p1 else snap.forecast_p2
    # ⑤ 連鎖所要時間: chain_event があれば観測、無ければ推定。
    dur, dur_src = _chain_duration(side)
    total_conn, _ = iv.connectivity_observation(board)
    row: dict[str, object] = {
        "video_id": video_id,
        "side": side_label,
        "t_sec": round(t_sec, 3),
        "frame": frame_idx,
        "tsumo": tsumo,
    }
    _fill_indicator_columns(row, board, tsumo, elapsed_sec, net, forecast, total_conn)
    row["chain_duration_sec"] = round(dur.raw, 3) if dur is not None else 0.0
    row["chain_duration_source"] = dur_src
    return row


def _fill_indicator_columns(
    row: dict[str, object],
    board: Board,
    tsumo: int,
    elapsed_sec: float,
    net: int,
    forecast: int,
    total_conn: iv.GroupObservation,
) -> None:
    """指標値を row dict に書き込む (chain_duration を除く)。"""
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
    row.update({
        "tsumo_count_rate": tc.score, "tsumo_count_raw": tc.raw,
        "board_puyo_total": bp.score, "board_puyo_total_raw": bp.raw,
        "board_color_puyo_total": bc.score, "board_color_puyo_total_raw": bc.raw,
        "margin_time_rate": mt.score, "margin_time_rate_raw": mt.raw,
        "max_column_height": mh.score, "max_column_height_raw": mh.raw,
        "column_bumpiness": bm.score, "column_bumpiness_raw": bm.raw,
        "death_margin": dm.score, "death_margin_raw": dm.raw,
        "death_margin_neighbor": dn.score, "death_margin_neighbor_raw": dn.raw,
        "current_max_chain": cm.score, "current_max_chain_raw": cm.raw,
        "immediate_fire_power": ifp.score, "immediate_fire_power_raw": ifp.raw,
        "chain_efficiency": ce.score, "chain_efficiency_raw": ce.raw,
        "min_puyos_to_ignite": mi.score, "min_puyos_to_ignite_raw": mi.raw,
        "conn_pair_count": total_conn.pair_count,
        "conn_triple_count": total_conn.triple_count,
        "conn_max_group_size": total_conn.max_group_size,
        "second_chain_potential": sc.score, "second_chain_potential_raw": sc.raw,
        "ojama_net_balance": nb.score, "ojama_net_balance_raw": nb.raw,
        "ojama_forecast": fc.score, "ojama_forecast_raw": fc.raw,
        "board_ojama_count": bo.score, "board_ojama_count_raw": bo.raw,
        "dig_resistance": dr.score, "dig_resistance_raw": dr.raw,
        "absorption_capacity": ab.score, "absorption_capacity_raw": ab.raw,
    })


def _chain_duration(side: SideResult) -> tuple[iv.IndicatorV2Value, str]:
    """連鎖所要時間を観測優先・推定フォールバックで返す。

    Returns:
        (IndicatorV2Value, source) where source ∈ {"observed", "estimated", "none"}。
    """
    ev = side.chain_event
    if ev is not None:
        observed = iv.chain_duration_observed(ev.trigger_sec, ev.end_sec)
        if observed is not None:
            return observed, "observed"
        return iv.chain_duration_estimated(ev.chain_count), "estimated"
    return iv.IndicatorV2Value(score=0.0, raw=0.0), "none"


def _update_game_idx(
    tracker: _SideTracker, score: int | None,
) -> None:
    """score 大幅減少で game_idx を進める (試合境界分割)。"""
    if score is not None and tracker.prev_score is not None:
        if tracker.prev_score - score >= SCORE_RESET_THRESHOLD:
            tracker.game_idx += 1
    if score is not None:
        tracker.prev_score = score


def _should_emit(
    tracker: _SideTracker, side: SideResult, board: Board,
) -> bool:
    """この STABLE snapshot を出力すべきか (間引き + 全消し除外)。"""
    if side.state != BoardState.STABLE or board is None:
        return False
    # 全消し直後 / 試合開始直後 (盤面ぷよ 0) は除外
    if board.count_puyos() == 0:
        return False
    # 連続フレーム間引き: 直前に出力した盤面と同一なら skip
    grid_bytes = board._grid.tobytes()
    if grid_bytes == tracker.last_emitted_grid:
        return False
    return True


def _process_side(
    video_id: str,
    side_label: str,
    side: SideResult,
    tracker: _SideTracker,
    pipeline: RecognitionPipeline,
    tsumo_tracker: OjamaAccountingTracker,
    t_sec: float,
    frame_idx: int,
    snap: OjamaAccountSnapshot,
    rows: list[dict[str, object]],
) -> None:
    """1 side を処理し、出力対象なら rows に行を追加する。"""
    _update_game_idx(tracker, side.score)
    board = side.confirmed_board
    if board is None or not _should_emit(tracker, side, board):
        return
    elapsed = tsumo_tracker._elapsed(t_sec)  # 試合相対経過秒 (マージンタイム用)
    tsumo = pipeline.tsumo_count(side_label)
    row = _compute_row(
        video_id, side_label, side, board, t_sec, frame_idx,
        tsumo, elapsed, snap,
    )
    row["game_idx"] = tracker.game_idx
    rows.append(row)
    tracker.last_emitted_grid = board._grid.tobytes()


def collect(
    video_path: Path,
    out_path: Path,
    max_sec: float = 0.0,
    sample_interval_sec: float = 0.0,
) -> int:
    """1 動画を処理して指標 dataset CSV を出力する。

    Returns:
        出力した行数。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video_path}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_sec > 0:
        n_frames = min(n_frames, int(max_sec * fps))
    video_id = video_path.stem

    # visualize_recognition と同じ load_default 経路 (自動 HSV のみ = per-video inject なし)
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

    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    tracker_p1 = _SideTracker()
    tracker_p2 = _SideTracker()
    rows: list[dict[str, object]] = []
    sample_interval_frames = max(1, int(round(sample_interval_sec * fps)))

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        if fi % sample_interval_frames != 0:
            continue
        result = pipeline.update(fi, t_sec, frame)
        # --- お邪魔会計駆動 (viz 統合と同様) ---
        snap = _drive_ojama(
            ojama_tracker, result.p1, result.p2,
            prev_state_p1, prev_state_p2, t_sec,
        )
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state
        # --- 各 side 処理 ---
        _process_side(
            video_id, "1P", result.p1, tracker_p1, pipeline,
            ojama_tracker, t_sec, fi, snap, rows,
        )
        _process_side(
            video_id, "2P", result.p2, tracker_p2, pipeline,
            ojama_tracker, t_sec, fi, snap, rows,
        )
    cap.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ALL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _drive_ojama(
    tracker: OjamaAccountingTracker,
    p1: SideResult,
    p2: SideResult,
    prev_p1: BoardState,
    prev_p2: BoardState,
    t_sec: float,
) -> OjamaAccountSnapshot:
    """OjamaAccountingTracker を on_state_transition / on_tsumo_settled で駆動。"""
    tsumo_settled_p1 = (
        prev_p1 == BoardState.TSUMO_FALL and p1.state == BoardState.STABLE
    )
    tsumo_settled_p2 = (
        prev_p2 == BoardState.TSUMO_FALL and p2.state == BoardState.STABLE
    )
    tracker.on_state_transition("p1", prev_p1, p1.state, p1.score, t_sec)
    tracker.on_state_transition("p2", prev_p2, p2.state, p2.score, t_sec)
    if tsumo_settled_p1:
        tracker.on_tsumo_settled("p1", t_sec)
    if tsumo_settled_p2:
        tracker.on_tsumo_settled("p2", t_sec)
    return tracker.get_snapshot(t_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="指標 v2 dataset 収集")
    parser.add_argument("--video", type=Path, required=True, help="入力動画")
    parser.add_argument("--out", type=Path, required=True, help="出力 CSV パス")
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="処理する最大秒数 (0 = 全長)",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.0,
        help="認識サンプル間隔秒 (0 = 全フレーム)",
    )
    args = parser.parse_args()
    n = collect(args.video, args.out, args.max_sec, args.sample_interval)
    print(f"[collect] {args.video.name} -> {args.out} : {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
