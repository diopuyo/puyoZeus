"""Phase H4.2: 盤面 (raw 6×13) 付き Phase H2 データ収集.

Phase H2 の indicator-CSV では board state が保存されないため、
Phase H4.2 (CNN end-to-end) 用に board state を NPZ で並列保存する。

設計:
    A. CSV (indicator features) は既存 phase_h2_collect_indicator_dataset.py
       と同形式で出力 (差分なし、列順互換).
    B. 動画ごとに 1 NPZ にまとめて board を永続化:
        data/training/phase_h2_boards/v01.npz
            keys:
              video_ids:     (N,) str
              match_indices: (N,) int
              frame_indices: (N,) int
              timestamps:    (N,) float
              labels:        (N,) int     1=1P 勝利, -1=2P 勝利
              p1_boards:     (N, 13, 6) uint8
              p2_boards:     (N, 13, 6) uint8
       N は CSV と同じ行数になる. CSV 行と board NPZ 行は同インデックス対応.

backwards compat: 既存 scripts/phase_h2_collect_indicator_dataset.py は触らない.
                 こちらは独立した新規 script.

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_h2_collect_board \
        --videos 1,4,7,12,20,22,28,40,51,57,70,89 \
        --workers 6 \
        --board-dir data/training/phase_h2_boards \
        --out-csv data/training/match_features_phase_h2_quick_with_board.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.indicators import IndicatorCalculator  # noqa: E402
from src.ojama_predictor import OjamaPredictor  # noqa: E402
from src.per_video_model_selector import select_phase_b_model  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.rotation_tracker import RotationTracker  # noqa: E402
from src.timeseries_indicator_wrapper import (  # noqa: E402
    DEFAULT_HISTORY_SEC,
    TimeseriesWrapper,
)
from scripts.generate_training_dataset import (  # noqa: E402
    MatchMeta,
    MIN_MATCH_DURATION_SEC,
    load_boundaries,
    load_winners,
    DEFAULT_WINNERS_DIR,
)
from scripts.phase_h2_collect_indicator_dataset import (  # noqa: E402
    ALL_FEATURE_NAMES,
    DEFAULT_FRAME_INTERVAL_SEC,
    DEFAULT_QUICK_VIDEOS,
    _BOUNDARY_DIRS,
    _compute_pair,
    compute_interaction_features,
    expand_diff_features,
)


# ============================
# 定数
# ============================

# 盤面 NPZ 出力先
DEFAULT_BOARD_DIR: Path = (
    _ROOT / "data" / "training" / "phase_h2_boards"
)
# CSV (board あり版)
DEFAULT_OUTPUT_CSV: Path = (
    _ROOT / "data" / "training" / "match_features_phase_h2_quick_with_board.csv"
)
# Board grid 形状
BOARD_GRID_ROWS: int = 13
BOARD_GRID_COLS: int = 6


# ============================
# meta 取得 (phase_h2 と同じ logic、内部関数として複製)
# ============================


def collect_match_meta(video_id: str) -> list[MatchMeta]:
    """v5 → v4 fallback で 1 動画分の MatchMeta リストを返す."""
    boundaries: dict[int, tuple[float, float]] = {}
    for d in _BOUNDARY_DIRS:
        path = d / f"video_{video_id}" / "matches.tsv"
        b = load_boundaries(path)
        if b:
            boundaries = b
            break
    winners = load_winners(
        DEFAULT_WINNERS_DIR / f"match_winners_v{video_id}.tsv",
    )
    metas: list[MatchMeta] = []
    for idx in sorted(boundaries.keys() & winners.keys()):
        start, end = boundaries[idx]
        if end - start < MIN_MATCH_DURATION_SEC:
            continue
        metas.append(MatchMeta(
            video_id=video_id, match_idx=idx,
            start_sec=start, end_sec=end, winner=winners[idx],
        ))
    return metas


# ============================
# 1 試合区間処理 (盤面付き)
# ============================


def _grid_or_zero(board) -> np.ndarray:
    """Board → (ROWS, COLS) uint8 ndarray. None なら zero 返却."""
    if board is None:
        return np.zeros((BOARD_GRID_ROWS, BOARD_GRID_COLS), dtype=np.uint8)
    grid = board.to_dict()["grid"]
    return np.asarray(grid, dtype=np.uint8)


def stream_match_with_board(
    video_path: Path, meta: MatchMeta, cnn_model: Path | None,
    *,
    frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
    history_sec: float = DEFAULT_HISTORY_SEC,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """1 試合区間を pipeline で通し、(rows, p1_grids, p2_grids) を返す.

    rows[i] と p1_grids[i] / p2_grids[i] は同フレームに対応 (1:1).
    """
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=2, load_score_ocr=True,
        enable_chain_tracker=True, cnn_model_path=cnn_model,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ([], [], [])

    calc = IndicatorCalculator()
    ojama_pred = OjamaPredictor()
    rot_1p = RotationTracker()
    rot_2p = RotationTracker()
    ts_1p = TimeseriesWrapper(history_sec=history_sec)
    ts_2p = TimeseriesWrapper(history_sec=history_sec)

    rows: list[dict] = []
    p1_grids: list[np.ndarray] = []
    p2_grids: list[np.ndarray] = []
    last_score_key: tuple[str, str] | None = None
    t = meta.start_sec
    frame_idx = 0
    while t < meta.end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(frame_idx, t, frame)
        ojama_pred.update(
            p1_score_delta=result.p1.score_delta,
            p2_score_delta=result.p2.score_delta,
            p1_board=result.p1.confirmed_board,
            p2_board=result.p2.confirmed_board,
        )
        emit = _try_emit_row_with_board(
            t, frame_idx, meta, result, calc,
            ojama_pred, rot_1p, rot_2p, ts_1p, ts_2p, pipe, last_score_key,
        )
        if emit is not None:
            row, key, p1_grid, p2_grid = emit
            rows.append(row)
            p1_grids.append(p1_grid)
            p2_grids.append(p2_grid)
            last_score_key = key
        t += frame_interval_sec
        frame_idx += 1
    cap.release()
    return (rows, p1_grids, p2_grids)


def _try_emit_row_with_board(
    t: float, frame_idx: int, meta: MatchMeta, result, calc, ojama_pred,
    rot_1p, rot_2p, ts_1p: TimeseriesWrapper, ts_2p: TimeseriesWrapper,
    pipe, last_score_key: tuple[str, str] | None,
) -> tuple[dict, tuple[str, str], np.ndarray, np.ndarray] | None:
    """両側 STABLE で盤面が変化した frame で (row, key, p1_grid, p2_grid) を emit.

    None なら emit なし. row 部分は既存 phase_h2 と同形式.
    """
    if not (
        result.p1.state == BoardState.STABLE
        and result.p2.state == BoardState.STABLE
        and result.p1.confirmed_board is not None
        and result.p2.confirmed_board is not None
    ):
        return None
    try:
        key = (
            result.p1.confirmed_board.to_json(),
            result.p2.confirmed_board.to_json(),
        )
        if key == last_score_key:
            return None
        ind1, ind2 = _compute_pair(
            result, calc, ojama_pred, rot_1p, rot_2p, pipe,
        )
        ts_1p.update(t, ind1)
        ts_2p.update(t, ind2)
        feats_1p = ts_1p.expand_features(ind1)
        feats_2p = ts_2p.expand_features(ind2)
        diff_feats = expand_diff_features(feats_1p, feats_2p)
        inter_feats = compute_interaction_features(ind1, ind2)
        row: dict = {
            "video_id": meta.video_id,
            "match_idx": meta.match_idx,
            "frame_idx": frame_idx,
            "timestamp": t,
        }
        row.update(diff_feats)
        row.update(inter_feats)
        row["label"] = 1 if meta.winner == "1P" else -1
        p1_grid = _grid_or_zero(result.p1.confirmed_board)
        p2_grid = _grid_or_zero(result.p2.confirmed_board)
        return (row, key, p1_grid, p2_grid)
    except Exception:
        return None


# ============================
# 1 動画処理
# ============================


def process_video_with_board(
    video_id: int, max_matches: int,
    frame_interval_sec: float, history_sec: float,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """1 動画を処理し (rows, p1_grids, p2_grids) を返す."""
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return ([], [], [])
    metas = collect_match_meta(f"{video_id:02d}")
    if not metas:
        print(f"[skip] v{video_id:02d}: no match meta")
        return ([], [], [])
    if max_matches > 0:
        metas = metas[:max_matches]
    cnn_model_str = select_phase_b_model(video_id)
    cnn_model = Path(cnn_model_str) if cnn_model_str else None

    all_rows: list[dict] = []
    all_p1: list[np.ndarray] = []
    all_p2: list[np.ndarray] = []
    for n, meta in enumerate(metas, 1):
        rows, p1_grids, p2_grids = stream_match_with_board(
            video_path, meta, cnn_model,
            frame_interval_sec=frame_interval_sec,
            history_sec=history_sec,
        )
        if not rows:
            print(f"[empty] v{video_id:02d}_m{meta.match_idx}: no STABLE")
            continue
        all_rows.extend(rows)
        all_p1.extend(p1_grids)
        all_p2.extend(p2_grids)
        if n % 5 == 0:
            print(
                f"  [v{video_id:02d}] {n}/{len(metas)} matches"
                f" ({len(all_rows)} cumulative rows)"
            )
    print(
        f"[done] v{video_id:02d}: {len(all_rows)} rows from {len(metas)} matches"
    )
    return (all_rows, all_p1, all_p2)


def save_board_npz(
    out_path: Path, rows: list[dict],
    p1_grids: list[np.ndarray], p2_grids: list[np.ndarray],
) -> None:
    """1 動画分の board NPZ を出力する (CSV と行同期)."""
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_ids = np.array([r["video_id"] for r in rows], dtype="<U8")
    match_indices = np.array([int(r["match_idx"]) for r in rows], dtype=np.int32)
    frame_indices = np.array([int(r["frame_idx"]) for r in rows], dtype=np.int32)
    timestamps = np.array([float(r["timestamp"]) for r in rows], dtype=np.float32)
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int8)
    p1_arr = np.stack(p1_grids, axis=0).astype(np.uint8)
    p2_arr = np.stack(p2_grids, axis=0).astype(np.uint8)
    np.savez_compressed(
        out_path,
        video_ids=video_ids,
        match_indices=match_indices,
        frame_indices=frame_indices,
        timestamps=timestamps,
        labels=labels,
        p1_boards=p1_arr,
        p2_boards=p2_arr,
    )


def save_csv_shard(
    shard_path: Path, rows: list[dict],
) -> None:
    """1 動画分の CSV shard を出力する (既存 phase_h2 と同形式)."""
    if not rows:
        return
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["video_id", "match_idx", "frame_idx", "timestamp"]
        + list(ALL_FEATURE_NAMES) + ["label"]
    )
    with shard_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _process_video_task(args_tuple: tuple) -> tuple[int, int]:
    """1 動画分処理. multiprocessing.Pool 用 worker."""
    (
        vid, max_matches, frame_interval_sec, history_sec,
        board_dir, shard_dir,
    ) = args_tuple
    npz_path = board_dir / f"v{vid:02d}.npz"
    shard_path = shard_dir / f"shard_v{vid:02d}.csv"
    # 既存スキップ判定: npz と shard 両方あれば skip
    if npz_path.exists() and shard_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            n_existing = int(data["labels"].shape[0])
            if n_existing > 0:
                print(f"[skip] v{vid:02d}: npz exists ({n_existing} rows)")
                return (vid, n_existing)
        except Exception:
            pass
    rows, p1_grids, p2_grids = process_video_with_board(
        vid, max_matches, frame_interval_sec, history_sec,
    )
    if not rows:
        return (vid, 0)
    save_csv_shard(shard_path, rows)
    save_board_npz(npz_path, rows, p1_grids, p2_grids)
    return (vid, len(rows))


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase H4.2 board 付き Phase H2 データ収集 "
            "(CSV indicator + NPZ raw board)"
        ),
    )
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument(
        "--max-matches", type=int, default=0,
        help="動画あたり最大試合数 (0=無制限)",
    )
    parser.add_argument(
        "--frame-interval-sec", type=float,
        default=DEFAULT_FRAME_INTERVAL_SEC,
        help="サンプリング間隔 (秒、デフォルト 0.6)",
    )
    parser.add_argument(
        "--history-sec", type=float, default=DEFAULT_HISTORY_SEC,
        help="時系列履歴の保持秒数 (デフォルト 30.0)",
    )
    parser.add_argument(
        "--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="並列ワーカー数 (>=2 で multiprocessing 起動)",
    )
    parser.add_argument(
        "--board-dir", type=Path, default=DEFAULT_BOARD_DIR,
        help="動画単位 board NPZ 出力先",
    )
    parser.add_argument(
        "--shard-dir", type=Path,
        default=_ROOT / "data" / "training" / "phase_h4_2_shards",
        help="動画単位 shard CSV 出力先",
    )
    args = parser.parse_args()

    if args.videos:
        target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    else:
        target_ids = list(DEFAULT_QUICK_VIDEOS)

    args.board_dir.mkdir(parents=True, exist_ok=True)
    args.shard_dir.mkdir(parents=True, exist_ok=True)

    if args.workers >= 2:
        all_rows = _run_parallel(args, target_ids)
    else:
        all_rows = _run_serial(args, target_ids)

    fieldnames = (
        ["video_id", "match_idx", "frame_idx", "timestamp"]
        + list(ALL_FEATURE_NAMES) + ["label"]
    )
    if not all_rows:
        print("[empty] no rows")
        return 0
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(
        f"\n[saved] {len(all_rows)} rows -> {to_windows_path(args.out_csv)}"
    )
    print(
        f"[saved] board npz -> {to_windows_path(args.board_dir)}"
    )
    return 0


def _run_parallel(args, target_ids: list[int]) -> list[dict]:
    """並列実行 + shard 統合."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks = [
        (vid, args.max_matches, args.frame_interval_sec,
         args.history_sec, args.board_dir, args.shard_dir)
        for vid in target_ids
    ]
    print(
        f"[parallel] workers={args.workers} videos={len(target_ids)}"
        f" -> board_dir={to_windows_path(args.board_dir)}"
    )
    with ctx.Pool(processes=args.workers) as pool:
        for vid, n in pool.imap_unordered(_process_video_task, tasks):
            print(f"  [done] v{vid:02d}: {n} rows")
    all_rows: list[dict] = []
    for vid in target_ids:
        shard_path = args.shard_dir / f"shard_v{vid:02d}.csv"
        if not shard_path.exists():
            continue
        with shard_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            all_rows.extend(list(r))
    return all_rows


def _run_serial(args, target_ids: list[int]) -> list[dict]:
    """直列実行 (workers=1)."""
    all_rows: list[dict] = []
    for vid in target_ids:
        rows, p1_grids, p2_grids = process_video_with_board(
            vid, args.max_matches, args.frame_interval_sec, args.history_sec,
        )
        if rows:
            shard_path = args.shard_dir / f"shard_v{vid:02d}.csv"
            save_csv_shard(shard_path, rows)
            save_board_npz(
                args.board_dir / f"v{vid:02d}.npz", rows, p1_grids, p2_grids,
            )
        all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    sys.exit(main())
