"""新方針 pipeline で 16 指標教師データを再生成 (Phase E-1).

既存 generate_training_dataset.py は ImageReader 直 + 単 frame 抽出
で生成されており、新方針 pipeline (state machine + 物理推論)
適用前の認識結果に基づく。これを破棄し、RecognitionPipeline 経由で
STABLE 確定盤面のみから 16 指標を抽出して再生成する。

入力:
    - data/frames/video_NN.mp4
    - data/verify/match_winners_vNN.tsv (勝者ラベル)
    - data/verify/match_boundaries_v?/video_NN/matches.tsv (試合区間)

出力:
    data/training/match_features_phase_e.csv
    既存 learn_weights_v3.py 互換フォーマット (FEATURE_NAMES 列 + label)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_collect_indicator_dataset \
        --videos 1,2,3 --max-matches 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.old.indicators import IndicatorCalculator, IndicatorSet  # noqa: E402
from src.ojama_predictor import OjamaPredictor  # noqa: E402
from src.per_video_model_selector import select_phase_b_model  # noqa: E402
from src.probabilistic_board import ProbabilisticBoard  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.old.rotation_tracker import RotationTracker  # noqa: E402
from scripts.old.generate_training_dataset import (  # noqa: E402
    DEFAULT_TIME_PHASES, DEFAULT_WINNERS_DIR, FEATURE_NAMES, MatchMeta,
    MIN_MATCH_DURATION_SEC, TIME_PHASE_END_MINUS, TIME_PHASE_MIDPOINT,
    TIME_PHASE_MID_MINUS, TIME_PHASE_MID_PLUS, TIME_PHASE_START_PLUS,
    extract_feature_diff, load_boundaries, load_winners,
)


# D-C: time_phase の相対化.
# 元の compute_sample_time は絶対秒 (start+20s 等) で計算していたため、
# 短試合 (例: 30 秒) と長試合 (200 秒) で「序盤+20s」の意味が変わる。
# 相対 % に変えて短試合・長試合で同じ意味の phase になるようにする。
RELATIVE_PHASE_PCT: dict[str, float] = {
    TIME_PHASE_START_PLUS: 0.10,
    TIME_PHASE_MID_MINUS: 0.30,
    TIME_PHASE_MIDPOINT: 0.50,
    TIME_PHASE_MID_PLUS: 0.70,
    TIME_PHASE_END_MINUS: 0.95,
}


def compute_sample_time(meta: MatchMeta, phase: str) -> float:  # noqa: F811
    """試合進捗 % ベースの絶対時刻を返す (D-C 相対化版)."""
    pct = RELATIVE_PHASE_PCT.get(phase)
    if pct is None:
        raise ValueError(f"未知の time_phase: {phase}")
    duration = meta.end_sec - meta.start_sec
    t = meta.start_sec + duration * pct
    # 境界 ±1 秒の安全マージン
    return max(meta.start_sec + 1.0, min(meta.end_sec - 1.0, t))


# v5 → v4 fallback で match boundaries を解決する.
# generate_training_dataset.collect_match_meta は v4 だけを見るので、
# 16/19 動画 (v04〜v19) が無視される。phase_e ではこれを直す。
_BOUNDARY_DIRS: tuple[Path, ...] = (
    _ROOT / "data" / "verify" / "match_boundaries_v5",
    _ROOT / "data" / "verify" / "match_boundaries_v4",
)


def collect_match_meta(video_id: str) -> list[MatchMeta]:  # noqa: F811
    """v5 → v4 fallback 版."""
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


DEFAULT_OUTPUT_CSV: Path = _ROOT / "data" / "training" / "match_features_phase_e.csv"
DEFAULT_FPS: float = 5.0  # サンプリング fps (eval は 5fps で十分)

# Phase G C-1: 確率モードのデフォルト Monte Carlo サンプル数。
# 多すぎると重くなり、少なすぎると分散の見積もりが粗くなる。
# 論理上 chain.py の PROBABILISTIC_DEFAULT_SAMPLES と揃えてあるが、
# CLI で上書き可能 (`--n-samples`)。
DEFAULT_N_SAMPLES: int = 10
# 確率モード OFF (default) で従来挙動と完全互換。
DEFAULT_PROBABILISTIC: bool = False


def stream_match_with_pipeline(
    video_path: Path, meta: MatchMeta, cnn_model: Path | None,
    fps_sample: float = DEFAULT_FPS,
    *,
    probabilistic: bool = DEFAULT_PROBABILISTIC,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> dict[float, tuple[IndicatorSet, IndicatorSet]] | None:
    """1 試合区間を pipeline で通し、STABLE 確定 frame の (indicator_1p,
    indicator_2p) を時刻ごとに辞書化して返す.

    両側 STABLE のときのみ採取。

    Args:
        probabilistic: True で `compute_all_probabilistic` 経由 (Phase G C-1)。
            False (デフォルト) で従来の `compute_all`。
        n_samples: probabilistic=True 時の Monte Carlo サンプル数。
    """
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=2,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    calc = IndicatorCalculator()
    # C-2 (W-γ): score 差分から予告おじゃま個数を時系列追跡
    ojama_pred = OjamaPredictor()
    # Phase F (B-4): 1P/2P それぞれで RotationTracker を持ち、
    # STABLE 確定時に board を update して rotation_score を取得
    rot_1p = RotationTracker()
    rot_2p = RotationTracker()
    interval = 1.0 / fps_sample
    t = meta.start_sec
    frame_idx = 0
    last_score_key: tuple[str, str] | None = None
    last_indicators: tuple[IndicatorSet, IndicatorSet] | None = None
    snapshots: dict[float, tuple[IndicatorSet, IndicatorSet]] = {}

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

        # C-2: 各 frame で score_delta を OjamaPredictor に渡し pending 追跡
        ojama_pred.update(
            p1_score_delta=result.p1.score_delta,
            p2_score_delta=result.p2.score_delta,
            p1_board=result.p1.confirmed_board,
            p2_board=result.p2.confirmed_board,
        )

        if (
            result.p1.state == BoardState.STABLE
            and result.p2.state == BoardState.STABLE
            and result.p1.confirmed_board is not None
            and result.p2.confirmed_board is not None
        ):
            try:
                key = (
                    result.p1.confirmed_board.to_json(),
                    result.p2.confirmed_board.to_json(),
                )
                if key != last_score_key:
                    sm_1p_ctx = pipe._sm_1p.context  # type: ignore[attr-defined]
                    sm_2p_ctx = pipe._sm_2p.context  # type: ignore[attr-defined]
                    p1_next = (
                        tuple(sm_1p_ctx.next_queue[-1])
                        if sm_1p_ctx.next_queue else None
                    )
                    p2_next = (
                        tuple(sm_2p_ctx.next_queue[-1])
                        if sm_2p_ctx.next_queue else None
                    )
                    p1_dnext = (
                        tuple(sm_1p_ctx.next_queue[-2])
                        if len(sm_1p_ctx.next_queue) >= 2 else None
                    )
                    p2_dnext = (
                        tuple(sm_2p_ctx.next_queue[-2])
                        if len(sm_2p_ctx.next_queue) >= 2 else None
                    )
                    # C-2: 各サイドの予告おじゃまを incoming_ojama として渡す
                    inc_1p = ojama_pred.pending_for("1P")
                    inc_2p = ojama_pred.pending_for("2P")
                    # Phase F (B-4): STABLE 盤面更新ごとに RotationTracker
                    # に board を渡し、rotation_skill スコアを取得
                    rot_1p.update(result.p1.confirmed_board)
                    rot_2p.update(result.p2.confirmed_board)
                    if probabilistic:
                        # W-α 統合: prob_board が None なら from_board で fallback
                        pb1 = result.p1.prob_board or (
                            ProbabilisticBoard.from_board(
                                result.p1.confirmed_board,
                            )
                        )
                        pb2 = result.p2.prob_board or (
                            ProbabilisticBoard.from_board(
                                result.p2.confirmed_board,
                            )
                        )
                        ind1 = calc.compute_all_probabilistic(
                            pb1,
                            next_pair=p1_next, dnext_pair=p1_dnext,
                            incoming_ojama=inc_1p,
                            opponent_board=result.p2.confirmed_board,
                            rotation_score=rot_1p.score,
                            n_samples=n_samples,
                        )
                        ind2 = calc.compute_all_probabilistic(
                            pb2,
                            next_pair=p2_next, dnext_pair=p2_dnext,
                            incoming_ojama=inc_2p,
                            opponent_board=result.p1.confirmed_board,
                            rotation_score=rot_2p.score,
                            n_samples=n_samples,
                        )
                    else:
                        ind1 = calc.compute_all(
                            result.p1.confirmed_board,
                            next_pair=p1_next, dnext_pair=p1_dnext,
                            incoming_ojama=inc_1p,
                            opponent_board=result.p2.confirmed_board,
                            rotation_score=rot_1p.score,
                        )
                        ind2 = calc.compute_all(
                            result.p2.confirmed_board,
                            next_pair=p2_next, dnext_pair=p2_dnext,
                            incoming_ojama=inc_2p,
                            opponent_board=result.p1.confirmed_board,
                            rotation_score=rot_2p.score,
                        )
                    last_indicators = (ind1, ind2)
                    last_score_key = key
            except Exception:
                pass

        if last_indicators is not None:
            snapshots[t] = last_indicators
        frame_idx += 1
        t += interval
    cap.release()
    return snapshots


def find_indicator_at_time(
    snapshots: dict[float, tuple[IndicatorSet, IndicatorSet]],
    target_t: float, max_lookback_sec: float = 3.0,
) -> tuple[IndicatorSet, IndicatorSet] | None:
    """target_t 以前の最も近い snapshot を返す (max_lookback_sec 内)."""
    best_t = None
    for snap_t in snapshots:
        if snap_t > target_t:
            continue
        if (target_t - snap_t) > max_lookback_sec:
            continue
        if best_t is None or snap_t > best_t:
            best_t = snap_t
    if best_t is None:
        return None
    return snapshots[best_t]


def process_video(
    video_id: int, max_matches: int,
    fps_sample: float, time_phases: list[str],
    *,
    probabilistic: bool = DEFAULT_PROBABILISTIC,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[dict]:
    """1 動画分の試合をすべて処理して、行リストを返す."""
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return []
    metas = collect_match_meta(f"{video_id:02d}")
    if not metas:
        print(f"[skip] v{video_id:02d}: no match meta")
        return []
    metas = [
        m for m in metas if (m.end_sec - m.start_sec) >= MIN_MATCH_DURATION_SEC
    ]
    if max_matches > 0:
        metas = metas[:max_matches]

    cnn_model_str = select_phase_b_model(video_id)
    cnn_model = Path(cnn_model_str) if cnn_model_str else None

    rows: list[dict] = []
    for n, meta in enumerate(metas, 1):
        snapshots = stream_match_with_pipeline(
            video_path, meta, cnn_model, fps_sample=fps_sample,
            probabilistic=probabilistic, n_samples=n_samples,
        )
        if not snapshots:
            print(f"[empty] v{video_id:02d}_m{meta.match_idx}: no STABLE")
            continue
        for phase in time_phases:
            t = compute_sample_time(meta, phase)
            ind_pair = find_indicator_at_time(snapshots, t)
            if ind_pair is None:
                continue
            ind1, ind2 = ind_pair
            features = extract_feature_diff(ind1, ind2)
            row = {
                "video_id": f"{video_id:02d}",
                "match_idx": meta.match_idx,
                "time_phase": phase,
            }
            for name in FEATURE_NAMES:
                row[name] = features.get(name, 0.0)
            row["label"] = 1 if meta.winner == "1P" else -1
            rows.append(row)
        if n % 5 == 0:
            print(f"  [v{video_id:02d}] {n}/{len(metas)} matches")
    print(
        f"[done] v{video_id:02d}: {len(rows)} rows from {len(metas)} matches"
    )
    return rows


def _process_video_shard(args_tuple: tuple) -> tuple[int, int]:
    """1 動画分 → shard csv 出力. multiprocessing.Pool 用.

    args_tuple は後方互換のため可変長 tuple として受け取り、6/7 要素両方に対応:
        旧: (vid, max_matches, fps, time_phases, shard_dir)
        新: (vid, max_matches, fps, time_phases, shard_dir,
             probabilistic, n_samples)
    """
    if len(args_tuple) >= 7:
        (
            vid, max_matches, fps, time_phases, shard_dir,
            probabilistic, n_samples,
        ) = args_tuple[:7]
    else:
        (vid, max_matches, fps, time_phases, shard_dir) = args_tuple[:5]
        probabilistic = DEFAULT_PROBABILISTIC
        n_samples = DEFAULT_N_SAMPLES
    shard_path = shard_dir / f"shard_v{vid:02d}.csv"
    if shard_path.exists():
        try:
            with shard_path.open("r", encoding="utf-8") as f:
                n_existing = sum(1 for _ in f) - 1
            if n_existing > 0:
                print(
                    f"[skip] v{vid:02d}: shard exists ({n_existing} rows)"
                )
                return (vid, n_existing)
        except Exception:
            pass
    rows = process_video(
        vid, max_matches, fps, time_phases,
        probabilistic=probabilistic, n_samples=n_samples,
    )
    if not rows:
        return (vid, 0)
    fieldnames = (
        ["video_id", "match_idx", "time_phase"]
        + list(FEATURE_NAMES) + ["label"]
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    with shard_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return (vid, len(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument(
        "--max-matches", type=int, default=0,
        help="動画あたり最大試合数 (0=無制限)",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--time-phases", nargs="+", default=list(DEFAULT_TIME_PHASES),
    )
    parser.add_argument(
        "--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="並列ワーカー数 (>=2 で multiprocessing 起動、shard 経由で再開可能)",
    )
    parser.add_argument(
        "--shard-dir", type=Path,
        default=_ROOT / "data" / "training" / "phase_e_shards",
        help="動画単位 shard csv 出力先 (workers>=2 時に使用)",
    )
    parser.add_argument(
        "--probabilistic", action="store_true",
        help="ProbabilisticBoard 経由で 16 指標を計算 (Phase G C-1)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help="--probabilistic 時の Monte Carlo サンプル数",
    )
    args = parser.parse_args()

    if args.videos:
        target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    else:
        target_ids = list(range(1, 20))

    if args.workers >= 2:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        args.shard_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            (vid, args.max_matches, args.fps, list(args.time_phases),
             args.shard_dir, args.probabilistic, args.n_samples)
            for vid in target_ids
        ]
        print(
            f"[parallel] workers={args.workers} videos={len(target_ids)}"
            f" -> shard_dir={to_windows_path(args.shard_dir)}"
        )
        with ctx.Pool(processes=args.workers) as pool:
            for vid, n in pool.imap_unordered(_process_video_shard, tasks):
                print(f"  [shard] v{vid:02d}: {n} rows")
        # shard 統合
        all_rows: list[dict] = []
        fieldnames = (
            ["video_id", "match_idx", "time_phase"]
            + list(FEATURE_NAMES) + ["label"]
        )
        for vid in target_ids:
            shard_path = args.shard_dir / f"shard_v{vid:02d}.csv"
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                all_rows.extend(list(r))
    else:
        all_rows = []
        for vid in target_ids:
            rows = process_video(
                vid, args.max_matches, args.fps, args.time_phases,
                probabilistic=args.probabilistic,
                n_samples=args.n_samples,
            )
            all_rows.extend(rows)
        fieldnames = (
            ["video_id", "match_idx", "time_phase"]
            + list(FEATURE_NAMES) + ["label"]
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
