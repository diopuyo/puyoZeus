"""Phase H2: 時系列展開 + interaction features データ pipeline.

phase_e_collect_indicator_dataset.py をベースに以下を変更:

A. サンプリング: 5 phase × 1 snapshot → 全 STABLE frame で record
B. 各 frame で 45 indicator × 6 軸 = 270 timeseries features を展開
C. interaction features 10 個を追加 → 計 280 features
D. CSV 列: video_id, match_idx, frame_idx, timestamp, [280 features], label

1 試合 30-100 行 × 12 動画 × 平均 80 試合 = 約 30,000-100,000 行想定。
backwards compat: 既存 phase_e_collect は触らない。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_h2_collect_indicator_dataset \
        --videos 1,4,7,12,20,22,28,40,51,57,70,89 \
        --workers 6 \
        --shard-dir data/training/phase_h2_shards \
        --out-csv data/training/match_features_phase_h2_quick.csv
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
from src.indicators import IndicatorCalculator, IndicatorSet  # noqa: E402
from src.ojama_predictor import OjamaPredictor  # noqa: E402
from src.per_video_model_selector import select_phase_b_model  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.rotation_tracker import RotationTracker  # noqa: E402
from src.timeseries_indicator_wrapper import (  # noqa: E402
    DEFAULT_HISTORY_SEC,
    TIMESERIES_FEATURE_NAMES,
    TimeseriesWrapper,
)
from scripts.generate_training_dataset import (  # noqa: E402
    MatchMeta,
    MIN_MATCH_DURATION_SEC,
    load_boundaries,
    load_winners,
    DEFAULT_WINNERS_DIR,
)


# ============================
# 定数
# ============================

# Phase H2 quick mode サンプリング fps (5 fps、EVAL_INTERVAL_SEC=0.6 と整合)
DEFAULT_FRAME_INTERVAL_SEC: float = 0.6

# 出力 CSV (quick mode、12 動画想定)
DEFAULT_OUTPUT_CSV: Path = (
    _ROOT / "data" / "training" / "match_features_phase_h2_quick.csv"
)
# shard ディレクトリ (Phase H1 と分離)
DEFAULT_SHARD_DIR: Path = (
    _ROOT / "data" / "training" / "phase_h2_shards"
)
# v5 → v4 fallback で match boundaries を解決する.
_BOUNDARY_DIRS: tuple[Path, ...] = (
    _ROOT / "data" / "verify" / "match_boundaries_v5",
    _ROOT / "data" / "verify" / "match_boundaries_v4",
)

# 比/差計算の epsilon (ゼロ割回避)
INTERACTION_EPSILON: float = 1e-3

# Phase H2 quick mode 既定動画 (12 動画)
DEFAULT_QUICK_VIDEOS: tuple[int, ...] = (
    1, 4, 7, 12, 20, 22, 28, 40, 51, 57, 70, 89,
)


# ============================
# Interaction feature 定義
# 仕様: self × opp の積/差/比を 10 個明示
# ============================

# 各 interaction feature の (列名, 計算 lambda) リスト。
# lambda 引数は (ind1: IndicatorSet, ind2: IndicatorSet) を取り float を返す。
# なお 1P 視点の opponent_* は既に「相手側の脅威」を含むため、
# 多くの積は 1P 値同士で完結する (compute_all 時に opponent_board 渡している)。


def _val(iset: IndicatorSet, name: str) -> float:
    """IndicatorSet から指標値を取得する (results / 属性 fallback)."""
    if name in iset.results:
        return float(iset.results[name].score)
    if name == "next_acceptance":
        return float(iset.next_acceptance)
    return 0.0


def _interaction_self_main_x_opp_threat(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """1P 視点: 自本線完成度 × 1P から見た相手連鎖脅威."""
    return _val(ind1, "main_chain_maturity") * _val(ind1, "opponent_chain_threat")


def _interaction_chain_power_product(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """maximum_fire_power_1p × maximum_fire_power_2p (双方の威力積)."""
    return _val(ind1, "maximum_fire_power") * _val(ind2, "maximum_fire_power")


def _interaction_self_saturation_x_opp_main(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """1P 飽和連鎖量 × 2P 本線完成度."""
    return _val(ind1, "maximum_fire_power") * _val(ind2, "main_chain_maturity")


def _interaction_self_harass_x_opp_defense(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """1P 催促可能性 × 2P お邪魔体制 (1P が打って 2P が守れるか)."""
    return _val(ind1, "harassment_readiness") * _val(ind2, "ojama_defense_capacity")


def _interaction_density_product(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """上部密度の積 (双方上部混雑 → 上級者対決)."""
    return _val(ind1, "upper_board_density") * _val(ind2, "upper_board_density")


def _interaction_ready_diff(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """連鎖数差: 1P - 2P."""
    return _val(ind1, "ready_chain_count") - _val(ind2, "ready_chain_count")


def _interaction_ignition_ratio(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """発火距離比: 1P / max(2P, eps)."""
    denom = max(_val(ind2, "ignition_distance"), INTERACTION_EPSILON)
    return _val(ind1, "ignition_distance") / denom


def _interaction_self_response_x_opp_threat(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """1P 中盤応答能力 × 1P から見た相手脅威."""
    return _val(ind1, "mid_game_response_capacity") * _val(ind1, "opponent_chain_threat")


def _interaction_chain_duration_diff(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """連鎖時間差: opp_chain_duration_1p - self_chain_duration_1p.

    1P 側の IndicatorSet は 1P 視点で「自連鎖時間」「相手連鎖時間」を持つ.
    """
    return (
        _val(ind1, "opp_chain_duration_frames")
        - _val(ind1, "self_chain_duration_frames")
    )


def _interaction_harass_density_ratio(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> float:
    """催促回数比 (1P / 2P): harass_event_count_30s."""
    denom = max(
        _val(ind2, "harass_event_count_30s"), INTERACTION_EPSILON,
    )
    return _val(ind1, "harass_event_count_30s") / denom


# (列名, 計算関数) 定義 — 順序が CSV 列順を決める
INTERACTION_FEATURES: tuple[tuple[str, callable], ...] = (
    ("self_main_x_opp_threat", _interaction_self_main_x_opp_threat),
    ("self_chain_power_x_opp_chain_power", _interaction_chain_power_product),
    ("self_saturation_x_opp_main", _interaction_self_saturation_x_opp_main),
    ("self_harass_x_opp_defense", _interaction_self_harass_x_opp_defense),
    ("self_density_x_opp_density", _interaction_density_product),
    ("self_ready_x_opp_ready", _interaction_ready_diff),
    ("self_ignition_x_opp_ignition", _interaction_ignition_ratio),
    ("self_response_x_opp_threat", _interaction_self_response_x_opp_threat),
    ("chain_duration_diff", _interaction_chain_duration_diff),
    ("harass_density_ratio", _interaction_harass_density_ratio),
)

INTERACTION_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name, _ in INTERACTION_FEATURES
)

# 全 280 features 列名 (270 timeseries diff + 10 interaction)
ALL_FEATURE_NAMES: tuple[str, ...] = (
    tuple(TIMESERIES_FEATURE_NAMES) + INTERACTION_FEATURE_NAMES
)


# ============================
# meta 取得 (phase_e と同じく v5 → v4 fallback)
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
# diff feature 抽出 (270 列 + 10 interaction)
# ============================


def expand_diff_features(
    feats_1p: dict[str, float], feats_2p: dict[str, float],
) -> dict[str, float]:
    """270 timeseries features 1P - 2P 差分を返す."""
    return {
        name: feats_1p.get(name, 0.0) - feats_2p.get(name, 0.0)
        for name in TIMESERIES_FEATURE_NAMES
    }


def compute_interaction_features(
    ind1: IndicatorSet, ind2: IndicatorSet,
) -> dict[str, float]:
    """10 個の interaction features を計算."""
    return {name: fn(ind1, ind2) for name, fn in INTERACTION_FEATURES}


# ============================
# 1 試合区間処理 (全 STABLE frame で record)
# ============================


def stream_match_for_h2(
    video_path: Path, meta: MatchMeta, cnn_model: Path | None,
    *,
    frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
    history_sec: float = DEFAULT_HISTORY_SEC,
) -> list[dict] | None:
    """1 試合区間を pipeline で通し、全 STABLE frame の 280 features 行を返す.

    両側 STABLE 確定 + 盤面が前 STABLE と異なる時のみ record。
    """
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=2, load_score_ocr=True,
        enable_chain_tracker=True, cnn_model_path=cnn_model,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    calc = IndicatorCalculator()
    ojama_pred = OjamaPredictor()
    rot_1p = RotationTracker()
    rot_2p = RotationTracker()
    # 1P/2P それぞれ TimeseriesWrapper を持って履歴管理
    ts_1p = TimeseriesWrapper(history_sec=history_sec)
    ts_2p = TimeseriesWrapper(history_sec=history_sec)

    rows: list[dict] = []
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
        row = _try_emit_row(
            t, frame_idx, meta, result, calc,
            ojama_pred, rot_1p, rot_2p, ts_1p, ts_2p, pipe, last_score_key,
        )
        if row is not None:
            rows.append(row[0])
            last_score_key = row[1]
        t += frame_interval_sec
        frame_idx += 1
    cap.release()
    return rows


def _try_emit_row(
    t: float, frame_idx: int, meta: MatchMeta, result, calc, ojama_pred,
    rot_1p, rot_2p, ts_1p: TimeseriesWrapper, ts_2p: TimeseriesWrapper,
    pipe, last_score_key: tuple[str, str] | None,
) -> tuple[dict, tuple[str, str]] | None:
    """両側 STABLE で盤面が変化した frame で 1 行 emit する.

    戻り値 = (row_dict, new_score_key) または None.
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
        # 時系列更新 (両サイド)
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
        return (row, key)
    except Exception:
        return None


def _compute_pair(
    result, calc, ojama_pred, rot_1p, rot_2p, pipe,
) -> tuple[IndicatorSet, IndicatorSet]:
    """1P/2P の IndicatorSet を計算 (phase_e と同等の入力で)."""
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
    inc_1p = ojama_pred.pending_for("1P")
    inc_2p = ojama_pred.pending_for("2P")
    rot_1p.update(result.p1.confirmed_board)
    rot_2p.update(result.p2.confirmed_board)
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
    return ind1, ind2


# ============================
# 1 動画処理 + shard 出力
# ============================


def process_video(
    video_id: int, max_matches: int,
    frame_interval_sec: float, history_sec: float,
) -> list[dict]:
    """1 動画分を処理し全 STABLE frame の行リストを返す."""
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return []
    metas = collect_match_meta(f"{video_id:02d}")
    if not metas:
        print(f"[skip] v{video_id:02d}: no match meta")
        return []
    if max_matches > 0:
        metas = metas[:max_matches]
    cnn_model_str = select_phase_b_model(video_id)
    cnn_model = Path(cnn_model_str) if cnn_model_str else None

    rows: list[dict] = []
    for n, meta in enumerate(metas, 1):
        match_rows = stream_match_for_h2(
            video_path, meta, cnn_model,
            frame_interval_sec=frame_interval_sec,
            history_sec=history_sec,
        )
        if not match_rows:
            print(f"[empty] v{video_id:02d}_m{meta.match_idx}: no STABLE")
            continue
        rows.extend(match_rows)
        if n % 5 == 0:
            print(
                f"  [v{video_id:02d}] {n}/{len(metas)} matches"
                f" ({len(rows)} cumulative rows)"
            )
    print(
        f"[done] v{video_id:02d}: {len(rows)} rows from {len(metas)} matches"
    )
    return rows


def _process_video_shard(args_tuple: tuple) -> tuple[int, int]:
    """1 動画分 → shard csv. multiprocessing.Pool 用."""
    (
        vid, max_matches, frame_interval_sec, history_sec, shard_dir,
    ) = args_tuple
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
        vid, max_matches, frame_interval_sec, history_sec,
    )
    if not rows:
        return (vid, 0)
    fieldnames = (
        ["video_id", "match_idx", "frame_idx", "timestamp"]
        + list(ALL_FEATURE_NAMES) + ["label"]
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    with shard_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return (vid, len(rows))


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase H2 時系列 + interaction features 収集 (280 cols)",
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
        "--shard-dir", type=Path, default=DEFAULT_SHARD_DIR,
        help="動画単位 shard csv 出力先",
    )
    args = parser.parse_args()

    if args.videos:
        target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    else:
        target_ids = list(DEFAULT_QUICK_VIDEOS)

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
    return 0


def _run_parallel(args, target_ids: list[int]) -> list[dict]:
    """並列実行 + shard 統合."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (vid, args.max_matches, args.frame_interval_sec,
         args.history_sec, args.shard_dir)
        for vid in target_ids
    ]
    print(
        f"[parallel] workers={args.workers} videos={len(target_ids)}"
        f" -> shard_dir={to_windows_path(args.shard_dir)}"
    )
    with ctx.Pool(processes=args.workers) as pool:
        for vid, n in pool.imap_unordered(_process_video_shard, tasks):
            print(f"  [shard] v{vid:02d}: {n} rows")
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
        rows = process_video(
            vid, args.max_matches, args.frame_interval_sec, args.history_sec,
        )
        all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    sys.exit(main())
