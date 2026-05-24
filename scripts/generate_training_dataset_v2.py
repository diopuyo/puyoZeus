"""
段階 2: 大規模データセット拡張 (10 時刻フェーズ)

目的:
    既存 data/training/match_features.csv (5 時刻 / 715 行) に対し、
    10 時刻にサンプリング点を増やすことで ~1380 行の学習データを構築する。

サンプリング時刻 (1 試合あたり 10 点):
    - start_plus_0   : 試合開始直後 (start + 1s)
    - start_plus_15  : 試合開始 15 秒後
    - start_plus_30  : 試合開始 30 秒後
    - mid_minus_30   : midpoint - 30s
    - mid_minus_15   : midpoint - 15s
    - midpoint       : (start + end) / 2
    - mid_plus_15    : midpoint + 15s
    - mid_plus_30    : midpoint + 30s
    - end_minus_15   : 試合終了 15 秒前
    - end_minus_5    : 試合終了 5 秒前

副作用なし: 既存 v1 csv は残す。出力は match_features_v2.csv。

実行例:
    python -m scripts.generate_training_dataset_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.generate_training_dataset import (  # noqa: E402
    DEFAULT_BOUNDARY_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_WINNERS_DIR,
    FEATURE_NAMES,
    MIN_MATCH_DURATION_SEC,
    VIDEO_IDS,
    FeatureRow,
    MatchMeta,
    collect_match_meta,
    extract_video_rows,
    write_csv,
)

# ============================
# 定数 (v2 専用 10 時刻フェーズ)
# ============================

DEFAULT_OUTPUT_CSV: Path = Path("data/training/match_features_v2.csv")

# v2: 10 時刻フェーズ識別子
TIME_PHASE_START_0: str = "start_plus_0"
TIME_PHASE_START_15: str = "start_plus_15"
TIME_PHASE_START_30: str = "start_plus_30"
TIME_PHASE_MID_M30: str = "mid_minus_30"
TIME_PHASE_MID_M15: str = "mid_minus_15"
TIME_PHASE_MIDPOINT: str = "midpoint"
TIME_PHASE_MID_P15: str = "mid_plus_15"
TIME_PHASE_MID_P30: str = "mid_plus_30"
TIME_PHASE_END_M15: str = "end_minus_15"
TIME_PHASE_END_M5: str = "end_minus_5"

DEFAULT_TIME_PHASES_V2: tuple[str, ...] = (
    TIME_PHASE_START_0,
    TIME_PHASE_START_15,
    TIME_PHASE_START_30,
    TIME_PHASE_MID_M30,
    TIME_PHASE_MID_M15,
    TIME_PHASE_MIDPOINT,
    TIME_PHASE_MID_P15,
    TIME_PHASE_MID_P30,
    TIME_PHASE_END_M15,
    TIME_PHASE_END_M5,
)

# 各 phase の (基準点, オフセット秒) 定義。
# 基準点: "start" / "mid" / "end" のいずれか。
PHASE_DEFINITIONS: dict[str, tuple[str, float]] = {
    TIME_PHASE_START_0: ("start", 1.0),
    TIME_PHASE_START_15: ("start", 15.0),
    TIME_PHASE_START_30: ("start", 30.0),
    TIME_PHASE_MID_M30: ("mid", -30.0),
    TIME_PHASE_MID_M15: ("mid", -15.0),
    TIME_PHASE_MIDPOINT: ("mid", 0.0),
    TIME_PHASE_MID_P15: ("mid", 15.0),
    TIME_PHASE_MID_P30: ("mid", 30.0),
    TIME_PHASE_END_M15: ("end", -15.0),
    TIME_PHASE_END_M5: ("end", -5.0),
}


def compute_sample_time_v2(meta: MatchMeta, phase: str) -> float:
    """v2 の 10 時刻フェーズに対応する試合内秒数 (動画上の絶対秒) を返す。"""
    if phase not in PHASE_DEFINITIONS:
        raise ValueError(f"未知の time_phase (v2): {phase}")
    anchor, offset = PHASE_DEFINITIONS[phase]
    midpoint = (meta.start_sec + meta.end_sec) / 2.0
    if anchor == "start":
        t = meta.start_sec + offset
    elif anchor == "mid":
        t = midpoint + offset
    elif anchor == "end":
        t = meta.end_sec + offset
    else:  # pragma: no cover (定義により到達しない)
        raise ValueError(f"未知の anchor: {anchor}")
    # 試合範囲内に丸める
    t = max(meta.start_sec + 0.5, min(meta.end_sec - 0.5, t))
    return t


# ============================
# v1 互換のパッチ
# ============================
# extract_video_rows は内部で `compute_sample_time` (v1) を呼ぶため、
# モンキーパッチで v2 仕様に切り替える。


def _patch_compute_sample_time() -> None:
    """generate_training_dataset.compute_sample_time を v2 版に差し替える。"""
    import scripts.generate_training_dataset as _gen
    _gen.compute_sample_time = compute_sample_time_v2


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="多動画 × 10 時刻 大規模特徴量データセット生成 (v2)",
    )
    parser.add_argument(
        "--videos", nargs="+", default=list(VIDEO_IDS),
    )
    parser.add_argument(
        "--time-phases", nargs="+", default=list(DEFAULT_TIME_PHASES_V2),
    )
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--max-matches-per-video", type=int, default=None,
    )
    args = parser.parse_args()

    _patch_compute_sample_time()
    all_rows: list[FeatureRow] = []
    for video_id in args.videos:
        video_path = args.video_dir / f"video_{video_id}.mp4"
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}", file=sys.stderr)
            continue
        metas = collect_match_meta(video_id)
        if args.max_matches_per_video is not None:
            metas = metas[: args.max_matches_per_video]
        # 試合長 30 秒未満は start+30s が end を超えるためスキップ
        metas = [
            m for m in metas
            if (m.end_sec - m.start_sec) >= max(MIN_MATCH_DURATION_SEC, 30.0)
        ]
        print(
            f"[video {video_id}] {len(metas)} 試合 × "
            f"{len(args.time_phases)} 時刻 = "
            f"{len(metas) * len(args.time_phases)} サンプル予定",
            flush=True,
        )
        rows = extract_video_rows(
            video_id, video_path, metas, args.time_phases,
        )
        all_rows.extend(rows)
        print(f"[video {video_id}] 抽出完了: {len(rows)} 行", flush=True)

    write_csv(all_rows, args.out)
    print(
        f"\n[save] {args.out}: {len(all_rows)} 行 "
        f"(列数 {3 + len(FEATURE_NAMES) + 1})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
