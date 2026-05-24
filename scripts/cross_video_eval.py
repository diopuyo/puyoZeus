"""
3 動画 (video_01 / video_02 / video_03) 横断クロス検証スクリプト。

目的:
    video_02 で grid search した重みが他の動画でも勝率予測に有効かを確認する。
    DEFAULT_WEIGHTS と各動画の tuned weights を 3 動画すべてで評価して比較する。

入力:
    - 動画: data/frames/video_{01,02,03}.mp4
    - 試合境界: data/verify/match_boundaries_v4/video_{01,02,03}/matches.tsv
    - 勝敗ラベル: data/verify/match_winners_v{01,02,03}.tsv

出力:
    data/verify/cross_video_eval.json
        {
          "default_weights": {video_XX: {matches, accuracy}, ...},
          "video_02_tuned_weights": {...},
          "video_01_tuned_weights": {...},
          "video_03_tuned_weights": {...},
          "summary": "..."
        }

使い方:
    ./venv/bin/python scripts/cross_video_eval.py
    ./venv/bin/python scripts/cross_video_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# プロジェクトルートを sys.path に追加 (script として直接呼ぶ場合のため)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.tune_weights import (  # noqa: E402
    GRID_SEARCH_TARGETS,
    MatchSample,
    evaluate_weights,
    extract_samples,
    grid_search,
    load_boundaries,
    load_winners,
)
from src.scorer import DEFAULT_WEIGHTS  # noqa: E402

# ============================
# 定数定義
# ============================

# クロス検証対象の動画 ID
VIDEO_IDS: tuple[str, ...] = ("video_01", "video_02", "video_03")

# サンプリングモード (試合中央) - tune_weights_v02_midpoint と整合させる
SAMPLE_MODE: str = "midpoint"

# データルート (リポジトリ相対)
FRAMES_ROOT: Path = Path("data/frames")
BOUNDARIES_ROOT: Path = Path("data/verify/match_boundaries_v4")
WINNERS_ROOT: Path = Path("data/verify")


@dataclass(frozen=True)
class VideoPaths:
    """1 動画分の入力ファイルパス。"""
    video: Path
    boundaries_tsv: Path
    winners_tsv: Path


@dataclass(frozen=True)
class VideoSamples:
    """1 動画分のサンプル束。"""
    video_id: str
    samples: list[MatchSample]


# ============================
# パス解決
# ============================


def video_paths(video_id: str, root: Path = _PROJ_ROOT) -> VideoPaths:
    """動画 ID から入力ファイルパスを組み立てる。"""
    suffix = video_id.replace("video_", "v")
    return VideoPaths(
        video=root / FRAMES_ROOT / f"{video_id}.mp4",
        boundaries_tsv=root / BOUNDARIES_ROOT / video_id / "matches.tsv",
        winners_tsv=root / WINNERS_ROOT / f"match_winners_{suffix}.tsv",
    )


# ============================
# サンプル抽出
# ============================


def extract_video_samples(paths: VideoPaths, video_id: str) -> VideoSamples:
    """1 動画から MatchSample 列を抽出する。"""
    if not paths.video.exists():
        raise FileNotFoundError(f"動画なし: {paths.video}")
    if not paths.boundaries_tsv.exists():
        raise FileNotFoundError(f"境界 TSV なし: {paths.boundaries_tsv}")
    if not paths.winners_tsv.exists():
        raise FileNotFoundError(f"勝敗 TSV なし: {paths.winners_tsv}")
    winners = load_winners(paths.winners_tsv)
    boundaries = load_boundaries(paths.boundaries_tsv)
    samples = extract_samples(
        paths.video, winners, boundaries=boundaries,
        sample_mode=SAMPLE_MODE,
    )
    return VideoSamples(video_id=video_id, samples=samples)


# ============================
# 評価ロジック
# ============================


def evaluate_on_videos(
    weights: dict[str, float],
    all_samples: dict[str, list[MatchSample]],
) -> dict[str, dict[str, float | int]]:
    """指定重みを 3 動画それぞれで評価し {video_id: {matches, accuracy}} を返す。"""
    out: dict[str, dict[str, float | int]] = {}
    for video_id, samples in all_samples.items():
        acc = evaluate_weights(samples, weights)
        out[video_id] = {"matches": len(samples), "accuracy": float(acc)}
    return out


def grid_search_per_video(
    all_samples: dict[str, list[MatchSample]],
) -> dict[str, dict[str, float]]:
    """各動画ごとに grid search した best_weights を返す。"""
    out: dict[str, dict[str, float]] = {}
    for video_id, samples in all_samples.items():
        if not samples:
            out[video_id] = dict(DEFAULT_WEIGHTS)
            continue
        best_w, _ = grid_search(samples, DEFAULT_WEIGHTS)
        out[video_id] = best_w
    return out


# ============================
# サマリ生成
# ============================


def build_summary(report: dict[str, dict[str, dict[str, float | int]]]) -> str:
    """主要結果の人間可読サマリを生成する。"""
    lines: list[str] = []
    lines.append("クロス検証結果 (accuracy):")
    weight_keys = [
        ("default_weights", "DEFAULT"),
        ("video_02_tuned_weights", "v02 tuned"),
        ("video_01_tuned_weights", "v01 tuned"),
        ("video_03_tuned_weights", "v03 tuned"),
    ]
    header = "  weights           | " + " | ".join(VIDEO_IDS)
    lines.append(header)
    for key, label in weight_keys:
        if key not in report:
            continue
        row = report[key]
        cells = []
        for vid in VIDEO_IDS:
            if vid in row:
                cells.append(f"{row[vid]['accuracy']:.3f}")
            else:
                cells.append("  -  ")
        lines.append(f"  {label:<17s} | " + " | ".join(cells))
    return "\n".join(lines)


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="3 動画クロス検証 (DEFAULT_WEIGHTS と各動画 tuned 重みを比較)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_PROJ_ROOT / "data/verify/cross_video_eval.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="動画読み込みをスキップしてパス解決と TSV 確認のみ",
    )
    args = parser.parse_args()

    # 入力パス確認
    paths_map: dict[str, VideoPaths] = {vid: video_paths(vid) for vid in VIDEO_IDS}
    for vid, p in paths_map.items():
        print(
            f"[paths] {vid}: video={p.video.exists()} "
            f"bounds={p.boundaries_tsv.exists()} "
            f"winners={p.winners_tsv.exists()}",
        )
    if args.dry_run:
        print("[dry-run] サンプル抽出をスキップして終了")
        return 0

    # サンプル抽出
    all_samples: dict[str, list[MatchSample]] = {}
    for vid in VIDEO_IDS:
        vs = extract_video_samples(paths_map[vid], vid)
        all_samples[vid] = vs.samples
        print(f"[extract] {vid}: {len(vs.samples)} 試合分のサンプル")

    # 重みごとに 3 動画評価
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    report["default_weights"] = evaluate_on_videos(DEFAULT_WEIGHTS, all_samples)
    tuned_per_video = grid_search_per_video(all_samples)
    for vid in VIDEO_IDS:
        key = f"{vid}_tuned_weights"
        report[key] = evaluate_on_videos(tuned_per_video[vid], all_samples)

    summary = build_summary(report)
    print()
    print(summary)

    # 出力
    out_payload = {
        "videos": list(VIDEO_IDS),
        "sample_mode": SAMPLE_MODE,
        "grid_targets": list(GRID_SEARCH_TARGETS),
        **report,
        "tuned_weights": {vid: tuned_per_video[vid] for vid in VIDEO_IDS},
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
