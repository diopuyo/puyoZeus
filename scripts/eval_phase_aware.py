"""
Phase-aware Scorer の精度評価スクリプト。

3 動画 (video_01 / video_02 / video_03) × 全試合で、4 種類の重み戦略を
試合終了直前 (end_minus_5) の特徴量に対して比較する:

    1. DEFAULT_WEIGHTS                            (既存ベースライン)
    2. LEARNED_WEIGHTS_GLOBAL                     (時刻フェーズ統合 LR)
    3. PhaseAwareScorer(interpolate=False)        (離散切替)
    4. PhaseAwareScorer(interpolate=True)         (線形補間)

入力:
    - data/training/match_features.csv (715 行 × 18 列)
    - data/verify/match_boundaries_v4/video_{01,02,03}/matches.tsv

出力:
    - data/verify/phase_aware_eval.json
    - data/verify/phase_aware_eval.md

使い方:
    ./venv/bin/python scripts/eval_phase_aware.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.indicators import (  # noqa: E402
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
)
from src.scorer import (  # noqa: E402
    DEFAULT_WEIGHTS,
    LEARNED_WEIGHTS_GLOBAL,
    PhaseAwareScorer,
    classify_phase,
)

# ============================
# 定数定義
# ============================

VIDEO_IDS: tuple[str, ...] = ("01", "02", "03")
TIME_PHASE_END: str = "end_minus_5"
TIME_PHASE_MIDPOINT: str = "midpoint"
TIME_PHASE_START: str = "start_plus_20"
END_OFFSET_SEC: float = 5.0
START_OFFSET_SEC: float = 20.0

DEFAULT_FEATURES_CSV: Path = (
    _PROJ_ROOT / "data" / "training" / "match_features.csv"
)
DEFAULT_BOUNDARIES_ROOT: Path = (
    _PROJ_ROOT / "data" / "verify" / "match_boundaries_v4"
)
DEFAULT_OUT_JSON: Path = (
    _PROJ_ROOT / "data" / "verify" / "phase_aware_eval.json"
)
DEFAULT_OUT_MD: Path = (
    _PROJ_ROOT / "data" / "verify" / "phase_aware_eval.md"
)


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class FeatureRow:
    """1 試合 × 1 フェーズの特徴量差分 (1P - 2P) と勝敗ラベル。"""
    video_id: str
    match_idx: int
    time_phase: str
    features: dict[str, float]
    label: int  # +1=1P win / -1=2P win


@dataclass
class EvalResult:
    """1 戦略 × 動画の評価結果。"""
    strategy: str
    per_video: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_correct: int = 0
    overall_total: int = 0

    @property
    def overall_accuracy(self) -> float:
        return (self.overall_correct / self.overall_total
                if self.overall_total else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "per_video": dict(self.per_video),
            "overall_accuracy": self.overall_accuracy,
            "overall_correct": self.overall_correct,
            "overall_total": self.overall_total,
        }


# ============================
# 入出力
# ============================


def load_feature_rows(csv_path: Path) -> list[FeatureRow]:
    """match_features.csv を読み込み FeatureRow 列に変換する。"""
    rows: list[FeatureRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_keys = [k for k in reader.fieldnames or []
                        if k not in {"video_id", "match_idx",
                                     "time_phase", "label"}]
        for r in reader:
            features = {k: float(r[k]) for k in feature_keys}
            rows.append(FeatureRow(
                video_id=str(r["video_id"]),
                match_idx=int(r["match_idx"]),
                time_phase=str(r["time_phase"]),
                features=features,
                label=int(r["label"]),
            ))
    return rows


def load_match_durations(
    boundaries_root: Path,
) -> dict[tuple[str, int], float]:
    """video_id+match_idx → 試合長 (秒) のマップを TSV から作る。"""
    out: dict[tuple[str, int], float] = {}
    for vid in VIDEO_IDS:
        tsv = boundaries_root / f"video_{vid}" / "matches.tsv"
        if not tsv.exists():
            continue
        for line in tsv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("idx"):
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            try:
                idx = int(float(cols[0]))
                start = float(cols[1])
                end = float(cols[2])
            except ValueError:
                continue
            out[(vid, idx)] = end - start
    return out


# ============================
# スコアリング
# ============================


def predict_with_weights(
    features: dict[str, float],
    weights: dict[str, float],
) -> int:
    """重み・特徴量差分から勝敗を予測する (+1=1P / -1=2P)。"""
    diff = 0.0
    for name in ALL_INDICATOR_NAMES:
        diff += features.get(name, 0.0) * weights.get(name, 0.0)
    for name in EXTRA_INDICATOR_NAMES:
        diff += features.get(name, 0.0) * weights.get(name, 0.0)
    return 1 if diff >= 0.0 else -1


def predict_with_phase_aware(
    features: dict[str, float],
    elapsed_sec: float,
    duration_sec: float,
    interpolate: bool,
) -> int:
    """PhaseAwareScorer の重みで勝敗を予測する。"""
    scorer = PhaseAwareScorer(interpolate=interpolate)
    weights = scorer.resolve_weights(elapsed_sec, duration_sec)
    return predict_with_weights(features, weights)


def phase_to_elapsed(time_phase: str, duration: float) -> float:
    """time_phase を試合内経過秒に変換する。

    duration <=0 の場合は概算で返し、PhaseAwareScorer 側で mid 扱いにする。
    """
    if time_phase == TIME_PHASE_END:
        return max(0.0, duration - END_OFFSET_SEC)
    if time_phase == TIME_PHASE_START:
        return START_OFFSET_SEC
    if time_phase == TIME_PHASE_MIDPOINT:
        return duration / 2.0 if duration > 0 else 0.0
    # 中盤前後 (mid_minus_20 / mid_plus_20)
    midpoint = duration / 2.0 if duration > 0 else 0.0
    if time_phase == "mid_minus_20":
        return max(0.0, midpoint - 20.0)
    if time_phase == "mid_plus_20":
        return midpoint + 20.0
    return midpoint


# ============================
# 評価
# ============================


def evaluate_strategy(
    strategy: str,
    rows: list[FeatureRow],
    durations: dict[tuple[str, int], float],
    weights: dict[str, float] | None = None,
    phase_aware: bool = False,
    interpolate: bool = False,
) -> EvalResult:
    """1 戦略を全行で評価して per_video / overall accuracy を返す。

    end_minus_5 行のみを対象とする (試合終了時の予測精度を計測)。
    """
    result = EvalResult(strategy=strategy)
    for video_id in VIDEO_IDS:
        result.per_video[video_id] = {"correct": 0, "total": 0,
                                       "accuracy": 0.0}
    for row in rows:
        if row.time_phase != TIME_PHASE_END:
            continue
        if row.video_id not in result.per_video:
            continue
        duration = durations.get((row.video_id, row.match_idx), 0.0)
        elapsed = phase_to_elapsed(row.time_phase, duration)
        if phase_aware:
            pred = predict_with_phase_aware(
                row.features, elapsed, duration, interpolate,
            )
        else:
            assert weights is not None
            pred = predict_with_weights(row.features, weights)
        is_correct = pred == row.label
        bucket = result.per_video[row.video_id]
        bucket["total"] += 1
        if is_correct:
            bucket["correct"] += 1
        result.overall_total += 1
        if is_correct:
            result.overall_correct += 1
    for video_id, bucket in result.per_video.items():
        total = bucket["total"]
        bucket["accuracy"] = (bucket["correct"] / total) if total else 0.0
    return result


def run_all_strategies(
    rows: list[FeatureRow],
    durations: dict[tuple[str, int], float],
) -> list[EvalResult]:
    """4 戦略すべてを評価して結果列を返す。"""
    out: list[EvalResult] = []
    out.append(evaluate_strategy(
        "DEFAULT", rows, durations, weights=DEFAULT_WEIGHTS,
    ))
    out.append(evaluate_strategy(
        "LEARNED_GLOBAL", rows, durations, weights=LEARNED_WEIGHTS_GLOBAL,
    ))
    out.append(evaluate_strategy(
        "PhaseAware_discrete", rows, durations,
        phase_aware=True, interpolate=False,
    ))
    out.append(evaluate_strategy(
        "PhaseAware_interpolated", rows, durations,
        phase_aware=True, interpolate=True,
    ))
    return out


# ============================
# レポート
# ============================


def build_markdown_report(results: list[EvalResult]) -> str:
    """Markdown サマリを生成する。"""
    lines: list[str] = []
    lines.append("# Phase-aware Scorer 評価レポート")
    lines.append("")
    lines.append("試合終了 5 秒前 (end_minus_5) の特徴量差分から勝敗を"
                 "予測した際の動画別 / 全体 accuracy。")
    lines.append("")
    header = (
        "| 戦略 | video_01 | video_02 | video_03 | 全体 (acc) |"
    )
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        cells = [r.strategy]
        for vid in VIDEO_IDS:
            bucket = r.per_video.get(vid, {})
            acc = bucket.get("accuracy", 0.0)
            n = bucket.get("total", 0)
            cells.append(f"{acc:.3f} (n={n})")
        cells.append(
            f"{r.overall_accuracy:.3f} "
            f"({r.overall_correct}/{r.overall_total})",
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    best = max(results, key=lambda x: x.overall_accuracy)
    lines.append(f"**最良戦略: {best.strategy} "
                 f"(overall acc={best.overall_accuracy:.3f})**")
    return "\n".join(lines)


def save_results(
    results: list[EvalResult],
    out_json: Path,
    out_md: Path,
) -> None:
    """JSON と Markdown を保存する。"""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "videos": list(VIDEO_IDS),
        "time_phase": TIME_PHASE_END,
        "results": [r.to_dict() for r in results],
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = build_markdown_report(results)
    out_md.write_text(md, encoding="utf-8")


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-aware Scorer 精度評価",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument(
        "--boundaries-root", type=Path, default=DEFAULT_BOUNDARIES_ROOT,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    rows = load_feature_rows(args.csv)
    print(f"[load] features: {len(rows)} 行")
    durations = load_match_durations(args.boundaries_root)
    print(f"[load] durations: {len(durations)} 試合")

    results = run_all_strategies(rows, durations)
    md = build_markdown_report(results)
    print()
    print(md)

    save_results(results, args.out_json, args.out_md)
    print()
    print(f"[save] {args.out_json}")
    print(f"[save] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
