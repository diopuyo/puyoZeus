"""
Phase-aware Scorer + DEFAULT_WEIGHTS のアンサンブル評価スクリプト。

PhaseAwareScorer (interpolate=True) と DEFAULT_WEIGHTS (Scorer) の生スコア
(p1 - p2 重み付け差分) を、混合比 α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} で
線形結合して試合終了予測精度を比較する。

    score_ensemble = (1 - α) * score_default + α * score_phase_aware

α=0.0 は DEFAULT のみ、α=1.0 は PhaseAware のみ。

入力:
    - data/training/match_features.csv (715 行)
    - data/verify/match_boundaries_v4/video_{01,02,03}/matches.tsv

出力:
    - data/verify/ensemble_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eval_phase_aware import (  # noqa: E402
    DEFAULT_BOUNDARIES_ROOT,
    DEFAULT_FEATURES_CSV,
    TIME_PHASE_END,
    VIDEO_IDS,
    FeatureRow,
    load_feature_rows,
    load_match_durations,
    phase_to_elapsed,
)
from src.indicators import (  # noqa: E402
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
)
from src.scorer import (  # noqa: E402
    DEFAULT_WEIGHTS,
    PhaseAwareScorer,
)

# ============================
# 定数定義
# ============================

# アンサンブル混合比のグリッド
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

DEFAULT_OUT_JSON: Path = (
    _PROJ_ROOT / "data" / "verify" / "ensemble_eval.json"
)


# ============================
# 補助関数
# ============================


def weighted_diff(
    features: dict[str, float],
    weights: dict[str, float],
) -> float:
    """指標差分 × 重みの和 (= p1_raw - p2_raw)。"""
    diff = 0.0
    for name in ALL_INDICATOR_NAMES:
        diff += features.get(name, 0.0) * weights.get(name, 0.0)
    for name in EXTRA_INDICATOR_NAMES:
        diff += features.get(name, 0.0) * weights.get(name, 0.0)
    return diff


def normalize_score(diff: float, weights: dict[str, float]) -> float:
    """重み L1 ノルムで割って [-1, 1] に正規化する。"""
    norm = sum(abs(w) for w in weights.values())
    if norm == 0.0:
        return 0.0
    return diff / norm


# ============================
# 評価
# ============================


@dataclass
class EnsembleResult:
    """1 α 値での評価結果。"""
    alpha: float
    per_video: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_correct: int = 0
    overall_total: int = 0

    @property
    def overall_accuracy(self) -> float:
        return (self.overall_correct / self.overall_total
                if self.overall_total else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "per_video": dict(self.per_video),
            "overall_accuracy": self.overall_accuracy,
            "overall_correct": self.overall_correct,
            "overall_total": self.overall_total,
        }


def evaluate_alpha(
    alpha: float,
    rows: list[FeatureRow],
    durations: dict[tuple[str, int], float],
    pa_scorer: PhaseAwareScorer,
) -> EnsembleResult:
    """指定 α でアンサンブル評価する。"""
    result = EnsembleResult(alpha=alpha)
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
        # 各 scorer での正規化済みスコア
        score_def = normalize_score(
            weighted_diff(row.features, DEFAULT_WEIGHTS), DEFAULT_WEIGHTS,
        )
        pa_weights = pa_scorer.resolve_weights(elapsed, duration)
        score_pa = normalize_score(
            weighted_diff(row.features, pa_weights), pa_weights,
        )
        score_mix = (1.0 - alpha) * score_def + alpha * score_pa
        pred = 1 if score_mix >= 0.0 else -1
        bucket = result.per_video[row.video_id]
        bucket["total"] += 1
        if pred == row.label:
            bucket["correct"] += 1
            result.overall_correct += 1
        result.overall_total += 1
    for bucket in result.per_video.values():
        total = bucket["total"]
        bucket["accuracy"] = (bucket["correct"] / total) if total else 0.0
    return result


def grid_search_alpha(
    rows: list[FeatureRow],
    durations: dict[tuple[str, int], float],
    grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[list[EnsembleResult], EnsembleResult]:
    """α grid をすべて評価し、結果列と最良結果を返す。"""
    pa_scorer = PhaseAwareScorer(interpolate=True)
    results: list[EnsembleResult] = []
    for alpha in grid:
        results.append(evaluate_alpha(alpha, rows, durations, pa_scorer))
    best = max(results, key=lambda r: r.overall_accuracy)
    return results, best


# ============================
# レポート
# ============================


def build_summary(results: list[EnsembleResult],
                   best: EnsembleResult) -> str:
    """サマリを文字列として返す。"""
    lines: list[str] = []
    lines.append("アンサンブル評価 (DEFAULT × PhaseAware_interpolated)")
    header = "  α   | " + " | ".join(f"video_{v}" for v in VIDEO_IDS)
    header += " | overall"
    lines.append(header)
    for r in results:
        cells = [f"{r.alpha:.2f}"]
        for vid in VIDEO_IDS:
            acc = r.per_video.get(vid, {}).get("accuracy", 0.0)
            cells.append(f"{acc:.3f}")
        cells.append(f"{r.overall_accuracy:.3f}")
        lines.append("  " + " | ".join(cells))
    lines.append("")
    lines.append(
        f"最良 α = {best.alpha:.2f} → overall acc {best.overall_accuracy:.3f}"
        f" ({best.overall_correct}/{best.overall_total})",
    )
    return "\n".join(lines)


def save_results(
    results: list[EnsembleResult],
    best: EnsembleResult,
    out_json: Path,
) -> None:
    """JSON 保存。"""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "alpha_grid": list(ALPHA_GRID),
        "results": [r.to_dict() for r in results],
        "best_alpha": best.alpha,
        "best_accuracy": best.overall_accuracy,
        "summary": build_summary(results, best),
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="アンサンブル (DEFAULT × PhaseAware) 精度評価",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument(
        "--boundaries-root", type=Path, default=DEFAULT_BOUNDARIES_ROOT,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    rows = load_feature_rows(args.csv)
    print(f"[load] features: {len(rows)} 行")
    durations = load_match_durations(args.boundaries_root)
    print(f"[load] durations: {len(durations)} 試合")

    results, best = grid_search_alpha(rows, durations)
    summary = build_summary(results, best)
    print()
    print(summary)

    save_results(results, best, args.out_json)
    print()
    print(f"[save] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
