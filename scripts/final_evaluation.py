"""
ぷよぷよ評価指標の最終統合評価スクリプト。

3 動画 × 138 試合 × 10 時刻フェーズ = 1390 サンプルで全戦略を比較する:

    1. DEFAULT                  (DEFAULT_WEIGHTS)
    2. LEARNED_GLOBAL           (LEARNED_WEIGHTS_GLOBAL)
    3. LEARNED_V3_GLOBAL        (LEARNED_WEIGHTS_V3_GLOBAL)
    4. PhaseAware_learned       (weight_mode="learned", interpolate=False)
    5. PhaseAware_optimal       (weight_mode="optimal",  interpolate=False)
    6. ENSEMBLE_optimal_default (PhaseAware optimal × DEFAULT 50:50)

各戦略について:
    - end_minus_5  時刻精度 (試合終了直前)
    - midpoint     時刻精度 (試合中央)
    - start_plus_30 時刻精度 (試合序盤、仕様の start_plus_20 に最も近い)
    - 動画別精度 (video_01 / video_02 / video_03)
    - 全 10 時刻フェーズ平均精度

入力:
    - data/training/match_features_v2.csv (1390 行)
    - data/verify/match_boundaries_v4/video_{01,02,03}/matches.tsv

出力:
    - data/verify/final_evaluation.json
    - data/verify/final_evaluation_report.md
    - data/verify/final_evaluation_chart.png

使い方:
    ./venv/bin/python scripts/final_evaluation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    LEARNED_WEIGHTS_V3_GLOBAL,
    PhaseAwareScorer,
    WEIGHT_MODE_LEARNED,
    WEIGHT_MODE_OPTIMAL,
)

# ============================
# 定数定義
# ============================

VIDEO_IDS: tuple[str, ...] = ("01", "02", "03")
TIME_PHASE_END: str = "end_minus_5"
TIME_PHASE_MIDPOINT: str = "midpoint"
TIME_PHASE_START: str = "start_plus_30"  # 仕様の start_plus_20 に最も近い実データ
START_OFFSET_SEC: float = 30.0
END_OFFSET_SEC: float = 5.0
ENSEMBLE_WEIGHT_OPTIMAL: float = 0.5

# time_phase ラベル → 試合内経過秒の換算で使う基準値
PHASE_OFFSET_SEC: dict[str, float] = {
    "start_plus_0": 0.0,
    "start_plus_15": 15.0,
    "start_plus_30": 30.0,
    # mid_*, midpoint, end_* は duration を使って実行時計算
}
MID_OFFSETS_FROM_MIDPOINT: dict[str, float] = {
    "mid_minus_30": -30.0,
    "mid_minus_15": -15.0,
    "mid_plus_15": 15.0,
    "mid_plus_30": 30.0,
}
END_OFFSETS_FROM_END: dict[str, float] = {
    "end_minus_15": -15.0,
    "end_minus_5": -5.0,
}

DEFAULT_FEATURES_CSV: Path = (
    _PROJ_ROOT / "data" / "training" / "match_features_v2.csv"
)
DEFAULT_BOUNDARIES_ROOT: Path = (
    _PROJ_ROOT / "data" / "verify" / "match_boundaries_v4"
)
DEFAULT_OUT_JSON: Path = (
    _PROJ_ROOT / "data" / "verify" / "final_evaluation.json"
)
DEFAULT_OUT_MD: Path = (
    _PROJ_ROOT / "data" / "verify" / "final_evaluation_report.md"
)
DEFAULT_OUT_PNG: Path = (
    _PROJ_ROOT / "data" / "verify" / "final_evaluation_chart.png"
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


# Predictor: features dict + (elapsed_sec, duration_sec) → ±1
Predictor = Callable[[dict[str, float], float, float], int]


@dataclass
class StrategyResult:
    """1 戦略の総合評価結果。

    時刻ごと/動画ごとの精度に加え、全 10 時刻平均も保持する。
    """
    strategy: str
    # time_phase → {"correct": n, "total": n, "accuracy": x}
    per_phase: dict[str, dict[str, float]] = field(default_factory=dict)
    # video_id → {...}
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
            "per_phase": dict(self.per_phase),
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
# 経過秒計算 / 単一重み予測
# ============================


def phase_to_elapsed(time_phase: str, duration: float) -> float:
    """time_phase ラベルを試合内経過秒に変換する。

    - start_plus_*  → 経過秒そのもの
    - midpoint      → duration/2
    - mid_minus_*/_plus_* → midpoint ± offset
    - end_minus_*   → duration - offset
    """
    if time_phase in PHASE_OFFSET_SEC:
        return PHASE_OFFSET_SEC[time_phase]
    midpoint = duration / 2.0 if duration > 0 else 0.0
    if time_phase == "midpoint":
        return midpoint
    if time_phase in MID_OFFSETS_FROM_MIDPOINT:
        return max(0.0, midpoint + MID_OFFSETS_FROM_MIDPOINT[time_phase])
    if time_phase in END_OFFSETS_FROM_END:
        return max(0.0, duration + END_OFFSETS_FROM_END[time_phase])
    return midpoint


def compute_diff_with_weights(
    features: dict[str, float], weights: dict[str, float],
) -> float:
    """全指標 (本指標 + 拡張指標) の重み付け差分スカラーを返す。"""
    diff = 0.0
    for name in ALL_INDICATOR_NAMES:
        diff += features.get(name, 0.0) * weights.get(name, 0.0)
    for name in EXTRA_INDICATOR_NAMES:
        diff += features.get(name, 0.0) * weights.get(name, 0.0)
    return diff


def make_static_predictor(weights: dict[str, float]) -> Predictor:
    """単一重み戦略の predictor を生成する。"""
    def predict(
        features: dict[str, float], _elapsed: float, _duration: float,
    ) -> int:
        diff = compute_diff_with_weights(features, weights)
        return 1 if diff >= 0.0 else -1
    return predict


def make_phase_aware_predictor(
    weight_mode: str, interpolate: bool = False,
) -> Predictor:
    """PhaseAwareScorer を使う predictor を生成する。"""
    scorer = PhaseAwareScorer(
        interpolate=interpolate, weight_mode=weight_mode,
    )

    def predict(
        features: dict[str, float], elapsed: float, duration: float,
    ) -> int:
        weights = scorer.resolve_weights(elapsed, duration)
        diff = compute_diff_with_weights(features, weights)
        return 1 if diff >= 0.0 else -1
    return predict


def make_ensemble_predictor(
    primary: Predictor, secondary: Predictor,
    primary_weight: float = ENSEMBLE_WEIGHT_OPTIMAL,
) -> Predictor:
    """2 つの predictor の符号付き判定を加重平均してアンサンブルする。

    各 predictor の出力 (+1/-1) を重み付けし、合計が >= 0 なら 1P 勝ち判定。
    """
    sec_weight = 1.0 - primary_weight

    def predict(
        features: dict[str, float], elapsed: float, duration: float,
    ) -> int:
        p1 = primary(features, elapsed, duration)
        p2 = secondary(features, elapsed, duration)
        combined = p1 * primary_weight + p2 * sec_weight
        return 1 if combined >= 0.0 else -1
    return predict


# ============================
# 評価ループ
# ============================


def evaluate_predictor(
    name: str,
    predictor: Predictor,
    rows: list[FeatureRow],
    durations: dict[tuple[str, int], float],
) -> StrategyResult:
    """全 1390 行を 1 戦略で評価し phase / video / overall 精度を集計する。"""
    result = StrategyResult(strategy=name)
    for row in rows:
        duration = durations.get((row.video_id, row.match_idx), 0.0)
        elapsed = phase_to_elapsed(row.time_phase, duration)
        pred = predictor(row.features, elapsed, duration)
        is_correct = pred == row.label

        # phase 集計
        ph = result.per_phase.setdefault(
            row.time_phase, {"correct": 0, "total": 0, "accuracy": 0.0},
        )
        ph["total"] += 1
        if is_correct:
            ph["correct"] += 1

        # video 集計
        vid = result.per_video.setdefault(
            row.video_id, {"correct": 0, "total": 0, "accuracy": 0.0},
        )
        vid["total"] += 1
        if is_correct:
            vid["correct"] += 1

        result.overall_total += 1
        if is_correct:
            result.overall_correct += 1

    _finalize_accuracy(result.per_phase)
    _finalize_accuracy(result.per_video)
    return result


def _finalize_accuracy(buckets: dict[str, dict[str, float]]) -> None:
    """各バケットの accuracy = correct/total を計算する。"""
    for bucket in buckets.values():
        total = bucket["total"]
        bucket["accuracy"] = (bucket["correct"] / total) if total else 0.0


def build_predictors() -> dict[str, Predictor]:
    """全戦略の predictor 辞書を構築する。"""
    pa_optimal = make_phase_aware_predictor(WEIGHT_MODE_OPTIMAL)
    default_pred = make_static_predictor(DEFAULT_WEIGHTS)
    return {
        "DEFAULT": default_pred,
        "LEARNED_GLOBAL": make_static_predictor(LEARNED_WEIGHTS_GLOBAL),
        "LEARNED_V3_GLOBAL": make_static_predictor(LEARNED_WEIGHTS_V3_GLOBAL),
        "PhaseAware_learned": make_phase_aware_predictor(WEIGHT_MODE_LEARNED),
        "PhaseAware_optimal": pa_optimal,
        "ENSEMBLE_optimal_default": make_ensemble_predictor(
            pa_optimal, default_pred,
        ),
    }


def run_all_strategies(
    rows: list[FeatureRow],
    durations: dict[tuple[str, int], float],
) -> list[StrategyResult]:
    """全 6 戦略で評価を実行する。"""
    predictors = build_predictors()
    results: list[StrategyResult] = []
    for name, predictor in predictors.items():
        results.append(evaluate_predictor(name, predictor, rows, durations))
    return results


# ============================
# レポート構築
# ============================


def build_markdown_report(results: list[StrategyResult]) -> str:
    """Markdown サマリを生成する。"""
    lines: list[str] = []
    lines.append("# ぷよぷよ評価指標 最終評価レポート")
    lines.append("")
    lines.append("データセット: 3 動画 × ~138 試合 × 10 時刻フェーズ")
    lines.append("(`data/training/match_features_v2.csv`)。")
    lines.append("")
    lines.append("## 1. 主要時刻フェーズの精度")
    lines.append("")
    lines.append(
        "| 戦略 | start_plus_30 | midpoint | end_minus_5 | 全 10 phase 平均 |",
    )
    lines.append("|---|---|---|---|---|")
    for r in results:
        cells = [r.strategy]
        for ph in (TIME_PHASE_START, TIME_PHASE_MIDPOINT, TIME_PHASE_END):
            bucket = r.per_phase.get(ph, {})
            cells.append(_format_acc(bucket))
        cells.append(
            f"{r.overall_accuracy:.3f} "
            f"({r.overall_correct}/{r.overall_total})",
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 2. 動画別精度 (10 phase 全体)")
    lines.append("")
    lines.append("| 戦略 | video_01 | video_02 | video_03 |")
    lines.append("|---|---|---|---|")
    for r in results:
        cells = [r.strategy]
        for vid in VIDEO_IDS:
            bucket = r.per_video.get(vid, {})
            cells.append(_format_acc(bucket))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 3. 全 10 時刻フェーズ別精度")
    lines.append("")
    phases = sorted({ph for r in results for ph in r.per_phase.keys()})
    header = "| 戦略 | " + " | ".join(phases) + " |"
    sep = "|---|" + "---|" * len(phases)
    lines.append(header)
    lines.append(sep)
    for r in results:
        cells = [r.strategy]
        for ph in phases:
            cells.append(f"{r.per_phase.get(ph, {}).get('accuracy', 0.0):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    best = max(results, key=lambda x: x.overall_accuracy)
    lines.append(
        f"**最良戦略 (10 phase 平均): {best.strategy} "
        f"(overall acc={best.overall_accuracy:.3f})**",
    )
    return "\n".join(lines)


def _format_acc(bucket: dict[str, float]) -> str:
    """(acc, n=N) 形式に整形する。"""
    acc = bucket.get("accuracy", 0.0)
    n = int(bucket.get("total", 0))
    return f"{acc:.3f} (n={n})"


def save_results(
    results: list[StrategyResult],
    out_json: Path,
    out_md: Path,
) -> None:
    """JSON と Markdown を保存する。"""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "videos": list(VIDEO_IDS),
        "main_phases": [TIME_PHASE_START, TIME_PHASE_MIDPOINT, TIME_PHASE_END],
        "results": [r.to_dict() for r in results],
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = build_markdown_report(results)
    out_md.write_text(md, encoding="utf-8")


def save_chart(results: list[StrategyResult], out_png: Path) -> None:
    """主要 3 時刻 + overall を比較する棒グラフ PNG を保存する。

    matplotlib が無い環境では何も書かずに終わる (後段でログのみ)。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[chart] matplotlib 利用不可 — PNG 出力をスキップ: {exc}")
        return

    metric_keys = (
        TIME_PHASE_START,
        TIME_PHASE_MIDPOINT,
        TIME_PHASE_END,
        "overall",
    )
    metric_labels = ("start_plus_30", "midpoint", "end_minus_5", "overall")
    strategies = [r.strategy for r in results]
    values = np.zeros((len(metric_keys), len(strategies)))
    for j, r in enumerate(results):
        for i, key in enumerate(metric_keys):
            if key == "overall":
                values[i, j] = r.overall_accuracy
            else:
                values[i, j] = r.per_phase.get(key, {}).get("accuracy", 0.0)

    x = np.arange(len(strategies))
    bar_w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, label in enumerate(metric_labels):
        ax.bar(x + (i - 1.5) * bar_w, values[i], bar_w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Final evaluation: phase accuracy by strategy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="ぷよぷよ評価指標 最終評価")
    parser.add_argument("--csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument(
        "--boundaries-root", type=Path, default=DEFAULT_BOUNDARIES_ROOT,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
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
    save_chart(results, args.out_png)
    print()
    print(f"[save] {args.out_json}")
    print(f"[save] {args.out_md}")
    print(f"[save] {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
