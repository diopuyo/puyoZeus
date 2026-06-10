"""
段階 4: 試合別予測 + 可視化

目的:
    3 動画 × 全試合に対し、3 戦略 (DEFAULT / LEARNED_GLOBAL /
    LEARNED_V3_GLOBAL) で勝者を予測し、実勝敗とのヒット率を試合単位で
    可視化する。試合中央 / 試合終了 5 秒前の 2 時点で評価する。

入力:
    data/training/match_features_v2.csv (10 時刻あるので mid / end5 抽出)
    src/scorer.py の DEFAULT_WEIGHTS / LEARNED_WEIGHTS_GLOBAL /
    LEARNED_WEIGHTS_V3_GLOBAL (本スクリプト実行時点で存在する場合)

出力:
    data/verify/per_match_prediction.json
    data/verify/per_match_prediction.png  (動画別 ヒット率比較棒グラフ)
    data/verify/per_match_predicted_vs_actual.png (試合×戦略 正誤マトリクス)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.old.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.old.learn_weights_v2 import predict_with_weights  # noqa: E402
from src.old.scorer import (  # noqa: E402
    DEFAULT_WEIGHTS,
    LEARNED_WEIGHTS_GLOBAL,
)

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features_v2.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/per_match_prediction.json")
DEFAULT_OUTPUT_BAR: Path = Path("data/verify/per_match_prediction.png")
DEFAULT_OUTPUT_MATRIX: Path = Path(
    "data/verify/per_match_predicted_vs_actual.png",
)

# 評価対象の time_phase
EVAL_PHASE_MID: str = "midpoint"
EVAL_PHASE_END: str = "end_minus_5"


# ============================
# 戦略
# ============================


@dataclass(frozen=True)
class Strategy:
    """重み戦略。"""
    name: str
    weights: dict[str, float]


def load_strategies() -> list[Strategy]:
    """利用可能な戦略リストを返す (V3 が import できれば追加)。"""
    strategies: list[Strategy] = [
        Strategy("DEFAULT", dict(DEFAULT_WEIGHTS)),
        Strategy("LEARNED_GLOBAL", dict(LEARNED_WEIGHTS_GLOBAL)),
    ]
    try:
        from src.old.scorer import LEARNED_WEIGHTS_V3_GLOBAL  # type: ignore[attr-defined]
        strategies.append(
            Strategy("LEARNED_V3_GLOBAL", dict(LEARNED_WEIGHTS_V3_GLOBAL)),
        )
    except (ImportError, AttributeError):
        print(
            "[info] LEARNED_WEIGHTS_V3_GLOBAL は未定義 (V3 学習未完了時はスキップ)",
            file=sys.stderr,
        )
    return strategies


# ============================
# 予測
# ============================


def predict_strategy(
    ds: Dataset, strategy: Strategy,
) -> np.ndarray:
    """指定戦略で +1/-1 予測ベクトルを返す。"""
    w = np.array([
        strategy.weights.get(n, 0.0) for n in ds.feature_names
    ])
    return predict_with_weights(ds.X, w)


def per_phase_predict(
    ds: Dataset, strategy: Strategy, phase: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """指定 phase の (pred, true, correct_mask, (video, match_idx))。"""
    mask = np.array([p == phase for p in ds.time_phases])
    sub_X = ds.X[mask]
    sub_y = ds.y[mask]
    sub_videos = [ds.video_ids[i] for i, m in enumerate(mask) if m]
    # CSV からは match_idx は別途必要 → load 時に保持してくれていない
    # ここでは行順で代用する。eda_features.Dataset は match_idx を持たない
    # ため、phase 内の連番をそのまま使う。
    sub_keys: list[tuple[str, int]] = [
        (v, i) for i, v in enumerate(sub_videos)
    ]
    w = np.array([
        strategy.weights.get(n, 0.0) for n in ds.feature_names
    ])
    pred = predict_with_weights(sub_X, w)
    correct = (pred == sub_y).astype(np.int32)
    return pred, sub_y, correct, sub_keys


# ============================
# 集計
# ============================


def hit_rate_by_video(
    ds: Dataset, strategy: Strategy, phase: str,
) -> dict[str, float]:
    """動画別ヒット率を返す。"""
    mask = np.array([p == phase for p in ds.time_phases])
    sub_X = ds.X[mask]
    sub_y = ds.y[mask]
    sub_videos = [ds.video_ids[i] for i, m in enumerate(mask) if m]
    w = np.array([
        strategy.weights.get(n, 0.0) for n in ds.feature_names
    ])
    pred = predict_with_weights(sub_X, w)
    correct = (pred == sub_y).astype(np.int32)
    out: dict[str, float] = {}
    for v in sorted(set(sub_videos)):
        v_mask = np.array([s == v for s in sub_videos])
        if v_mask.sum() == 0:
            continue
        out[v] = float(correct[v_mask].mean())
    out["overall"] = float(correct.mean()) if len(correct) else 0.0
    return out


# ============================
# 可視化
# ============================


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_hit_rate_bars(
    results: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    """戦略 × 動画 × phase のヒット率比較棒グラフ。"""
    plt = _ensure_matplotlib()
    phases = sorted(results.keys())
    if not phases:
        return
    strategies = sorted({s for p in phases for s in results[p].keys()})
    videos = sorted({
        v for p in phases for s in results[p]
        for v in results[p][s].keys() if v != "overall"
    })
    cols = ["overall"] + videos
    fig, axes = plt.subplots(
        len(phases), 1, figsize=(max(10, len(cols) * 2.0),
                                 max(4, 4 * len(phases))),
        squeeze=False,
    )
    for r, phase in enumerate(phases):
        ax = axes[r][0]
        x = np.arange(len(cols))
        width = 0.8 / max(1, len(strategies))
        for k, s in enumerate(strategies):
            vals = [
                results[phase][s].get(c, 0.0) for c in cols
            ]
            offset = (k - (len(strategies) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=s)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(cols)
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.5)
        ax.set_title(f"phase={phase}: hit rate (overall + per video)")
        ax.set_ylabel("accuracy")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_correctness_matrix(
    matrix: dict[str, dict[str, np.ndarray]],
    keys: dict[str, list[tuple[str, int]]],
    out_path: Path,
) -> None:
    """phase ごとに 試合 × 戦略 の correct/incorrect マトリクスを描画。"""
    plt = _ensure_matplotlib()
    phases = sorted(matrix.keys())
    if not phases:
        return
    fig, axes = plt.subplots(
        1, len(phases),
        figsize=(max(12, len(phases) * 8), 8), squeeze=False,
    )
    for c, phase in enumerate(phases):
        ax = axes[0][c]
        strategies = list(matrix[phase].keys())
        # 列方向に試合 (順序固定: keys[phase] の並びに従う)
        n_match = len(keys[phase])
        m = np.zeros((len(strategies), n_match), dtype=np.int8)
        for r, s in enumerate(strategies):
            m[r, :] = matrix[phase][s][:n_match]
        ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        # X 軸: 動画区切り
        videos = [k[0] for k in keys[phase]]
        boundaries = [
            i for i in range(1, len(videos)) if videos[i] != videos[i - 1]
        ]
        for bx in boundaries:
            ax.axvline(bx - 0.5, color="black", lw=1.0)
        ax.set_xticks([])
        ax.set_title(
            f"phase={phase}: green=correct, red=wrong (試合順, 動画 01→02→03)",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="試合別予測 + 可視化")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--out-bar", type=Path, default=DEFAULT_OUTPUT_BAR)
    parser.add_argument(
        "--out-matrix", type=Path, default=DEFAULT_OUTPUT_MATRIX,
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] n={len(ds.y)}, d={len(ds.feature_names)}")

    strategies = load_strategies()
    print(f"[strategies] {[s.name for s in strategies]}")
    eval_phases = [EVAL_PHASE_MID, EVAL_PHASE_END]

    results: dict[str, dict[str, dict[str, float]]] = {}
    correctness: dict[str, dict[str, np.ndarray]] = {}
    keys_by_phase: dict[str, list[tuple[str, int]]] = {}
    for phase in eval_phases:
        if not any(p == phase for p in ds.time_phases):
            print(f"[skip] phase={phase} のデータ無し")
            continue
        results[phase] = {}
        correctness[phase] = {}
        for s in strategies:
            results[phase][s.name] = hit_rate_by_video(ds, s, phase)
            _, _, correct, ks = per_phase_predict(ds, s, phase)
            correctness[phase][s.name] = correct
            keys_by_phase[phase] = ks
            print(
                f"  phase={phase} strategy={s.name} "
                f"overall={results[phase][s.name].get('overall', 0):.3f}",
            )

    plot_hit_rate_bars(results, args.out_bar)
    plot_correctness_matrix(correctness, keys_by_phase, args.out_matrix)

    out: dict[str, Any] = {
        "n_samples": len(ds.y),
        "feature_names": list(ds.feature_names),
        "strategies": [s.name for s in strategies],
        "results": results,
        "correctness_matrix": {
            phase: {
                s: correctness[phase][s].tolist()
                for s in correctness[phase]
            }
            for phase in correctness
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[save] {args.out_json}")
    print(f"[save] {args.out_bar}")
    print(f"[save] {args.out_matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
