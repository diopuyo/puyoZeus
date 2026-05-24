"""W3.2: video 別 holdout で MLP の cross-video 汎化精度を測定。

各 video N (1..19) を順に holdout にして訓練・評価。結果を tsv + bar グラフ。
学習曲線は出さない (各 video × 30 epochs で時間かかるため)。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.state_features import swap_p1_p2
from src.win_predictor import WinPredictorMLP


def evaluate(model: WinPredictorMLP, X, y) -> float:
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(np.int64)
    return float((preds == y).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/training_phase_w/win_pred_train_v2.npz",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment-swap", action="store_true", default=True)
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_results/per_video_eval.tsv",
    )
    parser.add_argument(
        "--out-png",
        default="data/verify/phase_w_results/per_video_eval.png",
    )
    args = parser.parse_args()

    print(f"loading: {args.input}")
    data = np.load(args.input)
    features = data["features"]
    labels = data["labels"].astype(np.int64)
    video_ids = data["video_ids"]
    print(f"  features: {features.shape}, videos: {sorted(np.unique(video_ids))}")

    rows: list[dict] = []
    rows.append({"video": "header"})  # placeholder
    rows = []
    for v in sorted(np.unique(video_ids)):
        mask = video_ids == v
        n_h = int(mask.sum())
        if n_h < 20:
            print(f"v{v:02d}: SKIP (only {n_h} samples)")
            continue
        X_h = features[mask]
        y_h = labels[mask]
        X_t = features[~mask]
        y_t = labels[~mask]

        # P1/P2 swap augment
        if args.augment_swap:
            X_t = np.concatenate([X_t, swap_p1_p2(X_t)])
            y_t = np.concatenate([y_t, 1 - y_t])

        # train
        model = WinPredictorMLP(seed=42)
        model.fit(
            X_t, y_t.astype(np.float32),
            epochs=args.epochs, lr=args.lr, verbose=False,
        )

        train_acc = evaluate(model, X_t, y_t)
        holdout_acc = evaluate(model, X_h, y_h)
        # baseline (majority)
        base = int(y_t.mean() >= 0.5)
        base_acc = float((y_h == base).mean())

        print(
            f"v{v:02d}: holdout n={n_h}, "
            f"train_acc={train_acc:.3f} holdout_acc={holdout_acc:.3f} "
            f"baseline={base_acc:.3f}"
        )
        rows.append({
            "video": v,
            "n_holdout": n_h,
            "train_acc": train_acc,
            "holdout_acc": holdout_acc,
            "baseline_acc": base_acc,
        })

    # tsv
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("video\tn_holdout\ttrain_acc\tholdout_acc\tbaseline_acc\n")
        for r in rows:
            f.write(
                f"{r['video']:02d}\t{r['n_holdout']}\t"
                f"{r['train_acc']:.4f}\t{r['holdout_acc']:.4f}\t"
                f"{r['baseline_acc']:.4f}\n"
            )
    print(f"saved tsv: {to_windows_path(out_tsv)}")

    # bar グラフ
    videos = [r["video"] for r in rows]
    holdout_accs = [r["holdout_acc"] for r in rows]
    baseline_accs = [r["baseline_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(videos))
    w = 0.4
    ax.bar(x - w / 2, holdout_accs, w, label="MLP holdout", color="#2080d0")
    ax.bar(x + w / 2, baseline_accs, w, label="baseline", color="#a0a0a0")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"v{v:02d}" for v in videos], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Cross-video holdout accuracy by held-out video")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = Path(args.out_png)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"saved png: {to_windows_path(out_png)}")
    print(f"\n=== Summary ===")
    print(f"mean holdout: {np.mean(holdout_accs):.4f}")
    print(f"mean baseline: {np.mean(baseline_accs):.4f}")
    print(f"MLP improvement: {np.mean(holdout_accs) - np.mean(baseline_accs):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
