"""W2.3: 勝率予測モデルの訓練 + ベースライン比較。

W2.2 が出力した npz をロードし、WinPredictorMLP を訓練。
holdout 評価で accuracy, log-loss, AUC を計測。

Holdout 戦略:
    - --holdout-video N で 1 動画を holdout (汎化性能評価)
    - --holdout-ratio R で全データから R 割を holdout (in-distribution)

ベースライン (今回は MLP のみ訓練、Scorer ベースは別スクリプト予定):
    - 全 50% (常に 1P 勝つと予測する dummy)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_train_predictor \
        --input data/training_phase_w/win_pred_train.npz \
        --holdout-video 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.win_predictor import WinPredictorMLP


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    video_ids: np.ndarray,
    holdout_video: int = 0,
    holdout_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """train / holdout に分割。

    Args:
        holdout_video: > 0 なら指定動画を holdout (1=video_01, 2=v02, 3=v03)
        holdout_ratio: > 0 なら全データから割合で holdout
    Returns:
        ((X_train, y_train), (X_holdout, y_holdout))
    """
    if holdout_video > 0:
        mask = video_ids == holdout_video
        X_h = features[mask]
        y_h = labels[mask]
        X_t = features[~mask]
        y_t = labels[~mask]
    elif holdout_ratio > 0:
        rng = np.random.default_rng(seed)
        n = features.shape[0]
        idx = rng.permutation(n)
        n_h = int(n * holdout_ratio)
        X_h = features[idx[:n_h]]
        y_h = labels[idx[:n_h]]
        X_t = features[idx[n_h:]]
        y_t = labels[idx[n_h:]]
    else:
        X_t = features
        y_t = labels
        X_h = features[:0]
        y_h = labels[:0]
    return (X_t, y_t), (X_h, y_h)


def evaluate(
    model: WinPredictorMLP,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """accuracy, log-loss, mean prob, 混同行列要素を計測。"""
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(np.int64)
    acc = float((preds == y).mean())
    eps = 1e-7
    log_loss = float(-np.mean(
        y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps),
    ))
    # 混同行列要素 (1=1P, 0=2P)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    return {
        "accuracy": acc,
        "log_loss": log_loss,
        "mean_prob": float(probs.mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def plot_learning_curve(
    losses: list[float], out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", color="#2080d0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(
    metrics: dict, title: str, out_path: Path,
) -> None:
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", color="black", fontsize=14,
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 2P", "pred 1P"])
    ax.set_yticklabels(["actual 2P", "actual 1P"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def baseline_majority(y_train: np.ndarray, y_holdout: np.ndarray) -> dict:
    """訓練データ多数決で全 holdout を予測した場合の精度。"""
    pred = int(y_train.mean() >= 0.5)
    acc = float((y_holdout == pred).mean())
    return {"accuracy": acc, "always_pred": pred}


def baseline_score_diff(
    X: np.ndarray, y: np.ndarray, scale: float = 5.0,
) -> dict:
    """encoded features の最後 4 dim (score_p1, score_p2, ojama_p1, ojama_p2)
    から「P1 が score 多い + ojama 少ない」なら 1P 有利と単純予測。

    state_features.encode_state の order に依存:
        [..., encode_score(score_p1), encode_score(score_p2),
              encode_ojama(ojama_p1), encode_ojama(ojama_p2)]
    """
    score_p1 = X[:, -4]
    score_p2 = X[:, -3]
    ojama_p1 = X[:, -2]
    ojama_p2 = X[:, -1]
    diff = (score_p1 - score_p2) - (ojama_p1 - ojama_p2)
    probs = 1.0 / (1.0 + np.exp(-diff * scale))
    preds = (probs >= 0.5).astype(np.int64)
    acc = float((preds == y).mean())
    eps = 1e-7
    log_loss = float(-np.mean(
        y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps),
    ))
    return {
        "accuracy": acc,
        "log_loss": log_loss,
        "mean_prob": float(probs.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/training_phase_w/win_pred_train.npz",
    )
    parser.add_argument(
        "--out-model", default="models/win_predictor_v1.pt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--holdout-video", type=int, default=0,
        help="1=v01, 2=v02, 3=v03 を holdout。0 なら ratio を使う",
    )
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment-swap", action="store_true",
        help="P1/P2 swap でデータ拡張 (訓練のみ、holdout は元のまま)",
    )
    args = parser.parse_args()

    print(f"loading: {args.input}")
    data = np.load(args.input)
    features = data["features"]
    labels = data["labels"].astype(np.int64)
    video_ids = data["video_ids"]
    print(f"  features: {features.shape}")
    print(f"  labels: {labels.shape}, 1P={int(labels.sum())}/{len(labels)}")
    print(f"  videos: {np.unique(video_ids)}")

    (X_t, y_t), (X_h, y_h) = split_data(
        features, labels, video_ids,
        holdout_video=args.holdout_video,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    print(f"\nsplit: train={X_t.shape[0]}, holdout={X_h.shape[0]}")

    if args.augment_swap:
        from src.state_features import swap_p1_p2
        X_t_swap = swap_p1_p2(X_t)
        y_t_swap = 1 - y_t  # label 反転
        X_t = np.concatenate([X_t, X_t_swap])
        y_t = np.concatenate([y_t, y_t_swap])
        print(f"after P1/P2 swap augment: train={X_t.shape[0]}")

    if X_h.shape[0] == 0:
        print("[WARN] holdout 0、train で評価します")
        X_h, y_h = X_t, y_t

    # ベースライン
    base = baseline_majority(y_t, y_h)
    print(f"\nbaseline (majority class): "
          f"accuracy={base['accuracy']:.4f} (always pred {base['always_pred']})")
    base_sd = baseline_score_diff(X_h, y_h)
    print(
        f"baseline (score_diff): "
        f"accuracy={base_sd['accuracy']:.4f} "
        f"loss={base_sd['log_loss']:.4f}"
    )

    # モデル訓練
    print(f"\ntraining MLP: epochs={args.epochs} lr={args.lr}")
    model = WinPredictorMLP(seed=args.seed)
    losses = model.fit(
        X_t, y_t.astype(np.float32),
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, verbose=True,
    )

    # 評価
    train_metrics = evaluate(model, X_t, y_t)
    holdout_metrics = evaluate(model, X_h, y_h)
    print(
        f"\ntrain   : acc={train_metrics['accuracy']:.4f} "
        f"loss={train_metrics['log_loss']:.4f} "
        f"mean_p={train_metrics['mean_prob']:.3f}"
    )
    print(
        f"holdout : acc={holdout_metrics['accuracy']:.4f} "
        f"loss={holdout_metrics['log_loss']:.4f} "
        f"mean_p={holdout_metrics['mean_prob']:.3f}"
    )
    print(
        f"baseline: acc={base['accuracy']:.4f} "
        f"(MLP - baseline = {holdout_metrics['accuracy'] - base['accuracy']:+.4f})"
    )

    # 保存
    out_path = Path(args.out_model)
    model.save(out_path)
    print(f"\nsaved: {to_windows_path(out_path)}")

    # 成果物 (グラフ等)
    out_stem = out_path.stem
    out_dir = Path("data/verify/phase_w_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 学習曲線
    lc_path = out_dir / f"{out_stem}_learning_curve.png"
    plot_learning_curve(losses, lc_path)
    print(f"learning curve: {to_windows_path(lc_path)}")
    # 混同行列 (holdout)
    cm_path = out_dir / f"{out_stem}_confusion.png"
    plot_confusion(
        holdout_metrics, f"holdout ({X_h.shape[0]} samples)", cm_path,
    )
    print(f"confusion matrix: {to_windows_path(cm_path)}")

    # 結果サマリ TSV
    tsv_path = out_dir / f"{out_stem}_summary.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("metric\ttrain\tholdout\n")
        f.write(f"accuracy\t{train_metrics['accuracy']:.4f}\t{holdout_metrics['accuracy']:.4f}\n")
        f.write(f"log_loss\t{train_metrics['log_loss']:.4f}\t{holdout_metrics['log_loss']:.4f}\n")
        f.write(f"mean_prob\t{train_metrics['mean_prob']:.4f}\t{holdout_metrics['mean_prob']:.4f}\n")
        f.write(f"baseline_majority\t-\t{base['accuracy']:.4f}\n")
        f.write(f"baseline_score_diff\t-\t{base_sd['accuracy']:.4f}\n")
        f.write(f"holdout_size\t-\t{X_h.shape[0]}\n")
        f.write(f"train_size\t{X_t.shape[0]}\t-\n")
    print(f"summary: {to_windows_path(tsv_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
