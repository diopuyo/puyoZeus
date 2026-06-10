"""
Train/Test split による重み学習の汎化精度評価 (過適合検出)

目的:
    `tune_weights.py` の grid search は 50 試合全体に最適化するため、
    特定セットへの 過適合 (overfitting) が懸念される。
    本 script は train/test split + K-fold で 真の汎化精度 を測定する。

実行例:
    ./venv/bin/python scripts/eval_weights_holdout.py \
        --video data/frames/video_02.mp4 \
        --boundaries data/verify/match_boundaries_v4/video_02/matches.tsv \
        --winners data/verify/match_winners_v02.tsv \
        --train-ratio 0.5 \
        --time-mode midpoint \
        --out data/verify/holdout_eval.json

    # tune_weights.py の midpoint cache を再利用 (動画読込スキップ)
    ./venv/bin/python scripts/eval_weights_holdout.py \
        --features-cache data/verify/tune_weights_v02_midpoint.json \
        --train-ratio 0.5 \
        --kfold 5 \
        --seed 42 \
        --out data/verify/holdout_eval.json

出力:
    data/verify/holdout_eval.json
        - default_train_acc / default_test_acc
        - grid_full_acc                : 50 試合全体 grid search (過適合参考値)
        - grid_holdout_train_acc / grid_holdout_test_acc
        - kfold_grid_test_accs / kfold_grid_test_mean / kfold_grid_test_std
        - lr_holdout_test_acc / lr_kfold_test_mean   (learn_weights_lr.py の結果, 後段で更新)
        - generalization_gap (train - test)
        - overfit_flag
        - comparison_table
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.old.tune_weights import (  # noqa: E402
    GRID_SEARCH_TARGETS,
    WEIGHT_GRID,
    MatchSample,
    evaluate_weights,
    extract_samples,
    grid_search,
    load_boundaries,
    load_winners,
)
from src.old.scorer import DEFAULT_WEIGHTS  # noqa: E402

# ============================
# 定数
# ============================

# 過適合判定のしきい値 (train_acc - test_acc がこれ以上なら過適合フラグ)
OVERFIT_GAP_THRESHOLD: float = 0.10

# K-fold 既定分割数
DEFAULT_KFOLD_K: int = 5

# 既定 train 比率
DEFAULT_TRAIN_RATIO: float = 0.5

# 既定 random seed
DEFAULT_SEED: int = 42


# ============================
# データ split
# ============================


@dataclass(frozen=True)
class HoldoutSplit:
    """train/test に分割されたサンプル列。"""
    train: list[MatchSample]
    test: list[MatchSample]
    seed: int
    train_ratio: float


def random_split(
    samples: list[MatchSample],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> HoldoutSplit:
    """シード固定で samples を train/test に分割する。

    Args:
        samples: 分割対象。
        train_ratio: train 比率 (0.0〜1.0)。
        seed: 乱数シード (再現性確保)。

    Returns:
        HoldoutSplit: 分割結果。
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1): {train_ratio}")
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    cut = int(round(len(samples) * train_ratio))
    train_idx = sorted(indices[:cut])
    test_idx = sorted(indices[cut:])
    return HoldoutSplit(
        train=[samples[i] for i in train_idx],
        test=[samples[i] for i in test_idx],
        seed=seed,
        train_ratio=train_ratio,
    )


def kfold_split(
    samples: list[MatchSample],
    k: int = DEFAULT_KFOLD_K,
    seed: int = DEFAULT_SEED,
) -> list[tuple[list[MatchSample], list[MatchSample]]]:
    """K-fold 分割を返す。各要素は (train, test) のタプル。

    Args:
        samples: 分割対象。
        k: 分割数。
        seed: 乱数シード。

    Returns:
        長さ k の (train, test) タプル列。fold 同士の test は disjoint。
    """
    if k < 2:
        raise ValueError(f"k must be >= 2: {k}")
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    out: list[tuple[list[MatchSample], list[MatchSample]]] = []
    for f in range(k):
        test_idx = sorted(folds[f])
        train_idx = sorted(i for i in indices if i not in set(test_idx))
        out.append((
            [samples[i] for i in train_idx],
            [samples[i] for i in test_idx],
        ))
    return out


# ============================
# サンプル取得 (cache or video)
# ============================


def load_samples_from_cache(cache_path: Path) -> list[MatchSample]:
    """tune_weights.py 出力 JSON から MatchSample 列を復元する。

    動画再読み込みを避けるため、既存キャッシュをそのまま読み戻す。
    """
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    out: list[MatchSample] = []
    for rec in payload.get("per_match_features", []):
        out.append(MatchSample(
            idx=int(rec["idx"]),
            end_sec=float(rec["end_sec"]),
            winner=str(rec["winner"]),
            p1_scores={k: float(v) for k, v in rec["p1"].items()},
            p2_scores={k: float(v) for k, v in rec["p2"].items()},
        ))
    return out


def get_samples(args: argparse.Namespace) -> list[MatchSample]:
    """--features-cache 優先で MatchSample 列を返す。"""
    if args.features_cache and args.features_cache.exists():
        print(f"[load] features cache: {args.features_cache}")
        return load_samples_from_cache(args.features_cache)
    if args.video is None or not args.video.exists():
        raise SystemExit(f"--video が必要 (見つからない: {args.video})")
    winners = load_winners(args.winners)
    boundaries = load_boundaries(args.boundaries)
    samples = extract_samples(
        args.video, winners,
        boundaries=boundaries,
        sample_mode=args.time_mode,
        offset_sec=args.offset_sec,
    )
    print(f"[extract] {len(samples)} / {len(winners)} 試合")
    return samples


# ============================
# 評価ロジック (holdout / kfold)
# ============================


@dataclass(frozen=True)
class HoldoutResult:
    """train/test 単一分割での grid search 評価結果。"""
    train_acc: float
    test_acc: float
    best_weights: dict[str, float]
    n_train: int
    n_test: int

    def gap(self) -> float:
        """train_acc - test_acc (汎化ギャップ)。"""
        return self.train_acc - self.test_acc


def evaluate_grid_holdout(split: HoldoutSplit) -> HoldoutResult:
    """train で grid search → train/test 精度を返す。"""
    best_w, train_acc = grid_search(split.train, DEFAULT_WEIGHTS)
    test_acc = evaluate_weights(split.test, best_w)
    return HoldoutResult(
        train_acc=train_acc,
        test_acc=test_acc,
        best_weights=best_w,
        n_train=len(split.train),
        n_test=len(split.test),
    )


def evaluate_grid_kfold(
    samples: list[MatchSample],
    k: int = DEFAULT_KFOLD_K,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """K-fold で grid search の汎化精度を測る。"""
    folds = kfold_split(samples, k=k, seed=seed)
    fold_results: list[dict[str, Any]] = []
    for i, (train, test) in enumerate(folds):
        best_w, train_acc = grid_search(train, DEFAULT_WEIGHTS)
        test_acc = evaluate_weights(test, best_w)
        fold_results.append({
            "fold": i,
            "n_train": len(train),
            "n_test": len(test),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_weights": {k_: float(v) for k_, v in best_w.items()},
        })
    test_accs = [f["test_acc"] for f in fold_results]
    return {
        "k": k,
        "seed": seed,
        "fold_results": fold_results,
        "test_mean": statistics.fmean(test_accs),
        "test_std": (
            statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0
        ),
        "train_mean": statistics.fmean(
            [f["train_acc"] for f in fold_results]
        ),
    }


# ============================
# レポート生成
# ============================


def build_report(
    samples: list[MatchSample],
    split: HoldoutSplit,
    holdout_result: HoldoutResult,
    kfold_result: dict[str, Any],
    grid_full_acc: float,
    default_full_acc: float,
    default_train_acc: float,
    default_test_acc: float,
    lr_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """holdout + kfold + 比較表を含むレポート辞書を作る。

    Args:
        lr_results: {正則化強度: {holdout, kfold}} があれば table と JSON に追加。
    """
    overfit_flag = holdout_result.gap() >= OVERFIT_GAP_THRESHOLD
    table = _make_comparison_table(
        default_full_acc=default_full_acc,
        default_train_acc=default_train_acc,
        default_test_acc=default_test_acc,
        grid_full_acc=grid_full_acc,
        holdout_result=holdout_result,
        kfold_result=kfold_result,
        lr_results=lr_results,
    )
    payload: dict[str, Any] = {
        "n_samples": len(samples),
        "seed": split.seed,
        "train_ratio": split.train_ratio,
        "default_full_acc": default_full_acc,
        "default_train_acc": default_train_acc,
        "default_test_acc": default_test_acc,
        "grid_full_acc": grid_full_acc,
        "grid_holdout_train_acc": holdout_result.train_acc,
        "grid_holdout_test_acc": holdout_result.test_acc,
        "grid_holdout_best_weights": holdout_result.best_weights,
        "generalization_gap": holdout_result.gap(),
        "overfit_flag": overfit_flag,
        "kfold": kfold_result,
        "comparison_table": table,
    }
    if lr_results:
        payload["lr_results"] = lr_results
    return payload


def _make_comparison_table(
    default_full_acc: float,
    default_train_acc: float,
    default_test_acc: float,
    grid_full_acc: float,
    holdout_result: HoldoutResult,
    kfold_result: dict[str, Any],
    lr_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """方法別 train/test 精度の比較表を返す。

    Args:
        lr_results: LR の結果 ({reg: {holdout, kfold}}) があれば末尾に追加。
    """
    rows: list[dict[str, Any]] = [
        {
            "method": "DEFAULT_WEIGHTS",
            "train_acc": default_train_acc,
            "test_acc": default_test_acc,
            "full_acc": default_full_acc,
            "note": "経験的重み (基準)",
        },
        {
            "method": "Grid Search (full 50, 過適合参考)",
            "train_acc": grid_full_acc,
            "test_acc": grid_full_acc,
            "full_acc": grid_full_acc,
            "note": "train==test==同セット → 過適合可能性大",
        },
        {
            "method": "Grid Search (holdout train→test)",
            "train_acc": holdout_result.train_acc,
            "test_acc": holdout_result.test_acc,
            "full_acc": None,
            "note": f"真の汎化精度 (gap={holdout_result.gap():+.3f})",
        },
        {
            "method": f"Grid Search (K-fold k={kfold_result['k']})",
            "train_acc": kfold_result["train_mean"],
            "test_acc": kfold_result["test_mean"],
            "full_acc": None,
            "note": f"std={kfold_result['test_std']:.3f}",
        },
    ]
    if lr_results:
        for reg, res in lr_results.items():
            holdout = res["holdout"]
            kfold = res["kfold"]
            rows.append({
                "method": f"LR (holdout, reg={reg})",
                "train_acc": holdout["train_acc"],
                "test_acc": holdout["test_acc"],
                "full_acc": None,
                "note": (
                    f"gap={holdout['generalization_gap']:+.3f}, "
                    f"overfit={holdout['overfit_flag']}"
                ),
            })
            rows.append({
                "method": f"LR (K-fold k={kfold['k']}, reg={reg})",
                "train_acc": kfold["train_mean"],
                "test_acc": kfold["test_mean"],
                "full_acc": None,
                "note": f"std={kfold['test_std']:.3f}",
            })
    return rows


def print_report(report: dict[str, Any]) -> None:
    """比較レポートを stdout に整形出力する。"""
    print("\n========== Holdout 評価レポート ==========")
    print(f"n_samples={report['n_samples']}, "
          f"seed={report['seed']}, "
          f"train_ratio={report['train_ratio']}")
    print(f"overfit_flag = {report['overfit_flag']} "
          f"(gap={report['generalization_gap']:+.3f}, "
          f"threshold={OVERFIT_GAP_THRESHOLD:+.3f})")
    print()
    header = f"{'method':<40s} {'train_acc':>10s} {'test_acc':>10s}  note"
    print(header)
    print("-" * len(header))
    for row in report["comparison_table"]:
        train_s = (
            f"{row['train_acc']:.3f}" if row["train_acc"] is not None
            else "  -  "
        )
        test_s = (
            f"{row['test_acc']:.3f}" if row["test_acc"] is not None
            else "  -  "
        )
        print(
            f"{row['method']:<40s} {train_s:>10s} {test_s:>10s}  "
            f"{row['note']}"
        )
    print()
    print("[K-fold 各 fold]")
    for f in report["kfold"]["fold_results"]:
        print(f"  fold {f['fold']}: train={f['train_acc']:.3f}, "
              f"test={f['test_acc']:.3f} "
              f"(n_train={f['n_train']}, n_test={f['n_test']})")


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重み学習の train/test holdout + K-fold 評価",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--winners", type=Path,
        default=Path("data/verify/match_winners_v02.tsv"),
    )
    parser.add_argument(
        "--boundaries", type=Path,
        default=Path("data/verify/match_boundaries_v4/video_02/matches.tsv"),
    )
    parser.add_argument(
        "--features-cache", type=Path, default=None,
        help="tune_weights.py の出力 JSON。指定時は動画を読まない",
    )
    parser.add_argument(
        "--time-mode", choices=("end", "midpoint"), default="midpoint",
    )
    parser.add_argument("--offset-sec", type=float, default=3.0)
    parser.add_argument(
        "--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
    )
    parser.add_argument("--kfold", type=int, default=DEFAULT_KFOLD_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/verify/holdout_eval.json"),
    )
    parser.add_argument(
        "--include-lr", action="store_true",
        help="LR (sklearn) ベース重み学習結果を comparison_table に併記",
    )
    parser.add_argument(
        "--lr-regularizations", type=str, default="0.5,5.0",
        help="比較対象の L2 正則化強度 (カンマ区切り)",
    )
    args = parser.parse_args()

    samples = get_samples(args)
    if len(samples) < args.kfold * 2:
        print(f"[warn] サンプル数 {len(samples)} は kfold={args.kfold} に対して少なすぎ")

    # 1) DEFAULT_WEIGHTS の baseline
    default_full_acc = evaluate_weights(samples, DEFAULT_WEIGHTS)

    # 2) holdout split
    split = random_split(
        samples, train_ratio=args.train_ratio, seed=args.seed,
    )
    default_train_acc = evaluate_weights(split.train, DEFAULT_WEIGHTS)
    default_test_acc = evaluate_weights(split.test, DEFAULT_WEIGHTS)

    # 3) grid search 50 試合全体 (過適合参考)
    _, grid_full_acc = grid_search(samples, DEFAULT_WEIGHTS)

    # 4) grid search holdout
    holdout_result = evaluate_grid_holdout(split)

    # 5) K-fold
    kfold_result = evaluate_grid_kfold(
        samples, k=args.kfold, seed=args.seed,
    )

    # 6) (任意) LR 比較
    lr_results: dict[str, Any] | None = None
    if args.include_lr:
        from scripts.old.learn_weights_lr import (  # noqa: WPS433
            kfold_lr_scores, run_holdout_lr,
        )
        regs = [float(x) for x in args.lr_regularizations.split(",") if x]
        lr_results = {}
        for reg in regs:
            holdout_lr = run_holdout_lr(
                samples, train_ratio=args.train_ratio,
                regularization=reg, seed=args.seed,
            )
            kfold_lr = kfold_lr_scores(
                samples, regularization=reg,
                k=args.kfold, seed=args.seed,
            )
            lr_results[str(reg)] = {
                "holdout": holdout_lr, "kfold": kfold_lr,
            }

    report = build_report(
        samples=samples,
        split=split,
        holdout_result=holdout_result,
        kfold_result=kfold_result,
        grid_full_acc=grid_full_acc,
        default_full_acc=default_full_acc,
        default_train_acc=default_train_acc,
        default_test_acc=default_test_acc,
        lr_results=lr_results,
    )
    report["grid_targets"] = list(GRID_SEARCH_TARGETS)
    report["weight_grid"] = list(WEIGHT_GRID)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_report(report)
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
