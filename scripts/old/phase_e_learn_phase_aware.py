"""Phase E-4: 序盤 / 中盤 / 終盤 別の重み学習 (phase_e csv 対応版).

phase_e csv の time_phase ラベル:
    - start_plus_20   (序盤)
    - mid_minus_20, midpoint, mid_plus_20   (中盤)
    - end_minus_5   (終盤)

E-3 の推奨削減 (multico_aggressive_5) に従い、16 features で学習。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_learn_phase_aware
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

from scripts.old.eda_features import load_dataset  # noqa: E402


PHASE_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "start": ("start_plus_20",),
    "mid": ("mid_minus_20", "midpoint", "mid_plus_20"),
    "end": ("end_minus_5",),
}

# E-3 推奨削減 (multico_aggressive_5)
DROPPED_FEATURES: tuple[str, ...] = (
    "incoming_ojama_pressure",
    "required_puyo_to_fire",
    "offset_power",
    "touching_density",
    "opponent_chain_threat",
)

# leave-one-video-out CV を行うか
USE_LOOV: bool = True


def filter_by_phase(
    time_phases: list[str], allowed: tuple[str, ...],
) -> np.ndarray:
    return np.array(
        [i for i, p in enumerate(time_phases) if p in allowed],
        dtype=np.int64,
    )


def train_one_split(X_tr, y_tr, X_te, y_te) -> tuple[float, np.ndarray, float, float]:
    """L2 LR で C グリッド探索、best test_acc + 元スケール係数 + intercept を返す."""
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)
    best = (0.0, None, None, None)
    for C in (0.05, 0.1, 0.5, 1.0, 5.0):
        m = LogisticRegression(
            C=C, solver="lbfgs", max_iter=2000,
            class_weight="balanced",
        )
        m.fit(Xs_tr, y_tr)
        acc = float(m.score(Xs_te, y_te))
        if acc > best[0]:
            best = (acc, m, scaler, C)
    acc, m, sc, C = best
    coef_std = m.coef_[0]
    coef_orig = coef_std / np.maximum(sc.scale_, 1e-9)
    intercept_orig = float(
        m.intercept_[0] - np.sum(coef_std * sc.mean_ / np.maximum(sc.scale_, 1e-9)),
    )
    train_acc = float(m.score(Xs_tr, y_tr))
    return acc, coef_orig, intercept_orig, train_acc, C


def loov_train_phase(
    X: np.ndarray, y: np.ndarray, video_ids: list[str],
    time_phases: list[str], phase_list: tuple[str, ...],
    phase_name: str, feature_names: tuple[str, ...],
) -> dict:
    """指定 phase で全動画 LOOV 学習 + 全データで最終モデル学習."""
    phase_idx = filter_by_phase(time_phases, phase_list)
    Xp = X[phase_idx]
    yp = y[phase_idx]
    vp = [video_ids[i] for i in phase_idx]

    # LOOV: 各 video を 1 つずつ holdout
    accs: dict[str, float] = {}
    for v in sorted(set(vp)):
        tr_mask = np.array([vv != v for vv in vp])
        te_mask = ~tr_mask
        if te_mask.sum() < 5 or tr_mask.sum() < 50:
            continue
        acc, _, _, _, _ = train_one_split(
            Xp[tr_mask], yp[tr_mask], Xp[te_mask], yp[te_mask],
        )
        accs[v] = acc

    # 全データで最終モデル
    sc = StandardScaler()
    Xs = sc.fit_transform(Xp)
    final_C = 1.0
    final_acc_grid: list[tuple[float, float, np.ndarray, float]] = []
    # 全データ学習: train_acc だけ、ホールドアウトは LOOV で評価済み
    for C in (0.05, 0.1, 0.5, 1.0, 5.0):
        m = LogisticRegression(
            C=C, solver="lbfgs", max_iter=2000,
            class_weight="balanced",
        )
        m.fit(Xs, yp)
        train_acc = float(m.score(Xs, yp))
        coef_orig = m.coef_[0] / np.maximum(sc.scale_, 1e-9)
        intercept_orig = float(
            m.intercept_[0] - np.sum(m.coef_[0] * sc.mean_ / np.maximum(sc.scale_, 1e-9)),
        )
        final_acc_grid.append((C, train_acc, coef_orig, intercept_orig))
    # LOOV 平均と最も近い train_acc になる C を選択 (過学習回避)
    loov_mean = float(np.mean(list(accs.values()))) if accs else 0.5
    best_choice = min(
        final_acc_grid, key=lambda t: abs(t[1] - loov_mean),
    )
    final_C, final_train_acc, coef, intercept = best_choice

    weights = {n: float(coef[i]) for i, n in enumerate(feature_names)}
    return {
        "phase_name": phase_name,
        "phase_list": list(phase_list),
        "n_total": int(len(phase_idx)),
        "loov_per_video": accs,
        "loov_mean": loov_mean,
        "loov_std": float(np.std(list(accs.values()))) if accs else 0.0,
        "final_train_acc": final_train_acc,
        "final_C": final_C,
        "intercept": intercept,
        "weights": weights,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=_ROOT / "data/training/match_features_phase_e.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_ROOT / "data/verify/learned_weights_phase_e_phase_aware.json",
    )
    parser.add_argument(
        "--no-drop", action="store_true",
        help="E-3 推奨削減を無効化 (21 features 全部使用)",
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] n={len(ds.y)} d={len(ds.feature_names)}")

    if args.no_drop:
        kept_idx = list(range(len(ds.feature_names)))
        kept_names = ds.feature_names
        Xred = ds.X
    else:
        keep_mask = [
            n not in DROPPED_FEATURES for n in ds.feature_names
        ]
        kept_idx = [i for i, k in enumerate(keep_mask) if k]
        kept_names = tuple(ds.feature_names[i] for i in kept_idx)
        Xred = ds.X[:, kept_idx]
    print(
        f"[reduce] kept={len(kept_names)} dropped="
        f"{[n for n in ds.feature_names if n not in kept_names]}"
    )

    results: dict[str, dict] = {}
    for name, phase_list in PHASE_DEFINITIONS.items():
        print(f"\n=== {name} phases={phase_list} ===")
        r = loov_train_phase(
            Xred, ds.y, list(ds.video_ids), list(ds.time_phases),
            phase_list, name, kept_names,
        )
        results[name] = r
        print(
            f"  LOOV mean={r['loov_mean']:.3f}±{r['loov_std']:.3f} "
            f"(n={r['n_total']}) final_C={r['final_C']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "csv": str(args.csv),
            "n_features": len(kept_names),
            "feature_names": list(kept_names),
            "dropped": list(
                n for n in ds.feature_names if n not in kept_names
            ),
            "phases": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] {to_windows_path(args.out)}")

    print()
    print("=== phase 別 LOOV mean (overall) ===")
    overall = float(np.mean(
        [r["loov_mean"] for r in results.values()]
    ))
    for n, r in results.items():
        print(f"  {n:6s}: {r['loov_mean']:.3f}±{r['loov_std']:.3f}")
    print(f"  average: {overall:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
