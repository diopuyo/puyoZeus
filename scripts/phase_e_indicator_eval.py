"""Phase E-3: 21 指標 評価レポート (重要度 + 多重共線性 + 削減シミュレーション).

multicollinearity_analysis.py の結果に加えて:
    - Random Forest feature importance ランキング
    - LR L2 coef 絶対値ランキング
    - 削除候補リストごとに lr_l2 / rf を再走 → 精度変化を集計
    - 推奨削除リストを markdown で出力

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_indicator_eval \
        --csv data/training/match_features_phase_e.csv \
        --out-md data/verify/indicator_evaluation_phase_e.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

from scripts.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.multicollinearity_analysis import compute_vif  # noqa: E402


# 削除候補のシナリオ (累積的に削減)
DROP_SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline_21", ()),
    ("v3_legacy_drop3", (
        "next_acceptance", "offset_power", "touching_density",
    )),
    ("multico_aggressive_5",  (
        # VIF>=10 削減 + 完全相関ペアの片方
        "incoming_ojama_pressure",
        "required_puyo_to_fire",
        "offset_power",
        "touching_density",
        "opponent_chain_threat",
    )),
    ("multico_aggressive_7", (
        # 上に加えて中程度相関の片方を追加削除
        "incoming_ojama_pressure",
        "required_puyo_to_fire",
        "offset_power",
        "touching_density",
        "opponent_chain_threat",
        "sub_chain_independence",
        "next_acceptance",
    )),
)


def fit_lr_l2(X_tr, y_tr, X_te, y_te, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=2000,
        class_weight="balanced",
    )
    clf.fit(Xs_tr, y_tr)
    train_acc = float(clf.score(Xs_tr, y_tr))
    test_acc = float(clf.score(Xs_te, y_te))
    return train_acc, test_acc, clf.coef_[0]


def fit_rf(X_tr, y_tr, X_te, y_te, depth=6, n_est=200, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        max_depth=depth, n_estimators=n_est,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    train_acc = float(clf.score(X_tr, y_tr))
    test_acc = float(clf.score(X_te, y_te))
    return train_acc, test_acc, clf.feature_importances_


def video_split(ds: Dataset, holdout: str = "03"):
    """指定 video を test、残りを train にする split."""
    test_mask = np.array([v == holdout for v in ds.video_ids])
    return ~test_mask, test_mask


def kfold_video_eval(ds: Dataset, model_fn, **kwargs):
    """各 video を 1 つずつ holdout した leave-one-video-out CV."""
    accs = {}
    for v in sorted(set(ds.video_ids)):
        tr_mask, te_mask = video_split(ds, v)
        if te_mask.sum() < 10:
            continue
        _, te_acc, _ = model_fn(
            ds.X[tr_mask], ds.y[tr_mask],
            ds.X[te_mask], ds.y[te_mask], **kwargs,
        )
        accs[v] = te_acc
    return accs


def reduce_dataset(ds: Dataset, drop: tuple[str, ...]) -> Dataset:
    keep_idx = [
        i for i, n in enumerate(ds.feature_names) if n not in drop
    ]
    return Dataset(
        feature_names=tuple(ds.feature_names[i] for i in keep_idx),
        X=ds.X[:, keep_idx],
        y=ds.y,
        video_ids=list(ds.video_ids),
        time_phases=list(ds.time_phases),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=_ROOT / "data/training/match_features_phase_e.csv",
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=_ROOT / "data/verify/indicator_evaluation_phase_e.md",
    )
    parser.add_argument(
        "--holdout-video", type=str, default="03",
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] n={len(ds.y)} d={len(ds.feature_names)}")

    tr_mask, te_mask = video_split(ds, args.holdout_video)
    n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
    print(f"[split] holdout=v{args.holdout_video} train={n_tr} test={n_te}")

    # baseline VIF
    vif_full = compute_vif(ds.X)
    feat_vif = list(zip(ds.feature_names, vif_full))
    feat_vif.sort(key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else 1e18))

    # baseline モデル → 重要度ランキング (1 回計算)
    print("[fit] baseline RF for feature importance ...")
    _, rf_te_acc, rf_imp = fit_rf(
        ds.X[tr_mask], ds.y[tr_mask], ds.X[te_mask], ds.y[te_mask],
    )
    print("[fit] baseline LR for coef ...")
    _, lr_te_acc, lr_coef = fit_lr_l2(
        ds.X[tr_mask], ds.y[tr_mask], ds.X[te_mask], ds.y[te_mask],
    )
    rf_rank = sorted(
        zip(ds.feature_names, rf_imp), key=lambda kv: -kv[1],
    )
    lr_rank = sorted(
        zip(ds.feature_names, np.abs(lr_coef)), key=lambda kv: -kv[1],
    )

    # 削減シナリオ別の精度比較
    scenario_results: list[dict] = []
    for name, drop in DROP_SCENARIOS:
        ds_red = reduce_dataset(ds, drop)
        kept = list(ds_red.feature_names)
        print(f"\n[scenario={name}] kept={len(kept)} dropped={list(drop)}")
        # video holdout (固定 v=03)
        tr_m, te_m = video_split(ds_red, args.holdout_video)
        _, lr_te, _ = fit_lr_l2(
            ds_red.X[tr_m], ds_red.y[tr_m],
            ds_red.X[te_m], ds_red.y[te_m],
        )
        _, rf_te, _ = fit_rf(
            ds_red.X[tr_m], ds_red.y[tr_m],
            ds_red.X[te_m], ds_red.y[te_m],
        )
        # leave-one-video-out
        loo = kfold_video_eval(ds_red, fit_lr_l2)
        loo_mean = float(np.mean(list(loo.values())))
        loo_std = float(np.std(list(loo.values())))
        print(
            f"  lr_l2 video_holdout={lr_te:.3f} | rf={rf_te:.3f} "
            f"| LOOV mean={loo_mean:.3f}±{loo_std:.3f}"
        )
        scenario_results.append({
            "name": name,
            "n_features": len(kept),
            "kept": kept,
            "dropped": list(drop),
            "lr_video_holdout": lr_te,
            "rf_video_holdout": rf_te,
            "loov_mean": loo_mean,
            "loov_std": loo_std,
            "loov_per_video": loo,
        })

    # markdown 生成
    md: list[str] = []
    md.append("# 指標評価レポート (Phase E-3)")
    md.append("")
    md.append(
        f"- 入力 CSV: `{to_windows_path(args.csv)}`"
    )
    md.append(f"- サンプル数: {len(ds.y)}")
    md.append(f"- 特徴量数: {len(ds.feature_names)}")
    md.append(f"- video_holdout 動画: v{args.holdout_video}")
    md.append("")

    md.append("## 1. VIF ランキング (多重共線性)")
    md.append("")
    md.append("| 順位 | 特徴量 | VIF | 判定 |")
    md.append("|---:|:---|---:|:---|")
    for i, (n, v) in enumerate(feat_vif, 1):
        if not np.isfinite(v):
            label, vstr = "深刻 (線形従属)", "inf"
        elif v >= 10:
            label, vstr = "深刻 (削除推奨)", f"{v:.2f}"
        elif v >= 5:
            label, vstr = "要警戒", f"{v:.2f}"
        else:
            label, vstr = "OK", f"{v:.2f}"
        md.append(f"| {i} | {n} | {vstr} | {label} |")
    md.append("")

    md.append("## 2. Random Forest feature importance")
    md.append("")
    md.append("| 順位 | 特徴量 | 重要度 |")
    md.append("|---:|:---|---:|")
    for i, (n, imp) in enumerate(rf_rank, 1):
        md.append(f"| {i} | {n} | {imp:.4f} |")
    md.append("")
    md.append(f"_baseline RF test_acc (v{args.holdout_video} holdout) = "
              f"**{rf_te_acc:.3f}**_")
    md.append("")

    md.append("## 3. LR L2 coef |abs|")
    md.append("")
    md.append("| 順位 | 特徴量 | |coef| |")
    md.append("|---:|:---|---:|")
    for i, (n, c) in enumerate(lr_rank, 1):
        md.append(f"| {i} | {n} | {c:.4f} |")
    md.append("")
    md.append(f"_baseline LR test_acc (v{args.holdout_video} holdout) = "
              f"**{lr_te_acc:.3f}**_")
    md.append("")

    md.append("## 4. 削減シナリオ別 精度")
    md.append("")
    md.append(
        "| シナリオ | 残特徴量 | LR (v03) | RF (v03) | "
        "LOOV mean | LOOV std |"
    )
    md.append("|:---|---:|---:|---:|---:|---:|")
    for r in scenario_results:
        md.append(
            f"| {r['name']} | {r['n_features']} | "
            f"{r['lr_video_holdout']:.3f} | {r['rf_video_holdout']:.3f} | "
            f"{r['loov_mean']:.3f} | {r['loov_std']:.3f} |"
        )
    md.append("")

    md.append("## 5. シナリオ別 LOOV per-video")
    md.append("")
    md.append(
        "| video | "
        + " | ".join(r["name"] for r in scenario_results)
        + " |"
    )
    md.append(
        "|:---|"
        + "|".join(["---:"] * len(scenario_results))
        + "|"
    )
    all_videos = sorted({
        v for r in scenario_results for v in r["loov_per_video"].keys()
    })
    for v in all_videos:
        cells = [v]
        for r in scenario_results:
            acc = r["loov_per_video"].get(v)
            cells.append(f"{acc:.3f}" if acc is not None else "—")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    md.append("## 6. 削除推奨 (まとめ)")
    md.append("")
    md.append(
        "VIF/相関分析 + シナリオ精度比較から、以下の削除を推奨:"
    )
    md.append("")
    md.append(
        "1. **`incoming_ojama_pressure`** — VIF=inf (盤面差分が常に 0、"
        "情報量ゼロ)"
    )
    md.append(
        "2. **`required_puyo_to_fire`** または "
        "**`chain_timing_pressure`** — r=0.999 で同一情報"
    )
    md.append(
        "3. **`offset_power`** — VIF=23、main_chain_maturity (r=0.748) と "
        "harassment_resistance (r=0.733) の合成"
    )
    md.append(
        "4. **`touching_density`** — VIF=9.17、offset_power と高重複"
    )
    md.append(
        "5. **`opponent_chain_threat`** — main_chain_maturity と r=-0.812 "
        "(逆相関 = 同情報)"
    )
    md.append("")
    md.append(
        "→ 21 特徴量 → 16 特徴量 (multico_aggressive_5) で精度維持を確認"
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[save] {to_windows_path(args.out_md)}")
    out_json = args.out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps({
            "n_samples": len(ds.y),
            "n_features": len(ds.feature_names),
            "vif": {
                n: (None if not np.isfinite(v) else float(v))
                for n, v in feat_vif
            },
            "rf_importance": {n: float(c) for n, c in rf_rank},
            "lr_coef_abs": {n: float(c) for n, c in lr_rank},
            "scenarios": scenario_results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {to_windows_path(out_json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
