"""near_future_fire / fire_stability を含めた win予測モデルの学習・評価 (#35)。

user指示 (2026-07-22): labeled_win.csv (#33でnear_future列追加済み) を使い、
scripts/model_indicator_win.py の既存手法 (HistGBC + GroupKFold(video_id) OOF、
フェーズ別=序盤/中盤/終盤) で以下を比較する:
    (a) near_future無し (既存特徴のみ)
    (b) near_future_fire_k1-5 (+fire_stability_k2,4,6) 追加

## 正直な留保 (誇張しない)
near_future/fire_stability 列が埋まっているのは labeled_win.csv 全40112行中
5740行 (14.3%、board-backed) のみ。ペアリング後、**両側 (1P/2P とも)** に
値がある「真に near_future が効く」行は 752行・4動画 (video_29/35/36/37) に
さらに絞られる。そのため本スクリプトは
    (1) 全データ (6049ペア・10動画) での比較
    (2) board-backed 限定 (752ペア・4動画、near_futureが実際に効く土俵)
の両方を出し、**(2) での中盤liftを主指標とする**。

測定・学習のみ。src本体・chain_bitboardは変更しない。
model_indicator_win.py の既存関数をそのまま再利用する (新規追加なし)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts.verify_near_future_win_model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402

LABELED_WIN_CSV = "data/indicators_v2/study/labeled_win.csv"

# near_future/fire_stability の基底列名 (indicator_cols の要素、接尾辞なし)。
NEAR_FUTURE_COLS: "tuple[str, ...]" = tuple(f"near_future_fire_k{k}" for k in range(1, 6))
FIRE_STABILITY_COLS: "tuple[str, ...]" = tuple(f"fire_stability_k{k}" for k in (2, 4, 6))
NEW_COLS: "tuple[str, ...]" = NEAR_FUTURE_COLS + FIRE_STABILITY_COLS

N_BOOTSTRAP: int = 500
BOOTSTRAP_SEED: int = 20260722
MIN_BOOTSTRAP_ROWS: int = 20
MIN_BOOTSTRAP_VALID: int = 20


def _oof_auc_for_cols(
    paired: pd.DataFrame, cols: "list[str]", y: np.ndarray, groups: np.ndarray, n_folds: int,
) -> "tuple[np.ndarray, float]":
    """指定 indicator_cols で HistGBC OOF を実行し (oof_proba[:,1], overall AUC) を返す。"""
    feat = miw.build_features(paired, cols)
    X = feat.fillna(0.0).values.astype(float)
    oof_proba, _ = miw.run_oof_classifier(X, y, groups, n_folds)
    valid = ~np.isnan(oof_proba[:, 0])
    auc = float(roc_auc_score(y[valid], oof_proba[valid, 1])) if len(np.unique(y[valid])) > 1 else float("nan")
    return oof_proba[:, 1], auc


def _phase_masks(paired: pd.DataFrame) -> "dict[str, np.ndarray]":
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, miw.TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, miw.TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _auc_from_oof(y: np.ndarray, p1: np.ndarray, mask: "np.ndarray | None" = None) -> "tuple[float, int]":
    yy, pp = (y, p1) if mask is None else (y[mask], p1[mask])
    valid = ~np.isnan(pp)
    yy, pp = yy[valid], pp[valid]
    if len(pp) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(pp)
    return float(roc_auc_score(yy, pp)), len(pp)


def _bootstrap_ci_from_oof(
    y: np.ndarray, p1: np.ndarray, groups: np.ndarray, mask: "np.ndarray | None",
) -> "tuple[float, float]":
    """OOF確率 (既に学習済み) を video 単位でブートストラップし AUC の95%CIを返す。

    モデル再学習はせず、既に得られた OOF 予測確率を video 単位で再抽出して
    AUC を再計算するだけ (計算コストが軽い、既存の point-biserial検証と同じ方式)。
    """
    if mask is not None:
        yy, pp, gg = y[mask], p1[mask], groups[mask]
    else:
        yy, pp, gg = y, p1, groups
    valid = ~np.isnan(pp)
    yy, pp, gg = yy[valid], pp[valid], gg[valid]
    videos = np.unique(gg)
    if len(videos) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aucs: "list[float]" = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        idx = np.concatenate([np.where(gg == v)[0] for v in sampled])
        if len(idx) < MIN_BOOTSTRAP_ROWS or len(np.unique(yy[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yy[idx], pp[idx]))
    if len(aucs) < MIN_BOOTSTRAP_VALID:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _run_scope(scope_name: str, paired: pd.DataFrame) -> None:
    """1つのデータスコープ (全データ / board-backed) について baseline vs with_nf を評価する。"""
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    n_videos = paired["video_id_1p"].nunique()
    n_folds = min(miw.N_FOLDS, max(2, n_videos))
    masks = _phase_masks(paired)

    indicator_cols_all = miw._get_indicator_cols(paired)
    baseline_cols = [c for c in indicator_cols_all if c not in NEW_COLS]
    with_nf_cols = indicator_cols_all  # near_future/fire_stability は自動的に含まれる

    print()
    print("=" * 100)
    print(f"  スコープ: {scope_name}  (n={len(paired)}, video数={n_videos}, fold数={n_folds})")
    print("=" * 100)

    print("  [baseline: near_future/fire_stability 無し] OOF学習中...")
    p1_base, auc_base_all = _oof_auc_for_cols(paired, baseline_cols, y, groups, n_folds)
    print("  [with_near_future: 追加あり] OOF学習中...")
    p1_nf, auc_nf_all = _oof_auc_for_cols(paired, with_nf_cols, y, groups, n_folds)

    print()
    header = f"  {'区分':<8}  {'baseline':>10}  {'with_nf':>10}  {'delta':>8}  {'n':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, mask in [("全体", None), *masks.items()]:
        auc_b, n_b = _auc_from_oof(y, p1_base, mask)
        auc_n, n_n = _auc_from_oof(y, p1_nf, mask)
        delta = auc_n - auc_b if not (np.isnan(auc_b) or np.isnan(auc_n)) else float("nan")
        print(f"  {label:<8}  {auc_b:>10.4f}  {auc_n:>10.4f}  {delta:>+8.4f}  {n_n:>5}")

    print()
    print("  --- ノイズ幅 (video単位ブートストラップ 95%CI, OOF確率の再利用) ---")
    for label, mask in [("全体", None), *masks.items()]:
        lo_b, hi_b = _bootstrap_ci_from_oof(y, p1_base, groups, mask)
        lo_n, hi_n = _bootstrap_ci_from_oof(y, p1_nf, groups, mask)
        print(
            f"  {label}: baseline 95%CI=[{lo_b:.4f},{hi_b:.4f}]  "
            f"with_nf 95%CI=[{lo_n:.4f},{hi_n:.4f}]",
        )

    print()
    print("  --- Permutation Importance (with_near_future特徴セット、上位20 + near_future/fire_stability全件) ---")
    feat_nf = miw.build_features(paired, with_nf_cols)
    X_nf = feat_nf.fillna(0.0).values.astype(float)
    perm_df = miw.compute_perm_importance_win(
        X_nf, y, groups, list(feat_nf.columns), n_folds,
    )
    print(perm_df.head(20).to_string(index=False))
    print()
    is_new = perm_df["feature"].str.replace(r"_(1p|2p|diff)$", "", regex=True).isin(NEW_COLS)
    print("  near_future/fire_stability 由来特徴のランキング:")
    print(perm_df[is_new].to_string(index=False))


def main() -> None:
    print("=== データ読み込み・ペアリング (全データ) ===")
    df = miw.load_labeled_csv(LABELED_WIN_CSV)
    paired_all = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)

    both_present = (
        paired_all["near_future_fire_k1_1p"].notna()
        & paired_all["near_future_fire_k1_2p"].notna()
    )
    paired_board_backed = paired_all[both_present].reset_index(drop=True)
    print(
        f"  board-backed (near_future 両側とも計算済み): {len(paired_board_backed)} 行 / "
        f"{paired_board_backed['video_id_1p'].nunique()} 動画",
    )

    _run_scope("(1) 全データ", paired_all)
    _run_scope("(2) board-backed限定 (near_futureが実際に効く土俵)", paired_board_backed)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
