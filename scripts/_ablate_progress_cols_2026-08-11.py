"""B-2 (PR #22 マージ済み) の留保事項「進行度列の寄与切り分け」測定。

**読み取り専用の測定タスク**。scripts/visualize_advantage_overlay.py は
import のみで一切編集しない (別コーダが作業中のため)。

背景 (2026-08-11 依頼):
B-2 で追加された2列 (66動画5foldで全体AUC 0.6605→0.6688, +0.0083):
  - match_progress_diff (進行度そのもの、side不変)
      permutation importance +0.01625
  - color_puyo_x_earliness_diff (色ぷよ差×序盤度)
      permutation importance +0.01122
仮説「序盤ほど色ぷよ差が効く」に対し、実測の改善幅は終盤が最大 (+0.0164) で
素直でない。「進行度という文脈自体」の効果と「色ぷよ×序盤度」の効果が
混ざっている可能性があるため、4構成アブレーションで切り分ける。

4構成 (scripts/_verify_progress_context_2026-08-10.py の型を流用):
  1. base            : 2列とも無し (= --no-progress 相当、0.6605 の再現確認)
  2. +progress_only  : match_progress_diff のみ追加
  3. +earliness_only : color_puyo_x_earliness_diff のみ追加 (内部計算は進行度
                        を使うが、進行度そのものは特徴量に含めない)
  4. +both           : 両方 (= 既定の本番構成、0.6688 の再現確認)

visualize_advantage_overlay._add_interaction_columns() は②のブロックで
両方の列を同じ if 節でまとめて追加するため、 個別追加はできない
(read-only 制約により同ファイルは変更不可)。 このスクリプトでは①の
おじゃまフラット交互作用ブロックのみ同関数を呼び出して再利用し (これは
全4構成で共通・既存の本番挙動と同一)、 ②の進行度/早さ交互作用列は
_match_progress_from_totals() (stateless ヘルパー、 import のみ) を使って
本スクリプト側で個別に組み立てる。 計算式は完全に同一実装を再利用するため
本番ロジックの複製にはならない。

使い方:
    python -m scripts._ablate_progress_cols_2026-08-11 --smoke   # 動作確認
    python -m scripts._ablate_progress_cols_2026-08-11           # 本測定
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.visualize_advantage_overlay import (  # noqa: E402
    COLOR_EARLINESS_INTERACTION_COL,
    MATCH_PROGRESS_COL,
    OJAMA_FLAT_COL,
    TRAIN_CSV_PATH,
    _add_interaction_columns,
    _match_progress_from_totals,
    _mirror_sign,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

# _verify_progress_context_2026-08-10.py と同じ層別・位相閾値 (踏襲)。
FLAT_HI: float = 0.8
FLAT_LO: float = 0.37
TSUMO_EARLY_RATIO: float = 0.33
TSUMO_LATE_RATIO: float = 0.67

N_FOLDS: int = 5
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42

# 既報値 (依頼仕様書・_verify_progress_context_2026-08-10.py の実測値)。
BASELINE_OVERALL_AUC: float = 0.6605
BOTH_OVERALL_AUC: float = 0.6688
# 再現許容誤差。既報値は小数4桁で丸められているため丸め誤差(最大0.00005)
# 以上のマージンを取る。これを超えたら測定器の相違とみなし中断する。
REPRO_TOLERANCE: float = 0.003

OUT_DIR = Path("data/verify/progress_ablation_2026-08-11")

# 4構成: (名前, progress列を含めるか, earliness列を含めるか)
CONFIGS: list[tuple[str, bool, bool]] = [
    ("1_base", False, False),
    ("2_progress_only", True, False),
    ("3_earliness_only", False, True),
    ("4_both", True, True),
]


def _prepare(
    smoke: bool, use_progress_col: bool, use_earliness_col: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """paired 特徴量・使用列を用意する (構成ごとに列の有無を切り替え)。"""
    df = load_labeled_csv(TRAIN_CSV_PATH)
    if smoke:
        vids = sorted(df["video_id"].unique())[:3]
        df = df[df["video_id"].isin(vids)].reset_index(drop=True)
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)

    # ブロック① (色ぷよ×おじゃまフラット度) は全4構成で共通・本番と同一実装を
    # そのまま再利用する (paired=None を渡すと②はスキップされる、
    # visualize_advantage_overlay.py 側の既存の列存在ガード挙動そのまま)。
    feat, cols = _add_interaction_columns(feat, feat_cols, paired=None)

    # ブロック② (進行度 / 色ぷよ×早さ) を構成ごとに個別追加。
    # _match_progress_from_totals() は本番と同じ stateless ヘルパーを import
    # して使うため、計算式は本番実装の複製ではなく再利用。
    progress_need = ("board_puyo_total_1p", "board_puyo_total_2p")
    has_progress_input = all(c in paired.columns for c in progress_need)
    if has_progress_input and "board_color_puyo_total_diff" in feat.columns:
        progress = _match_progress_from_totals(
            paired["board_puyo_total_1p"], paired["board_puyo_total_2p"],
        )
        if use_progress_col:
            feat[f"{MATCH_PROGRESS_COL}_diff"] = progress
            cols.append(f"{MATCH_PROGRESS_COL}_diff")
        if use_earliness_col:
            feat[f"{COLOR_EARLINESS_INTERACTION_COL}_diff"] = (
                feat["board_color_puyo_total_diff"] * (1.0 - progress)
            )
            cols.append(f"{COLOR_EARLINESS_INTERACTION_COL}_diff")
    elif use_progress_col or use_earliness_col:
        print("  [警告] board_puyo_total_{1p,2p} が無く進行度列を追加できません"
              " (列存在ガードによりベース構成へフォールバック)")

    feat["video_id_1p"] = paired["video_id_1p"].values
    feat["won_1p"] = paired["won_1p"].astype(int).values
    feat["tsumo_1p"] = paired["tsumo_1p"].astype(float).values
    return feat, cols


def _flat_score_from_diff(feat: pd.DataFrame) -> np.ndarray:
    col = f"{OJAMA_FLAT_COL}_diff"
    if col not in feat.columns:
        return np.full(len(feat), np.nan)
    return feat[col].fillna(0.0).values


def _auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _phase_masks(tsumo: np.ndarray) -> dict[str, np.ndarray]:
    q_low = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q_high = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q_low,
        "中盤": (tsumo > q_low) & (tsumo <= q_high),
        "終盤": tsumo > q_high,
    }


def run_config(
    name: str, use_progress_col: bool, use_earliness_col: bool,
    smoke: bool, gbc_params: dict,
) -> dict:
    """1構成分の OOF 評価を実行し、指標一式を dict で返す。"""
    print(f"\n{'='*70}\n構成: {name}"
          f" (progress={use_progress_col}, earliness={use_earliness_col},"
          f" smoke={smoke})\n{'='*70}")
    feat, cols = _prepare(smoke, use_progress_col, use_earliness_col)
    X = feat[cols].fillna(0.0).values
    y = feat["won_1p"].values
    groups = feat["video_id_1p"].values
    tsumo = feat["tsumo_1p"].values
    flat = _flat_score_from_diff(feat)

    progress_col = f"{MATCH_PROGRESS_COL}_diff"
    earliness_col = f"{COLOR_EARLINESS_INTERACTION_COL}_diff"
    perm_targets = [c for c in (progress_col, earliness_col) if c in cols]

    n_splits = 2 if smoke else N_FOLDS
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    perm_importances: dict[str, list[float]] = {n: [] for n in perm_targets}

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        sign = _mirror_sign(cols)
        X_tr_sym = np.vstack([X_tr, X_tr * sign])
        y_tr_sym = np.concatenate([y_tr, 1 - y_tr])
        model = HistGradientBoostingClassifier(**gbc_params)
        model.fit(X_tr_sym, y_tr_sym)
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]
        for pname in perm_targets:
            idx = cols.index(pname)
            pi = permutation_importance(
                model, X[te_idx], y[te_idx], n_repeats=PERM_N_REPEATS,
                random_state=PERM_RANDOM_STATE, scoring="roc_auc",
            )
            perm_importances[pname].append(float(pi.importances_mean[idx]))
        print(f"  fold {fold+1}/{n_splits} 完了 (train={len(tr_idx)} test={len(te_idx)})")

    valid = ~np.isnan(oof)
    overall_auc = _auc(y[valid], oof[valid])
    print(f"\n  全体 OOF AUC: {overall_auc:.4f}")

    flat_auc = float("nan")
    if not np.all(np.isnan(flat)):
        m = (flat >= FLAT_HI) & valid
        if m.sum() >= 50:
            flat_auc = _auc(y[m], oof[m])
        print(f"  おじゃまフラット (>= {FLAT_HI}) n={int(m.sum())}  AUC={flat_auc:.4f}")

    phase_auc: dict[str, float] = {}
    for phase, mask in _phase_masks(tsumo).items():
        m = mask & valid
        n = int(m.sum())
        if n < 50 or len(np.unique(y[m])) < 2:
            phase_auc[phase] = float("nan")
            print(f"    {phase}: n={n} (データ不足 -> nan)")
            continue
        phase_auc[phase] = _auc(y[m], oof[m])
        print(f"    {phase}: n={n:6d}  AUC={phase_auc[phase]:.4f}")

    perm_mean = {n: float(np.mean(v)) for n, v in perm_importances.items()}
    for pname, v in perm_mean.items():
        print(f"  permutation importance[{pname}]: {v:+.5f}")

    return {
        "config": name,
        "n_features": len(cols),
        "overall_auc": overall_auc,
        "flat_auc": flat_auc,
        "early_auc": phase_auc.get("序盤", float("nan")),
        "mid_auc": phase_auc.get("中盤", float("nan")),
        "late_auc": phase_auc.get("終盤", float("nan")),
        "perm_imp_progress": perm_mean.get(progress_col, float("nan")),
        "perm_imp_earliness": perm_mean.get(earliness_col, float("nan")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="動画3本・2fold・軽量パラメータで構造検証のみ行う")
    ap.add_argument("--skip-repro-check", action="store_true",
                     help="1/4構成の既報値再現チェックをスキップして4構成全て実行"
                          " (デバッグ用途のみ、通常は使わない)")
    a = ap.parse_args()
    gbc_params = dict(GBC_PARAMS)
    if a.smoke:
        gbc_params["max_iter"] = 30

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    # まず 1 (base) と 4 (both) を実行し、既報値 0.6605 / 0.6688 の再現を確認。
    # 再現しなければ測定器の相違なので、以降の構成は実行せず中断・報告する。
    for name, up, ue in [c for c in CONFIGS if c[0] in ("1_base", "4_both")]:
        results[name] = run_config(name, up, ue, a.smoke, gbc_params)

    if not a.smoke and not a.skip_repro_check:
        base_diff = abs(results["1_base"]["overall_auc"] - BASELINE_OVERALL_AUC)
        both_diff = abs(results["4_both"]["overall_auc"] - BOTH_OVERALL_AUC)
        print(f"\n{'='*70}\n再現確認: base={results['1_base']['overall_auc']:.4f}"
              f" (既報{BASELINE_OVERALL_AUC}, diff={base_diff:.4f})"
              f" / both={results['4_both']['overall_auc']:.4f}"
              f" (既報{BOTH_OVERALL_AUC}, diff={both_diff:.4f})\n{'='*70}")
        if base_diff > REPRO_TOLERANCE or both_diff > REPRO_TOLERANCE:
            print("\n[中断] 既報値との差が許容誤差を超えています。"
                  " 測定器 (データ/fold分割/パラメータ) の相違が疑われるため、"
                  " 2/3構成の実行を中止し、ここまでの結果のみ報告します。")
            _write_tsv(results)
            return 1
        print("[OK] 既報値を許容誤差内で再現。残り2構成の測定を続行します。")

    # 再現確認 OK (または smoke / スキップ指定) なら残り2構成を実行。
    for name, up, ue in [c for c in CONFIGS if c[0] in ("2_progress_only", "3_earliness_only")]:
        results[name] = run_config(name, up, ue, a.smoke, gbc_params)

    _write_tsv(results)
    _print_summary(results)
    return 0


def _write_tsv(results: dict[str, dict]) -> None:
    order = [c[0] for c in CONFIGS if c[0] in results]
    rows = [results[n] for n in order]
    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "ablation_results.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\n[保存] {out_path}")


def _print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'='*70}\n4構成サマリ\n{'='*70}")
    order = [c[0] for c in CONFIGS if c[0] in results]
    header = ["config", "overall", "flat", "早", "中", "終"]
    print("\t".join(header))
    for n in order:
        r = results[n]
        print(f"{n}\t{r['overall_auc']:.4f}\t{r['flat_auc']:.4f}\t"
              f"{r['early_auc']:.4f}\t{r['mid_auc']:.4f}\t{r['late_auc']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
