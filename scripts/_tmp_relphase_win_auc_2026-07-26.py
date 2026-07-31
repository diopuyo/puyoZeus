"""相対位相(セグメント内進行率)で位相分割した新データの win-AUC を評価する。

## 背景 (2026-07-26 コーディネーター依頼、project_win_eval_regen_2026-07-26)
従来の位相分割 (序盤/中盤/終盤) は手数 (tsumo) の絶対値をプール全体の三分位で
区切っていた。これは試合ごとの手数長がばらつく (新データ seg_max 中央値45・
IQR 34-56・最大146) 場合に「終盤」の意味がぶれる欠陥がある。

本スクリプトは各行が属する試合 (tsumo>=10 -> tsumo<=3 の reset で区切られた
セグメント) を検出し、位相 = tsumo / セグメント最大手数 (相対進行率) で
序盤(<1/3)・中盤(1/3-2/3)・終盤(>2/3) を再定義して同一 HistGBC OOF AUC を
再評価する。seg_max>60 (境界検知取りこぼし濃厚) セグメントは汚染疑いとして
除外版も出す。

model_indicator_win.py は変更しない (import して関数を再利用するのみ、
既存互換完全維持)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402

STUDY_DIR = Path("data/verify/labeled_win_regen_2026-07-26/study")
LABELED_WIN_CSV = "data/verify/labeled_win_regen_2026-07-26/labeled_win.csv"
OUT_DIR = Path("data/verify/win_eval_regen_2026-07-26")

# 試合境界 reset 検出しきい値 (tsumo_segments_new.csv と同一ロジック)
RESET_FROM_MIN: float = 10.0
RESET_TO_MAX: float = 3.0
# 汚染疑いセグメントしきい値 (1試合の現実的手数上限, project既定)
CONTAMINATION_SEG_MAX: float = 60.0
# 相対位相境界
REL_EARLY: float = 1.0 / 3.0
REL_LATE: float = 2.0 / 3.0


def _detect_segments(tsumo: np.ndarray) -> np.ndarray:
    """時系列 tsumo 配列からセグメント ID (0始まり) を返す。"""
    seg_id = np.zeros(len(tsumo), dtype=int)
    cur = 0
    for i in range(1, len(tsumo)):
        if tsumo[i - 1] >= RESET_FROM_MIN and tsumo[i] <= RESET_TO_MAX:
            cur += 1
        seg_id[i] = cur
    return seg_id


def build_phase_map(study_dir: Path) -> pd.DataFrame:
    """study CSV 群から (video_id, side, t_sec) -> (seg_max, rel_phase) の対応表を作る。"""
    rows: list[dict] = []
    for path in sorted(study_dir.glob("*.csv")):
        if path.name.startswith(("corr_", "labeled_")):
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=["video_id", "side"])
        for (vid, side), grp in df.groupby(["video_id", "side"]):
            grp = grp.sort_values("t_sec")
            tsumo = grp["tsumo"].astype(float).values
            seg_id = _detect_segments(tsumo)
            for sid in np.unique(seg_id):
                mask = seg_id == sid
                seg_max = float(tsumo[mask].max())
                rel_phase = np.where(seg_max > 0, tsumo[mask] / max(seg_max, 1e-6), 0.0)
                t_secs = grp["t_sec"].values[mask]
                for t, tv, rp in zip(t_secs, tsumo[mask], rel_phase):
                    rows.append({
                        "video_id": vid, "side": side, "t_sec": float(t),
                        "tsumo_raw": float(tv), "seg_max": seg_max,
                        "rel_phase": float(rp),
                    })
    out = pd.DataFrame(rows)
    print(f"[phase_map] 構築完了: {len(out)} 行 (video*side*t_sec単位)")
    return out


def merge_phase(paired: pd.DataFrame, phase_map: pd.DataFrame) -> pd.DataFrame:
    """paired (1p/2p) に 1P 側の rel_phase / seg_max を結合する (既存 tsumo_1p 基準の慣例踏襲)。"""
    pm = phase_map.rename(columns={
        "video_id": "video_id_1p", "side": "side_1p", "t_sec": "t_sec_1p",
        "seg_max": "seg_max_1p", "rel_phase": "rel_phase_1p",
    })[["video_id_1p", "side_1p", "t_sec_1p", "seg_max_1p", "rel_phase_1p"]]
    merged = paired.merge(pm, on=["video_id_1p", "side_1p", "t_sec_1p"], how="left")
    n_matched = int(merged["rel_phase_1p"].notna().sum())
    print(f"[merge_phase] マッチ行数: {n_matched} / {len(merged)}")
    return merged


def _oof_auc_for_mask(
    paired: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> float:
    """指定 mask 部分集合で GroupKFold OOF AUC を計算する。"""
    Xp, yp, gp = X[mask], y[mask], groups[mask]
    n_unique = len(np.unique(gp))
    folds = min(miw.N_FOLDS, max(2, n_unique))
    if len(Xp) < 20 or len(np.unique(yp)) < 2:
        print(f"    [{label}] データ不足 (n={len(Xp)}) -> nan")
        return float("nan")
    oof_proba, _ = miw.run_oof_classifier(Xp, yp, gp, folds)
    valid = ~np.isnan(oof_proba[:, 0])
    auc = (
        float(roc_auc_score(yp[valid], oof_proba[valid, 1]))
        if len(np.unique(yp[valid])) > 1 else float("nan")
    )
    print(f"    [{label}] n={int(valid.sum())}  OOF AUC={auc:.4f}")
    return auc


def evaluate(paired: pd.DataFrame, exclude_contaminated: bool) -> dict[str, float]:
    """相対位相ベースで 全体/序盤/中盤/終盤 OOF AUC を計算する。"""
    label_tag = "汚染除外" if exclude_contaminated else "込み"
    print(f"\n=== 相対位相評価 ({label_tag}) ===")

    base = paired[paired["rel_phase_1p"].notna()].copy()
    if exclude_contaminated:
        base = base[base["seg_max_1p"] <= CONTAMINATION_SEG_MAX].copy()
    print(f"  対象行数: {len(base)} / {len(paired)}")

    y = base["won_1p"].astype(int).values
    groups = base["video_id_1p"].values
    # rel_phase/seg_max はフィルタ条件そのものであり特徴量に含めるとリーク
    # (位相境界情報を直接学習してしまう) ので indicator_cols から除外する。
    _PHASE_LEAK_COLS = frozenset(["seg_max", "rel_phase"])
    indicator_cols = [
        c for c in miw._get_indicator_cols(base) if c not in _PHASE_LEAK_COLS
    ]
    feat_df = miw.build_features(base, indicator_cols)
    X = feat_df.fillna(0.0).values.astype(float)

    rel = base["rel_phase_1p"].astype(float).values
    masks = {
        "全体": np.ones(len(base), dtype=bool),
        "序盤": rel < REL_EARLY,
        "中盤": (rel >= REL_EARLY) & (rel <= REL_LATE),
        "終盤": rel > REL_LATE,
    }
    result: dict[str, float] = {}
    for name, mask in masks.items():
        result[name] = _oof_auc_for_mask(base, X, y, groups, mask, name)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. phase_map 構築 (study CSV からセグメント検出) ===")
    phase_map = build_phase_map(STUDY_DIR)
    phase_map.to_csv(OUT_DIR / "relphase_phase_map_new.csv", index=False)

    print("\n=== 2. labeled_win.csv 読み込み・ペアリング ===")
    df = miw.load_labeled_csv(LABELED_WIN_CSV)
    paired = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)
    paired = merge_phase(paired, phase_map)

    results: dict[str, dict[str, float]] = {}
    results["relphase_all"] = evaluate(paired, exclude_contaminated=False)
    results["relphase_clean"] = evaluate(paired, exclude_contaminated=True)

    print("\n" + "=" * 70)
    print("  相対位相 win-AUC 結果 (新データ)")
    print("=" * 70)
    print(f"  {'条件':<20}  {'全体':>8}  {'序盤':>8}  {'中盤':>8}  {'終盤':>8}")
    for cond, res in results.items():
        print(
            f"  {cond:<20}  "
            + "  ".join(f"{res.get(ph, float('nan')):>8.4f}" for ph in ["全体", "序盤", "中盤", "終盤"])
        )

    rows = []
    for cond, res in results.items():
        row = {"condition": cond}
        row.update({f"auc_{k}": v for k, v in res.items()})
        rows.append(row)
    out_csv = OUT_DIR / "relphase_auc_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")


if __name__ == "__main__":
    main()
