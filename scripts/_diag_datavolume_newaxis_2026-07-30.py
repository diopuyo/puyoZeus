# -*- coding: utf-8 -*-
"""データ量(20/40/66本)に対する中盤AUC推移を新軸(ぷよ総量)で再評価する (2026-07-30)。

閾値パラメータ化: winp モジュール (_diag_relphase_by_winpanel_2026-07-30) が
環境変数 PUYO_PHASE_EARLY_MAX / PUYO_PHASE_LATE_MIN から解決した閾値を
そのまま踏襲する(単一ソース)。デフォルトは2026-07-30改定の確定値(18/48)。
出力先 OUT_DIR は閾値の組ごとに自動で分かれるため、過去の20/57結果は
上書きされない。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402

base = importlib.import_module("scripts._tmp_relphase_win_auc_2026-07-26")
winp = importlib.import_module("scripts._diag_relphase_by_winpanel_2026-07-30")

LABELED_WIN_CSV = "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
def _out_dir_for_thresholds(early: float, late: float) -> Path:
    """閾値の組に応じて出力先ディレクトリを分ける(過去結果を上書きしないため)。"""
    if early == 20.0 and late == 57.0:
        return Path("data/verify/datavolume_newaxis_2026-07-30")
    if early == 18.0 and late == 48.0:
        return Path("data/verify/datavolume_newaxis_th18_48_2026-07-30")
    return Path(f"data/verify/datavolume_newaxis_th{early:.0f}_{late:.0f}_2026-07-30")


OUT_DIR = _out_dir_for_thresholds(winp.PUYO_TOTAL_EARLY_MAX, winp.PUYO_TOTAL_LATE_MIN)

M20_VIDEOS_TXT = Path("data/verify/labeled_win_m20_2026-07-28/selected_videos_m20.txt")
C20_VIDEOS_TXT = Path("data/verify/labeled_win_c20_2026-07-26/selected_videos.txt")
M30_VIDEOS_TXT = Path("data/verify/labeled_win_m30_2026-07-28/selected_videos_m30.txt")

M20_OLD_AXIS_CSV = Path("data/verify/win_eval_m20_2026-07-28/relphase_m20/relphase_auc_summary.csv")
COMBINED40_OLD_AXIS_CSV = Path(
    "data/verify/win_eval_m20_2026-07-28/relphase_combined40/relphase_auc_summary.csv")
COMBINED66_OLD_AXIS_CSV = Path(
    "data/verify/win_eval_combined66_2026-07-29/relphase_combined66/relphase_auc_summary.csv")
# winp.OUT_DIR は閾値の組に応じて自動で分かれる(単一ソースの閾値解決を踏襲)。
COMBINED66_NEW_AXIS_CSV = winp.OUT_DIR / "relphase_winpanel_auc_summary.csv"

RESAMPLE_SEED: int = 42
RESAMPLE_SIZES: tuple[int, ...] = (20, 40)
RESAMPLE_REPEATS: int = 10
LEAK_COLS: frozenset[str] = frozenset(
    ["game_abs_idx", "game_duration_sec", "seg_max", "rel_phase"])
SPOT_CHECK_TOL: float = 1e-6


def load_video_list(path: Path) -> list[str]:
    """1行1動画IDのテキストファイルを読み込む。"""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_cohorts() -> dict[str, list[str]]:
    """m20 / combined40(m20+c20) / combined66(m20+c20+m30) の動画リストを返す。"""
    m20 = load_video_list(M20_VIDEOS_TXT)
    c20 = load_video_list(C20_VIDEOS_TXT)
    m30 = load_video_list(M30_VIDEOS_TXT)
    combined40 = m20 + c20
    combined66 = m20 + c20 + m30
    return {"m20": m20, "combined40": combined40, "combined66": combined66}


def check_cohort_nesting(cohorts: dict[str, list[str]]) -> None:
    """m20 が combined40 の部分集合か、combined40 が combined66 の部分集合かを確認・報告する。"""
    s20, s40, s66 = set(cohorts["m20"]), set(cohorts["combined40"]), set(cohorts["combined66"])
    nested_20_40 = s20.issubset(s40)
    nested_40_66 = s40.issubset(s66)
    print(f"[入れ子確認] m20の部分集合か: {nested_20_40}  combined40の部分集合か: {nested_40_66}")
    print(f"[コホート内訳] m20={len(s20)}本 combined40={len(s40)}本(追加分{len(s40 - s20)}本)"
          f" combined66={len(s66)}本(追加分{len(s66 - s40)}本)")
    print("  補足: 追加分はティア構成が異なる(c20はチャレンジャー中心、m30はマスター追加)ため"
          "動画数増加の効果とティア構成変化の効果が交絡している。")


def load_and_pair_full() -> pd.DataFrame:
    """combined66 labeled_win CSV を読み込み1P/2Pペアリングし、旧cut(手数リセット)の
    seg_max_1p/rel_phase_1p を結合したデータフレームを返す(66動画全件)。"""
    df = miw.load_labeled_csv(LABELED_WIN_CSV)
    paired = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)
    merged = winp.load_old_cut_seg_max(paired)
    return merged


def build_old_matched_with_puyo_phase(merged_full: pd.DataFrame) -> pd.DataFrame:
    """66動画全件から旧cutにマッチした行のみ残し、新軸(ぷよ総量)位相を付与する。

    _puyo_phase は行単位(board_puyo_total_raw)の純関数なので、後で
    コホート別に部分集合を取り出しても値は変わらない(計算は1回のみで済む)。
    """
    old_matched = merged_full[merged_full["rel_phase_1p"].notna()].copy()
    old_matched["_puyo_phase"] = winp.build_puyo_total_phase(old_matched)
    return old_matched


def report_population_identity(cohort_name: str, cohort_all: pd.DataFrame) -> dict[str, int]:
    """コホートの母集団サイズ(全件/clean)を報告し、旧軸・新軸で共通母集団を使うことの根拠とする。"""
    n_total = int(len(cohort_all))
    n_matched = int(cohort_all["rel_phase_1p"].notna().sum())
    n_clean = int((cohort_all["rel_phase_1p"].notna()
                   & (cohort_all["seg_max_1p"] <= winp.CONTAMINATION_SEG_MAX)).sum())
    match_rate = n_matched / n_total if n_total else float("nan")
    clean_rate = n_clean / n_matched if n_matched else float("nan")
    print(f"[母集団 {cohort_name}] paired全件={n_total} 旧cutマッチ={n_matched}"
          f"({match_rate:.1%}) clean(seg_max<=60)={n_clean}({clean_rate:.1%} of matched)")
    return {"n_total": n_total, "n_matched": n_matched, "n_clean": n_clean}


def compute_old_axis_for_cohort(cohort_all: pd.DataFrame) -> dict[str, dict[str, float]]:
    """base.evaluate (手数相対進行率、既存コード一切変更なし) でコホートの旧軸AUCを計算する。"""
    res_all = base.evaluate(cohort_all, exclude_contaminated=False)
    res_clean = base.evaluate(cohort_all, exclude_contaminated=True)
    return {"all": res_all, "clean": res_clean}


def compute_new_axis_for_cohort(
    old_matched_full: pd.DataFrame, video_ids: list[str], label: str,
) -> dict[str, dict[str, float]]:
    """old_matched_full(66動画・旧cutマッチ済・_puyo_phase付与済)からコホート分を
    切り出し、新軸(ぷよ総量)AUCを winp.evaluate_by_phase で計算する。"""
    cohort_matched = old_matched_full[old_matched_full["video_id_1p"].isin(video_ids)].copy()
    cohort_clean = cohort_matched[cohort_matched["seg_max_1p"] <= winp.CONTAMINATION_SEG_MAX].copy()
    res_all = winp.evaluate_by_phase(cohort_matched, cohort_matched["_puyo_phase"], label + "_new_all")
    res_clean = winp.evaluate_by_phase(cohort_clean, cohort_clean["_puyo_phase"], label + "_new_clean")
    return {"all": res_all, "clean": res_clean}


def load_existing_relphase_csv(path: Path) -> dict[str, dict[str, float]]:
    """既存の relphase_auc_summary.csv (旧軸フォーマット) を読み込む。"""
    df = pd.read_csv(path)
    phases = ["全体", "序盤", "中盤", "終盤"]

    def _row(cond: str) -> dict[str, float]:
        r = df[df["condition"] == cond].iloc[0]
        return {p: float(r["auc_" + p]) for p in phases}

    return {"all": _row("relphase_all"), "clean": _row("relphase_clean")}


def load_existing_winpanel_new_axis(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """既存の relphase_winpanel_auc_summary.csv から combined66 の新軸(D変種)行を読み込む。"""
    df = pd.read_csv(path)
    phases = ["全体", "序盤", "中盤", "終盤"]

    def _row(cond: str) -> dict[str, float]:
        r = df[df["condition"] == cond].iloc[0]
        return {p: float(r["auc_" + p]) for p in phases}

    def _n(cond: str) -> int:
        return int(df[df["condition"] == cond].iloc[0]["n_rows"])

    result = {"all": _row("D_oldcut_puyoaxis_all"), "clean": _row("D_oldcut_puyoaxis_clean")}
    counts = {"n_all": _n("D_oldcut_puyoaxis_all"), "n_clean": _n("D_oldcut_puyoaxis_clean")}
    return result, counts


def spot_check_m20_old_axis(fresh: dict[str, dict[str, float]]) -> bool:
    """m20旧軸の自前再計算と既存ファイルの値が一致するか確認する(無検証引用回避)。"""
    existing = load_existing_relphase_csv(M20_OLD_AXIS_CSV)
    ok = True
    for cond in ("all", "clean"):
        for phase, v_fresh in fresh[cond].items():
            v_old = existing[cond][phase]
            diff = abs(v_fresh - v_old)
            both_nan = np.isnan(v_fresh) and np.isnan(v_old)
            if diff > SPOT_CHECK_TOL and not both_nan:
                ok = False
                print("  [スポットチェック不一致] " + cond + "/" + phase
                      + " fresh=" + str(v_fresh) + " existing=" + str(v_old))
    print("[スポットチェック結果] m20旧軸 自前再計算 vs 既存ファイル: " + ("一致" if ok else "不一致"))
    return ok


def _mid_auc_only(df_subset: pd.DataFrame, label: str) -> float:
    """evaluate_by_phase の中盤マスクのみを計算する軽量版(リサンプリング高速化用)。

    全体/序盤/終盤を計算しないことで1回あたりの学習コストを約1/4にする。
    """
    indicator_cols = [c for c in miw._get_indicator_cols(df_subset) if c not in LEAK_COLS]
    feat_df = miw.build_features(df_subset, indicator_cols)
    X = feat_df.fillna(0.0).values.astype(float)
    y = df_subset["won_1p"].astype(int).values
    groups = df_subset["video_id_1p"].values
    mask = (df_subset["_puyo_phase"] == "中盤").values
    return winp.base._oof_auc_for_mask(df_subset, X, y, groups, mask, label)


def resample_midphase_auc(
    old_matched_full: pd.DataFrame, pool_videos: list[str],
    sample_size: int, n_repeats: int, seed: int,
) -> pd.DataFrame:
    """combined66プールからランダムに sample_size 本を複数回抽出し、
    新軸・全件条件の中盤AUCのみを計算する(動画数の効果とティア構成の効果を分離)。"""
    rng = np.random.default_rng(seed)
    pool_arr = np.array(pool_videos)
    rows: list[dict] = []
    for rep in range(n_repeats):
        chosen = rng.choice(pool_arr, size=sample_size, replace=False)
        sub = old_matched_full[old_matched_full["video_id_1p"].isin(chosen)].copy()
        auc = _mid_auc_only(sub, "resample n=" + str(sample_size) + " rep=" + str(rep))
        rows.append({
            "sample_size": sample_size, "repeat": rep, "auc_mid": auc,
            "n_rows": len(sub), "videos": ",".join(sorted(chosen)),
        })
        print("  [resample n=" + str(sample_size) + " rep=" + str(rep + 1) + "/" + str(n_repeats)
              + "] auc_mid=" + str(auc) + " n_rows=" + str(len(sub)))
    return pd.DataFrame(rows)


def _flatten_row(cohort: str, axis: str, condition: str, res: dict[str, float],
                  n_rows: int | None, source: str) -> dict:
    """比較表1行分を辞書化する。"""
    row = {"cohort": cohort, "axis": axis, "condition": condition,
           "n_rows": n_rows, "source": source}
    for phase in ("全体", "序盤", "中盤", "終盤"):
        row["auc_" + phase] = res.get(phase, float("nan"))
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quick = "--quick" in sys.argv

    print("=== 1. labeled_win.csv 読み込み・ペアリング・旧cut seg_max 結合 ===")
    merged_full = load_and_pair_full()
    old_matched_full = build_old_matched_with_puyo_phase(merged_full)

    cohorts = build_cohorts()
    if quick:
        cohorts = {k: v[:2] for k, v in cohorts.items()}
        print("[--quick] 各コホートを先頭2本に縮小(配線検証用)")
    check_cohort_nesting(cohorts)

    rows_out: list[dict] = []
    pop_report: dict[str, dict[str, int]] = {}

    print("\n=== 2. m20 旧軸スポットチェック(自前再計算 vs 既存ファイル) ===")
    m20_all_df = merged_full[merged_full["video_id_1p"].isin(cohorts["m20"])].copy()
    pop_report["m20"] = report_population_identity("m20", m20_all_df)
    m20_old_fresh = compute_old_axis_for_cohort(m20_all_df)
    spot_ok = spot_check_m20_old_axis(m20_old_fresh) if not quick else None

    print("\n=== 3. m20 新軸(本タスクの新規計算) ===")
    m20_new = compute_new_axis_for_cohort(old_matched_full, cohorts["m20"], "m20")

    print("\n=== 4. combined40 旧軸(既存ファイル読込)・新軸(新規計算) ===")
    c40_all_df = merged_full[merged_full["video_id_1p"].isin(cohorts["combined40"])].copy()
    pop_report["combined40"] = report_population_identity("combined40", c40_all_df)
    if quick:
        combined40_old = compute_old_axis_for_cohort(c40_all_df)
    else:
        combined40_old = load_existing_relphase_csv(COMBINED40_OLD_AXIS_CSV)
    combined40_new = compute_new_axis_for_cohort(old_matched_full, cohorts["combined40"], "combined40")

    print("\n=== 5. combined66 旧軸・新軸(既存ファイル読込) ===")
    c66_all_df = merged_full[merged_full["video_id_1p"].isin(cohorts["combined66"])].copy()
    pop_report["combined66"] = report_population_identity("combined66", c66_all_df)
    if quick:
        combined66_old = compute_old_axis_for_cohort(c66_all_df)
        combined66_new = compute_new_axis_for_cohort(old_matched_full, cohorts["combined66"], "combined66")
    else:
        combined66_old = load_existing_relphase_csv(COMBINED66_OLD_AXIS_CSV)
        combined66_new, _ = load_existing_winpanel_new_axis(COMBINED66_NEW_AXIS_CSV)

    reused40 = "reused(既存ファイル,同一コードで生成済み)"
    reused66 = "reused(既存ファイル,task提示値と一致確認済み)"
    table_defs = [
        ("m20", "旧軸(手数相対)", m20_old_fresh, "computed(spot-check)"),
        ("m20", "新軸(ぷよ総量)", m20_new, "computed"),
        ("combined40", "旧軸(手数相対)", combined40_old, "computed" if quick else reused40),
        ("combined40", "新軸(ぷよ総量)", combined40_new, "computed"),
        ("combined66", "旧軸(手数相対)", combined66_old, "computed" if quick else reused66),
        ("combined66", "新軸(ぷよ総量)", combined66_new, "computed" if quick else reused66),
    ]
    for cohort, axis, res_dict, source in table_defs:
        for cond in ("all", "clean"):
            n_key = "n_matched" if cond == "all" else "n_clean"
            rows_out.append(_flatten_row(cohort, axis, cond, res_dict[cond],
                                          pop_report[cohort][n_key], source))

    summary_df = pd.DataFrame(rows_out)
    summary_csv = OUT_DIR / "datavolume_newaxis_auc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n[save] " + str(summary_csv))
    print(summary_df.to_string(index=False))

    print("\n=== 6. コホート内リサンプリング(動画数 vs ティア構成の交絡分離) ===")
    n_repeats = 2 if quick else RESAMPLE_REPEATS
    pool_size = len(cohorts["combined66"])
    sizes = [min(s, pool_size) for s in RESAMPLE_SIZES] if quick else list(RESAMPLE_SIZES)
    resample_frames = [
        resample_midphase_auc(old_matched_full, cohorts["combined66"], size, n_repeats, RESAMPLE_SEED)
        for size in sizes
    ]
    resample_df = pd.concat(resample_frames, ignore_index=True)
    resample_csv = OUT_DIR / "resample_midphase_auc.csv"
    resample_df.to_csv(resample_csv, index=False)
    print("[save] " + str(resample_csv))

    resample_summary = resample_df.groupby("sample_size")["auc_mid"].agg(
        ["mean", "std", "count"]).reset_index()
    resample_summary_csv = OUT_DIR / "resample_midphase_summary.csv"
    resample_summary.to_csv(resample_summary_csv, index=False)
    print(resample_summary.to_string(index=False))
    print("[save] " + str(resample_summary_csv))

    print("\n=== 完了 ===")
    check_str = "未実施(--quick)" if quick else ("一致" if spot_ok else "不一致(要確認)")
    print("m20旧軸スポットチェック: " + check_str)


if __name__ == "__main__":
    main()
