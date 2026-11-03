"""rows_v1 の欠測率・行数・各特徴の分布 (層別: 局面・本線・撃ち合い/静止) を SUMMARY.md に出す。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["k1_ojama_p1", "k5_ojama_p1", "phase_total_max", "icon_corr_total_p1", "sl_pending_p1", "ta_window_p1",
            "ta_H_med_p1", "ta_H_fast_p1", "ta_q_med_p1", "ta_i_pred_p1", "ta_i_icon_p1", "ta_m_pred_med_p1",
            "ta_recv_pred_med_p1", "chain_rem_sec_p1", "ep_ratio_formula_p1", "ep_ratio_board_p1", "land_icon_drop_p1"]
MASK_COLS = ["mk_board_p1", "mk_ind_p1", "mk_k_p1", "mk_queue_p1", "mk_icon_p1", "mk_sl_p1", "mk_land_p1", "mk_chain_p1",
             "mk_ep_f_p1", "mk_ep_b_p1", "mk_pf_p1", "mk_ta_p1", "mk_ta2_p1", "mk_phase"]
STRATA = {"phase": "phase", "mode_exchange": "mode_exchange"}


def _q(x: pd.Series) -> str:
    x = x.dropna()
    return "-" if x.empty else f"{x.mean():.3g} / {x.quantile(.1):.3g} / {x.median():.3g} / {x.quantile(.9):.3g} (n={len(x)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    a = ap.parse_args()
    frames, mans = [], []
    need = ["video_id", "orientation", "weight", "game_idx", "mode_exchange", "phase", "ep_honsen60_formula_p1",
            "ep_honsen60_formula_p2", "ep_chain7plus_p1", "ep_chain7plus_p2", "ta_active_p1", "ta_active_p2",
            "ta_unc72_pred_med_p1", "trigger_mask"] + KEY_COLS + MASK_COLS
    for m in sorted(a.rows.glob("*.manifest.json")):
        if not m.with_suffix("").with_suffix(".done").exists() and not (a.rows / (m.name.split(".")[0] + ".done")).exists():
            continue
        mans.append(json.loads(m.read_text(encoding="utf-8")))
        frames.append(pd.read_parquet(a.rows / f"{m.name.split('.')[0]}.parquet", columns=list(dict.fromkeys(need))))
    df = pd.concat(frames, ignore_index=True)
    out = ["# B3 rows_v1 SUMMARY", "", f"- 動画数: {len(mans)}  /  行数 (原向き): {sum(m['rows_original'] for m in mans)}  /  複製込み: {len(df)}",
           f"- 試合数 (行あり): {sum(len(m['games_with_rows']) for m in mans)}  /  重み合計: {df.weight.sum():.2f} (試合数と一致が期待値)",
           f"- 生成所要 (動画別合計): {sum(m['elapsed_sec'] for m in mans)/3600:.2f} 時間 (単独worker換算)",
           f"- schema: {mans[0]['schema_version']}  列数: {mans[0]['n_columns']}  リーク違反行: {sum(m['frame_info']['leak_violations'] for m in mans)}", "",
           "## 欠測理由コード (mk 列, 原向き行の p1 側 / 母数=全行)", "",
           "| mask列 | 0 OK | 1 旧schema | 2 観測不能 | 3 未利用可 | 4 境界 | 5 片側未観測 | 6 予測不能 | 7 打切り | 8 非適用 | 9 未実装 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in MASK_COLS:
        vc = df[c].value_counts(normalize=True)
        out.append(f"| {c} | " + " | ".join(f"{vc.get(k, 0)*100:.1f}%" for k in range(10)) + " |")
    out += ["", "## 層別の行数・重み (複製込み)", "", "| 層 | 行数 | 重み合計 |", "|---|---|---|"]
    for name, mask in _strata(df).items():
        out.append(f"| {name} | {int(mask.sum())} | {df.weight[mask].sum():.2f} |")
    out += ["", "## 主要特徴の分布 (平均 / p10 / 中央 / p90, 欠測除外)", "", "| 特徴 | 全体 | 撃ち合い(mode=1) | 静止(mode=0) | 序盤 | 中盤 | 終盤 |", "|---|---|---|---|---|---|---|"]
    for c in KEY_COLS:
        cells = [_q(df[c]), _q(df[c][df.mode_exchange == 1]), _q(df[c][df.mode_exchange == 0])] + [_q(df[c][df.phase == p]) for p in (0, 1, 2)]
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out += ["", "## 動画別", "", "| 動画 | 行(原) | 試合数 | 所要s | ep式NaN率 |", "|---|---|---|---|---|"]
    for m in mans:
        out.append(f"| {m['video_id']} | {m['rows_original']} | {len(m['games_with_rows'])} | {m['elapsed_sec']:.0f} | "
                   f"{m['missing']['nan_rate'].get('ep_ratio_formula_p1', float('nan')):.2f} |")
    (a.rows / "SUMMARY.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", a.rows / "SUMMARY.md")


def _strata(df: pd.DataFrame) -> dict[str, pd.Series]:
    hon = df.ep_honsen60_formula_p1
    return {"局面 序盤": df.phase == 0, "局面 中盤": df.phase == 1, "局面 終盤": df.phase == 2,
            "撃ち合い (mode=1)": df.mode_exchange == 1, "静止 (mode=0)": df.mode_exchange == 0,
            "直近発火 本線(式60%以上, p1)": hon == 1, "直近発火 催促/60%未満(式, p1)": hon == 0, "直近発火 不明(式欠測/未発火, p1)": hon.isna(),
            "7連鎖以上完走後 (p1)": df.ep_chain7plus_p1 == 1, "時間軸 a→b 計算あり (p1攻撃)": df.ta_active_p1 == 1,
            "相殺不能72以上 (pred_med, p1攻撃)": df.ta_unc72_pred_med_p1 == 1}


if __name__ == "__main__":
    main()
