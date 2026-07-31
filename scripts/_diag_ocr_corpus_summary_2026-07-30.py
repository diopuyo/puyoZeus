"""連鎖数OCR全イベント検証corpus (23動画・472行) の全体集計 (2026-07-30)。

背景: data/verify/chain_count_ocr_full_corpus_2026-07-29.csv が完走した。
途中集計で「screen_chain_count と new_chain_count(simulate) の一致率8.3%」という
低い数字が出ていたが、これは両者とも信頼できない推定値どうしの一致でしかない
(2026-07-30 実フレーム目視9事例、simulateは9事例全敗、うち真値が大きいほど
過小評価が拡大: 真値8→simulate1, 真値9→2, 真値10→4)。

本スクリプトは「一致率が低い=どちらが悪いか」を断定せず、層別(連鎖数帯・
動画・1P/2P・delta_score規模)と食い違いの方向・大きさ(中央値/p90/最大)、
欠測率を出すことに専念する。一致率の解釈は実フレーム目視の前提を必ず併記
すること (feedback_stratify_before_pooling_2026-07-29)。

使用データ: data/verify/chain_count_ocr_full_corpus_2026-07-29.csv (読み取り専用)。
実行方法: WSL経由、nice -n 19、単一プロセス (集計のみで軽量)。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
CORPUS_CSV: Path = PROJ_ROOT / "data" / "verify" / "chain_count_ocr_full_corpus_2026-07-29.csv"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "ocr_corpus_summary_2026-07-30"

# 連鎖数帯 (new_chain_count=simulate側基準、screen側基準の両方で集計する)。
CHAIN_BANDS: list[tuple[int, int, str]] = [
    (1, 1, "1連鎖"),
    (2, 2, "2連鎖"),
    (3, 4, "3-4連鎖"),
    (5, 6, "5-6連鎖"),
    (7, 999, "7連鎖以上"),
]

# delta_score規模帯 (得点の絶対値、催促のような小得点イベントと大連鎖を分離)。
SCORE_BANDS: list[tuple[int, int, str]] = [
    (0, 2000, "小(<2000)"),
    (2000, 10000, "中(2000-10000)"),
    (10000, 30000, "大(10000-30000)"),
    (30000, 10**9, "特大(30000+)"),
]


def _band_label(value: float, bands: list[tuple[int, int, str]]) -> str:
    for lo, hi, label in bands:
        if lo <= value <= hi:
            return label
    return "帯域外"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _match_rate_report(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """axis列で層別した一致率・欠測率・件数を集計する。"""
    rows = []
    for key, sub in df.groupby(axis, dropna=False):
        n_total = len(sub)
        n_missing = int(sub["screen_chain_count"].isna().sum())
        avail = sub[sub["screen_chain_count"].notna()]
        n_avail = len(avail)
        n_match = int((avail["screen_chain_count"] == avail["new_chain_count"]).sum())
        match_rate = (n_match / n_avail) if n_avail > 0 else float("nan")
        rows.append({
            axis: key, "n_total": n_total, "n_missing": n_missing,
            "missing_rate": n_missing / n_total if n_total else float("nan"),
            "n_avail": n_avail, "n_match": n_match, "match_rate_among_avail": match_rate,
        })
    out = pd.DataFrame(rows).sort_values("n_total", ascending=False)
    return out


def _gap_report(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """axis列で層別したgap(screen-new)の方向・大きさ分布を集計する。"""
    avail = df[df["screen_chain_count"].notna()].copy()
    avail["gap"] = avail["screen_chain_count"] - avail["new_chain_count"]
    rows = []
    for key, sub in avail.groupby(axis, dropna=False):
        gap = sub["gap"]
        rows.append({
            axis: key, "n": len(sub),
            "screen大きい率": (gap > 0).mean(),
            "一致率": (gap == 0).mean(),
            "simulate大きい率": (gap < 0).mean(),
            "gap_中央値": gap.median(),
            "gap_p90(絶対値)": gap.abs().quantile(0.9),
            "gap_最大(絶対値)": gap.abs().max(),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CORPUS_CSV)
    n_total = len(df)
    n_videos = df["video_stem"].nunique()
    print(f"[準備] corpus行数={n_total}, 動画数={n_videos}")
    print(f"[準備] 動画一覧: {sorted(df['video_stem'].unique().tolist())}")

    df = df.copy()
    df["chain_band_simulate"] = df["new_chain_count"].apply(lambda v: _band_label(v, CHAIN_BANDS))
    df["chain_band_screen"] = df["screen_chain_count"].apply(
        lambda v: _band_label(v, CHAIN_BANDS) if pd.notna(v) else "欠測"
    )
    df["score_band"] = df["delta_score"].apply(lambda v: _band_label(v, SCORE_BANDS))

    # --- 全体 ---
    _print_header("全体")
    n_missing = int(df["screen_chain_count"].isna().sum())
    avail = df[df["screen_chain_count"].notna()]
    n_match = int((avail["screen_chain_count"] == avail["new_chain_count"]).sum())
    print(f"総イベント数={n_total}")
    print(f"OCR欠測(screen_chain_count読めず)={n_missing}件 ({100*n_missing/n_total:.1f}%)")
    print(f"欠測を除いた母数(両方値あり)={len(avail)}件")
    print(f"一致数(screen==simulate)={n_match}件"
          f" ({100*n_match/len(avail):.1f}% / 欠測含む全体比{100*n_match/n_total:.1f}%)")
    gap_all = avail["screen_chain_count"] - avail["new_chain_count"]
    print(f"gap(screen-new) 中央値={gap_all.median():.2f} 平均={gap_all.mean():.2f}"
          f" p90(絶対値)={gap_all.abs().quantile(0.9):.2f} 最大(絶対値)={gap_all.abs().max():.2f}")
    print(f"  screenが大きい(過大)={100*(gap_all > 0).mean():.1f}%"
          f" / 一致={100*(gap_all == 0).mean():.1f}%"
          f" / simulateが大きい(screen過小)={100*(gap_all < 0).mean():.1f}%")

    # --- 層別: 連鎖数帯(simulate基準) ---
    _print_header("層別: 連鎖数帯 (simulate=new_chain_count基準)")
    rep = _match_rate_report(df, "chain_band_simulate")
    print(rep.to_string(index=False))
    gaprep = _gap_report(df, "chain_band_simulate")
    print(gaprep.to_string(index=False))

    # --- 層別: 連鎖数帯(screen基準) ---
    _print_header("層別: 連鎖数帯 (screen OCR基準、欠測は別カテゴリ)")
    rep2 = _match_rate_report(df, "chain_band_screen")
    print(rep2.to_string(index=False))

    # --- 層別: 動画 ---
    _print_header("層別: 動画 (video_stem)")
    rep3 = _match_rate_report(df, "video_stem")
    print(rep3.to_string(index=False))

    # --- 層別: 1P/2P ---
    _print_header("層別: side (1P/2P)")
    rep4 = _match_rate_report(df, "side")
    print(rep4.to_string(index=False))
    gaprep4 = _gap_report(df, "side")
    print(gaprep4.to_string(index=False))

    # --- 層別: delta_score規模 ---
    _print_header("層別: delta_score規模")
    rep5 = _match_rate_report(df, "score_band")
    print(rep5.to_string(index=False))
    gaprep5 = _gap_report(df, "score_band")
    print(gaprep5.to_string(index=False))

    # --- 2桁連鎖(10連鎖以上)の専用集計 (既知のテンプレ欠如が疑われる帯域) ---
    _print_header("2桁連鎖(simulate>=10)の個票 (テンプレ欠如=digit_5-9のみの疑い検証)")
    two_digit = df[df["new_chain_count"] >= 10][
        ["video_stem", "side", "game_idx", "new_chain_count", "screen_chain_count", "delta_score"]
    ]
    print(two_digit.to_string(index=False))

    # 保存
    rep.to_csv(OUT_DIR / "match_rate_by_chain_band_simulate.csv", index=False)
    rep2.to_csv(OUT_DIR / "match_rate_by_chain_band_screen.csv", index=False)
    rep3.to_csv(OUT_DIR / "match_rate_by_video.csv", index=False)
    rep4.to_csv(OUT_DIR / "match_rate_by_side.csv", index=False)
    rep5.to_csv(OUT_DIR / "match_rate_by_score_band.csv", index=False)
    gaprep.to_csv(OUT_DIR / "gap_by_chain_band_simulate.csv", index=False)
    gaprep4.to_csv(OUT_DIR / "gap_by_side.csv", index=False)
    gaprep5.to_csv(OUT_DIR / "gap_by_score_band.csv", index=False)
    two_digit.to_csv(OUT_DIR / "two_digit_chain_events.csv", index=False)
    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
