"""軽量診断: 66動画 (combined66) の解像度・fps・色調特性と学習/認識の
難易度 (動画別AUC・得点整合性不整合率) との相関を調べる (2026-07-29)。

user質問「動画によって色調や解像度など違うと思うけど学習した中で厳しいものは
あるか」への回答用。読み取り専用・軽量集計のみ:
  - cv2.VideoCapture のプロパティ読み (解像度・fps) のみ、フレームデコードなし。
  - AUC は既存ログ (combined66_video_breakdown.log) をパースするだけ。
  - 得点整合性チェックは既存 npz キャッシュ (boards_lean_fixed) 上で
    ChainSimulator.simulate() を回すのみ (動画I/O・認識の再実行なし、
    scripts/_verify_score_consistency_2026-07-29.py と同じ軽量処理)。
  - HSV注入/baseline-reset/undershoot 件数は既存ログファイルの grep 集計のみ。

新規スクリプト、既存ファイルは無変更。

使い方:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_video_difficulty_2026-07-29.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.scoring import calculate_chain_score, is_score_consistent, score_consistency_ratio  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    NPZ_DIR, SCORE_MISSING_SENTINEL, TIER_MAP, FireEvent, _load_npz,
    _process_video, _subset,
)

FRAMES_DIR: Path = PROJ_ROOT / "data" / "frames"
BREAKDOWN_LOG: Path = (
    PROJ_ROOT / "data" / "verify" / "win_eval_combined66_2026-07-29"
    / "combined66_video_breakdown.log"
)
LOGS_DIR: Path = PROJ_ROOT / "logs"
OUT_CSV: Path = PROJ_ROOT / "data" / "verify" / "video_difficulty_2026-07-29.csv"

# combined66 で使われた66動画 (combined66_run.log の video数=66 一覧と同一)
VIDEO_STEMS: tuple[str, ...] = (
    "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19",
    "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c28", "c29",
    "c34", "c35", "c36", "c37", "c40", "c41", "c42", "c43", "c44", "c45",
    "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55",
    "c56", "c57", "c58", "c59", "c60", "c61", "c62", "c63", "c64", "c65",
    "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75",
    "c76", "c77", "c78", "c79", "c80", "c81",
)

# ログのファイル名内でstemを誤マッチさせない (c1がc10にマッチしない等) ための境界付き正規表現
_STEM_BOUNDARY_TMPL = r"(?:^|[_.\-]){stem}(?:[_.\-]|$)"

# grep集計するログ内パターン (代理指標)
LOG_PATTERNS: dict[str, str] = {
    "hsv_injected_lines": r"online_hsv injected",
    "baseline_reset_lines": r"\[baseline-reset\]",
    "chain_tsumo_undershoot_lines": r"\[chain-tsumo-undershoot\]",
}


def _read_video_props(stem: str) -> dict:
    """cv2.VideoCapture のプロパティのみ読む (フレームデコードなし)。"""
    path = FRAMES_DIR / f"video_{stem}.mp4"
    if not path.exists():
        return {
            "video_stem": stem, "file_exists": False,
            "width": None, "height": None, "fps": None, "frame_count": None,
        }
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return {
        "video_stem": stem, "file_exists": True,
        "width": width, "height": height, "fps": round(fps, 2),
        "frame_count": frame_count,
    }


def _fps_bucket(fps: float | None) -> str:
    """fps実測値を30/60の名目バケットに丸める(24-45→30、45-90→60、他はそのまま)。"""
    if fps is None:
        return "不明"
    if 24.0 <= fps <= 45.0:
        return "30fps系"
    if 45.0 < fps <= 90.0:
        return "60fps系"
    return f"{fps:.0f}fps系(異例)"


def _parse_breakdown_log() -> pd.DataFrame:
    """combined66_video_breakdown.log をパースし video×scope の AUC 表を作る。"""
    text = BREAKDOWN_LOG.read_text(encoding="utf-8")
    scope_blocks = re.split(r"={10,}\n\s*スコープ:\s*(\S+)\s+", text)[1:]
    rows: list[dict] = []
    for scope, body in zip(scope_blocks[0::2], scope_blocks[1::2]):
        for line in body.splitlines():
            m = re.match(
                r"^video_(c\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                line.strip(),
            )
            if m:
                rows.append({
                    "video_stem": m.group(1), "scope": scope,
                    "n": int(m.group(2)), "auc": float(m.group(8)),
                })
    return pd.DataFrame(rows)


def _check_event(sim: ChainSimulator, npz_path: Path, ev: FireEvent) -> dict | None:
    """1 FireEvent の得点整合性チェック (_verify_score_consistency_2026-07-29.py と同一ロジック)。"""
    if ev.delta_score == SCORE_MISSING_SENTINEL:
        return None
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if ev.fire_side not in by_side:
        return None
    rec = by_side[ev.fire_side]
    mask = rec.game_idx == ev.game_idx
    g = _subset(rec, mask)
    if ev.before_idx < 0 or ev.before_idx >= len(g.t_sec):
        return None
    before = Board.from_list(g.grids[ev.before_idx].tolist())
    result = sim.simulate(before)
    expected = calculate_chain_score(result).total_score
    ratio = score_consistency_ratio(expected, ev.delta_score)
    consistent = is_score_consistent(expected, ev.delta_score)
    return {"ratio": ratio, "consistent": consistent}


def _score_consistency_per_video() -> pd.DataFrame:
    """66動画それぞれの得点整合性チェック不整合率を計算する。"""
    sim = ChainSimulator()
    rows: list[dict] = []
    for stem in VIDEO_STEMS:
        npz_path = NPZ_DIR / f"{stem}.npz"
        if not npz_path.exists():
            rows.append({
                "video_stem": stem, "n_score_events": 0,
                "n_inconsistent": 0, "inconsistent_ratio": np.nan, "note": "npz不在",
            })
            continue
        _, defrag, _ = _process_video(npz_path, sim, 0)
        n_total = 0
        n_bad = 0
        for ev in defrag:
            r = _check_event(sim, npz_path, ev)
            if r is None:
                continue
            n_total += 1
            if not r["consistent"]:
                n_bad += 1
        rows.append({
            "video_stem": stem, "n_score_events": n_total,
            "n_inconsistent": n_bad,
            "inconsistent_ratio": (n_bad / n_total) if n_total else np.nan,
            "note": "",
        })
    return pd.DataFrame(rows)


def _log_counts_per_video() -> pd.DataFrame:
    """logs/ 配下のファイル名でstemに一致するものを対象に、代理指標パターンの出現行数を集計する。"""
    all_log_names = [p.name for p in LOGS_DIR.glob("*.log")]
    rows: list[dict] = []
    for stem in VIDEO_STEMS:
        boundary = re.compile(_STEM_BOUNDARY_TMPL.format(stem=re.escape(stem)))
        matched_files = [n for n in all_log_names if boundary.search(n)]
        counts = {k: 0 for k in LOG_PATTERNS}
        n_files_read = 0
        for fname in matched_files:
            try:
                text = (LOGS_DIR / fname).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            n_files_read += 1
            for key, pat in LOG_PATTERNS.items():
                counts[key] += len(re.findall(pat, text))
        row = {"video_stem": stem, "n_matched_log_files": n_files_read}
        row.update(counts)
        rows.append(row)
    return pd.DataFrame(rows)


def _stratified_report(df: pd.DataFrame, group_col: str, value_col: str, label: str) -> None:
    """プールせず層別に中央値・平均・std・件数を表示する (feedback_stratify_before_pooling_2026-07-29 準拠)。"""
    print(f"\n[{label}] {group_col} 別 {value_col} 分布 (層別、プールなし)")
    g = df.dropna(subset=[value_col]).groupby(group_col)[value_col]
    stat = g.agg(["count", "mean", "median", "std", "min", "max"])
    print(stat.to_string())


def main() -> None:
    print("=== (1) 解像度・fps 実測 (cv2プロパティ読みのみ) ===")
    props_rows = [_read_video_props(stem) for stem in VIDEO_STEMS]
    props_df = pd.DataFrame(props_rows)
    props_df["fps_bucket"] = props_df["fps"].apply(_fps_bucket)
    props_df["resolution"] = props_df.apply(
        lambda r: f"{r['width']}x{r['height']}" if r["file_exists"] else "動画ファイル削除済み",
        axis=1,
    )
    n_missing = int((~props_df["file_exists"]).sum())
    print(f"動画ファイル削除済み: {n_missing}/{len(props_df)}")
    print(props_df[["video_stem", "file_exists", "resolution", "fps", "fps_bucket"]].to_string(index=False))

    print("\n=== (2) AUCログのパース ===")
    auc_df = _parse_breakdown_log()
    auc_wide = auc_df.pivot(index="video_stem", columns="scope", values="auc").reset_index()
    print(f"AUC取得済み動画数={auc_wide['video_stem'].nunique()}")

    print("\n=== (3) 得点整合性チェック (66動画、npzキャッシュ利用) ===")
    consist_df = _score_consistency_per_video()
    print(consist_df.to_string(index=False))
    total_events = int(consist_df["n_score_events"].sum())
    total_bad = int(consist_df["n_inconsistent"].sum())
    print(f"\n全体: {total_bad}/{total_events} 不整合 ({total_bad / max(1, total_events):.1%})")

    print("\n=== (4) ログ代理指標集計 (HSV注入 / baseline-reset / undershoot) ===")
    log_df = _log_counts_per_video()
    print(log_df.to_string(index=False))
    print(f"\n対象logファイル総数(重複含まず) = {len(list(LOGS_DIR.glob('*.log')))}")

    # マージ
    merged = props_df.merge(auc_wide, on="video_stem", how="left")
    merged = merged.merge(consist_df, on="video_stem", how="left")
    merged = merged.merge(log_df, on="video_stem", how="left")
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {OUT_CSV}")

    print("\n=== (5) fps別 / 解像度別 AUC 層別比較 (プールせず) ===")
    for scope_name in ["全体", "序盤", "中盤", "終盤"]:
        col = scope_name
        if col not in merged.columns:
            continue
        _stratified_report(merged, "fps_bucket", col, f"AUC({scope_name})")
        _stratified_report(merged, "resolution", col, f"AUC({scope_name})")

    print("\n=== (6) AUC(全体) ワースト5動画 ===")
    worst5 = merged.dropna(subset=["全体"]).sort_values("全体").head(5)
    cols = [
        "video_stem", "全体", "序盤", "中盤", "終盤", "resolution", "fps",
        "inconsistent_ratio", "n_score_events", "hsv_injected_lines",
        "baseline_reset_lines", "chain_tsumo_undershoot_lines",
    ]
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].loc[worst5.index].to_string(index=False))

    print("\n=== (7) 相関係数 (Pearson、n注意・層別確認済みの上での参考値) ===")
    for scope_name in ["全体", "序盤", "中盤", "終盤"]:
        if scope_name not in merged.columns:
            continue
        sub = merged.dropna(subset=[scope_name, "inconsistent_ratio"])
        if len(sub) >= 3:
            corr = sub[scope_name].corr(sub["inconsistent_ratio"])
            print(f"  corr(AUC[{scope_name}], 得点不整合率) n={len(sub)}: {corr:.3f}")
        sub2 = merged.dropna(subset=[scope_name, "fps"])
        if len(sub2) >= 3 and sub2["fps"].nunique() > 1:
            corr2 = sub2[scope_name].corr(sub2["fps"])
            print(f"  corr(AUC[{scope_name}], fps実測値) n={len(sub2)}: {corr2:.3f}")
        for hk in LOG_PATTERNS:
            sub3 = merged.dropna(subset=[scope_name, hk])
            if len(sub3) >= 3 and sub3[hk].nunique() > 1:
                corr3 = sub3[scope_name].corr(sub3[hk])
                print(f"  corr(AUC[{scope_name}], {hk}) n={len(sub3)}: {corr3:.3f}")


if __name__ == "__main__":
    main()
