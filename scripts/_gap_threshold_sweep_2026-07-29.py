"""#24 Step0 ゲート判定: OPP_GAP_THRESHOLD_SEC 感度スイープ (2026-07-29)。

背景 (確定事実、userタスク明記):
    被覆率 (opp_available率) は OPP_GAP_THRESHOLD_SEC=2.0 で 6.7% (34/506)。
    ゲート基準 20% に届かない。内訳の gap_in_window 189件 (37.4%) が
    「閾値を緩めれば救えるのか、それとも桁違いで無駄なのか」を実測するのが
    本スクリプトの目的。

⚠️ 認識の再実行は一切行わない。既存 npz (data/indicators_v2/
   boards_lean_fixed_regen_2026-07-28/、m30収集とは無関係の完了済みデータ)
   と既存 CSV (exchange_landing_delay_regen_2026-07-28.csv) の読み取り集計
   のみで完結する。scripts/measure_ojama_landing_delay.py・
   scripts/measure_exchange_dynamics.py は import のみで一切変更しない
   (両ファイルの既存方針を継承)。

_find_landing() (scripts/measure_ojama_landing_delay.py:103-139) は
gap_threshold_sec を既に引数として受け取る設計 (関数シグネチャ変更不要)。
本スクリプトは複数の閾値でこれを呼び分けるだけの「再測定スクリプト側の
コピー実装」(userタスク許可範囲) として、ギャップ実測用の付随関数のみを
新規に追加する。

使い方:
    nice -n 19 venv/bin/python -m scripts._gap_threshold_sweep_2026-07-29
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    TIER_MAP, FireEvent, _process_video, _subset,
)
from scripts.measure_ojama_landing_delay import (  # noqa: E402
    MAX_LANDING_SEARCH_SEC, _find_landing, _load_npz, _visible_ojama_counts,
)

# 再収集済み npz (m30収集とは別プロセス、完了済み・変更なし)
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"

# スイープ対象の閾値 (userタスク指定、45.0 = 探索窓と同値 = 実質無制限)
GAP_THRESHOLDS_SEC: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0, 45.0)

# ゲート基準 (userタスク・memory project_exchange_meter_design_b 準拠)
COVERAGE_GATE_PCT: float = 20.0

# ベースライン (現状確定済み、gap分布抽出の対象を特定するための閾値)
BASELINE_GAP_THRESHOLD_SEC: float = 2.0


def _first_violating_gap(
    t_sec: np.ndarray, t_fire: float, gap_threshold_sec: float, search_max_sec: float,
) -> float:
    """gap_in_window 判定の原因になった実際のギャップ長 (秒) を返す。

    _find_landing (measure_ojama_landing_delay.py:103) と同じ走査順序で、
    「連続観測間隔が gap_threshold_sec を初めて超えた地点」の間隔そのものを
    返す (該当なしなら NaN)。判定ロジック自体は変更せず、原因の実測のみ
    行う付随関数 (再測定スクリプト側のコピー実装、既存関数は不変更)。
    """
    idx_before = int(np.searchsorted(t_sec, t_fire, side="right")) - 1
    if idx_before < 0:
        return float("nan")
    window_end = t_fire + search_max_sec
    n = len(t_sec)
    prev_t = float(t_sec[idx_before])
    i = idx_before + 1
    while i < n and float(t_sec[i]) <= window_end:
        cur_t = float(t_sec[i])
        gap = cur_t - prev_t
        if gap > gap_threshold_sec:
            return gap
        prev_t = cur_t
        i += 1
    return float("nan")


def _sweep_one_video(
    npz_path: Path, events: list[FireEvent], gap_thresholds: tuple[float, ...],
) -> list[dict]:
    """1動画分、発火イベントごとに複数閾値の判定結果 + 原因ギャップ長を計算する。"""
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if "1P" not in by_side or "2P" not in by_side:
        return []
    counts_by_side = {side: _visible_ojama_counts(rec.grids) for side, rec in by_side.items()}

    rows: list[dict] = []
    for ev in events:
        opp_side = "2P" if ev.fire_side == "1P" else "1P"
        opp_rec = by_side[opp_side]
        mask = opp_rec.game_idx == ev.game_idx
        row: dict = {
            "video_stem": ev.video_stem, "tier": ev.tier,
            "game_idx": ev.game_idx, "fire_side": ev.fire_side,
            "t_fire": ev.t_fire, "chain_count": ev.chain_count,
        }
        if not mask.any():
            for gt in gap_thresholds:
                row[f"status_gt{gt}"] = "no_opp_game_data"
                row[f"available_gt{gt}"] = False
            row["baseline_trigger_gap_sec"] = float("nan")
            rows.append(row)
            continue

        opp_game = _subset(opp_rec, mask)
        opp_counts = counts_by_side[opp_side][mask]

        for gt in gap_thresholds:
            result = _find_landing(
                opp_game.t_sec, opp_counts, ev.t_fire, gt, MAX_LANDING_SEARCH_SEC,
            )
            row[f"status_gt{gt}"] = result.status
            row[f"available_gt{gt}"] = result.available

        # 現行閾値 (2.0) で gap_in_window となった原因ギャップ長を実測
        if row[f"status_gt{BASELINE_GAP_THRESHOLD_SEC}"] == "gap_in_window":
            row["baseline_trigger_gap_sec"] = _first_violating_gap(
                opp_game.t_sec, ev.t_fire, BASELINE_GAP_THRESHOLD_SEC, MAX_LANDING_SEARCH_SEC,
            )
        else:
            row["baseline_trigger_gap_sec"] = float("nan")
        rows.append(row)
    return rows


def _print_coverage_table(df: pd.DataFrame, gap_thresholds: tuple[float, ...]) -> None:
    """閾値別の全体被覆率表を出力する (userタスク指定フォーマット)。"""
    print("\n[閾値別 全体被覆率]")
    print(f"{'閾値(秒)':>10} | {'全体被覆率':>10} | {'opp_available':>16} | {'landed件数':>10} | {'20%基準':>10}")
    total = len(df)
    for gt in gap_thresholds:
        avail_col = f"available_gt{gt}"
        status_col = f"status_gt{gt}"
        n_avail = int(df[avail_col].sum())
        n_landed = int((df[status_col] == "landed").sum())
        pct = n_avail / total * 100.0
        verdict = "OK" if pct > COVERAGE_GATE_PCT else "未達"
        label = "45.0(無制限相当)" if gt == 45.0 else f"{gt}"
        print(f"{label:>10} | {pct:>9.1f}% | {n_avail:>6}/{total:<8} | {n_landed:>10} | {verdict:>10}")


def _print_gap_distribution(df: pd.DataFrame) -> None:
    """gap_in_window (閾値2.0) の実際のギャップ長分布を出力する (userタスク必須項目)。"""
    gaps = df["baseline_trigger_gap_sec"].dropna()
    print(f"\n[gap_in_window (閾値={BASELINE_GAP_THRESHOLD_SEC}s) の実際のギャップ長分布] n={len(gaps)}")
    if gaps.empty:
        print("  (該当行が0件、出せない)")
        return
    print(f"  中央値   = {gaps.median():.2f} 秒")
    print(f"  75%tile  = {gaps.quantile(0.75):.2f} 秒")
    print(f"  90%tile  = {gaps.quantile(0.90):.2f} 秒")
    print(f"  最大値   = {gaps.max():.2f} 秒")
    print(f"  最小値   = {gaps.min():.2f} 秒")
    print(f"  平均値   = {gaps.mean():.2f} 秒")


def main(limit_videos: int | None = None) -> None:
    """メイン処理。limit_videos 指定時はその本数だけ処理する (見積もり用)。"""
    warnings.filterwarnings("ignore")
    npz_paths = [NPZ_DIR_REGEN / f"{stem}.npz" for stem in TIER_MAP]
    present = [p for p in npz_paths if p.exists()]
    if limit_videos is not None:
        present = present[:limit_videos]
    print(f"[INFO] 対象 {len(present)} 動画 (閾値スイープ: {GAP_THRESHOLDS_SEC})")

    t_start = time.time()
    sim = ChainSimulator()
    all_rows: list[dict] = []
    seq_id = 0
    for npz_path in sorted(present, key=lambda p: p.stem):
        _, defrag_events, seq_id = _process_video(npz_path, sim, seq_id)
        rows = _sweep_one_video(npz_path, defrag_events, GAP_THRESHOLDS_SEC)
        all_rows.extend(rows)
        print(f"  {npz_path.stem}: 発火{len(defrag_events)}件処理済み")

    elapsed = time.time() - t_start
    print(f"\n[所要時間] {elapsed:.2f} 秒 ({len(present)}動画, n_events={len(all_rows)})")

    if not all_rows:
        print("[ERROR] 発火イベントが0件でした。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    _print_coverage_table(df, GAP_THRESHOLDS_SEC)
    _print_gap_distribution(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-videos", type=int, default=None)
    args = parser.parse_args()
    main(limit_videos=args.limit_videos)
