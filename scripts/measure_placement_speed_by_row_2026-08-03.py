"""段別最速設置時間の実測較正 (2026-08-03 user提案 + 外れ値除外の追加指示)。

背景 (user伝授、推測禁止のドメイン知見): ぷよの落下速度は一定だが、盤面が
埋まるほど落下距離が短くなるため、上部 (段が高い=盤面が埋まっている状態)
ほど同じ1手を速く置ける。既存 estimate_available_hands は SEC_PER_HAND
(全段共通の固定値0.733秒) を使っており、この段別の物理を無視している。

本スクリプトは 66動画 npz 全部から「通常設置イベント」(色ぷよ+2、おじゃま
増加なし、消去を挟まない) を段別に抽出し、以下の手順で「その段の最速設置
時間」を求める (2026-08-03 user追加指示の外れ値除外込み):
    1. 物理下限による除外: 全体プールのdt分布から「主要モードの開始点」を
       ヒストグラムで検出し、それより速い (=物理的に不可能) 孤立左裾を
       認識アーティファクトとして除外する (根拠は _detect_physical_min_dt_sec
       のヒストグラム証拠として出力する)。
    2. 段ごとに、物理下限を通過したdtの最速四分位 (下位25%、user指定) を取る。
    3. その四分位内でさらに IQR 基準の頑健化を行う (外れ値を除いた平均)。
    4. 段ごとに 全サンプル数/物理下限除外n/四分位n/IQR除外n/最終n を報告する。

n が薄い段は隣接段とプールする (どの段をプールしたか明記)。

本スクリプトは計測のみ (estimate_available_hands への配線は別タスク)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from src.board import BOARD_ROWS, COLOR_OJAMA
from scripts.label_exchange_outcome import NpzRecord, _load_npz

NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/placement_speed_by_row_2026-08-03")

FASTEST_QUARTILE_FRAC: float = 0.25       # user指定「最速側上位25%平均」
MIN_RAW_SAMPLES_PER_ROW: int = 20         # これ未満は隣接段とプール (2026-08-03指示)
HISTOGRAM_BIN_SEC: float = 0.05           # 物理下限検出用ヒストグラム幅
HISTOGRAM_MAX_SEC: float = 3.0            # ヒストグラム走査上限
MIN_DENSITY_FRAC: float = 0.005           # 「主要モード開始」判定の密度閾値 (全体件数比)
CONSECUTIVE_BINS_FOR_MODE_START: int = 2  # 主要モード開始の連続ビン数 (単発スパイク除外)
IQR_OUTLIER_MULTIPLIER: float = 1.5       # 四分位内さらにIQRで頑健化 (標準的な係数)


@dataclass(frozen=True)
class PlacementEvent:
    """1回の通常設置イベント (段別最速時間較正用)。"""
    row_index: int   # 新規2セルのうちtopmost (row_index最小、盤面座標そのまま、0=最上段)
    dt_sec: float     # 直前の設置からの経過秒


def _color_cell_count(grid: np.ndarray) -> int:
    """色ぷよ (空・おじゃま以外) のセル数。"""
    return int(((grid != 0) & (grid != COLOR_OJAMA)).sum())


def _new_color_positions(before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    """beforeで空だったがafterで色ぷよになったセル一覧。"""
    rows, cols = np.where((before == 0) & (after != 0) & (after != COLOR_OJAMA))
    return list(zip(rows.tolist(), cols.tolist()))


def _is_simple_placement(before: np.ndarray, after: np.ndarray) -> bool:
    """通常設置 (色ぷよ+2のみ、おじゃま増加・消去を挟まない) か判定する。"""
    if int((after == COLOR_OJAMA).sum()) != int((before == COLOR_OJAMA).sum()):
        return False
    if _color_cell_count(after) - _color_cell_count(before) != 2:
        return False
    cleared = (before != 0) & (before != COLOR_OJAMA) & (after == 0)
    return not bool(cleared.any())


def extract_placement_events(rec: NpzRecord) -> list[PlacementEvent]:
    """1動画1サイド分、game_idx別に連続STABLE行から設置イベントを抽出する。"""
    events: list[PlacementEvent] = []
    for game_idx in np.unique(rec.game_idx):
        mask = rec.game_idx == game_idx
        order = np.argsort(rec.t_sec[mask])
        t, grids = rec.t_sec[mask][order], rec.grids[mask][order]
        for i in range(1, len(t)):
            before, after = grids[i - 1], grids[i]
            if not _is_simple_placement(before, after):
                continue
            positions = _new_color_positions(before, after)
            if len(positions) != 2:
                continue
            dt = float(t[i] - t[i - 1])
            if dt <= 0.0:
                continue
            events.append(PlacementEvent(row_index=min(p[0] for p in positions), dt_sec=dt))
    return events


def collect_all_events(npz_dir: Path) -> list[PlacementEvent]:
    """66動画分の npz 全部から設置イベントを集める (1P/2P両サイド)。"""
    events: list[PlacementEvent] = []
    npz_paths = sorted(npz_dir.glob("*.npz"))
    for path in npz_paths:
        for rec in _load_npz(path):
            events.extend(extract_placement_events(rec))
    print(f"[collect] {len(npz_paths)}動画から設置イベント {len(events)}件 抽出")
    return events


def detect_physical_min_dt_sec(all_dt: np.ndarray) -> tuple[float, dict]:
    """全体プールのdt分布から「物理的に不可能な速さの孤立左裾」の下限を検出する。

    ヒストグラム (HISTOGRAM_BIN_SEC刻み) で、密度が全体件数比 MIN_DENSITY_FRAC を
    CONSECUTIVE_BINS_FOR_MODE_START 連続で超えた最初のビンを「主要モード開始」
    とみなし、そのビンの左端を保守的な下限とする (それより速い間隔は認識
    アーティファクト=孤立ノイズとして除外する根拠、証拠のヒストグラムも返す)。
    """
    bins = np.arange(0.0, HISTOGRAM_MAX_SEC + HISTOGRAM_BIN_SEC, HISTOGRAM_BIN_SEC)
    counts, edges = np.histogram(all_dt, bins=bins)
    threshold = MIN_DENSITY_FRAC * len(all_dt)
    mode_start_idx = 0
    for i in range(len(counts) - CONSECUTIVE_BINS_FOR_MODE_START + 1):
        if all(c >= threshold for c in counts[i:i + CONSECUTIVE_BINS_FOR_MODE_START]):
            mode_start_idx = i
            break
    floor_sec = float(edges[mode_start_idx])
    evidence = {
        "threshold_count": float(threshold),
        "bins_sec": edges[:mode_start_idx + 6].tolist(),
        "counts": counts[:mode_start_idx + 5].tolist(),
    }
    return floor_sec, evidence


def _trimmed_by_iqr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IQR基準で外れ値を除いた配列 (kept) と除外された配列 (excluded) を返す。"""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - IQR_OUTLIER_MULTIPLIER * iqr, q3 + IQR_OUTLIER_MULTIPLIER * iqr
    keep_mask = (values >= lo) & (values <= hi)
    return values[keep_mask], values[~keep_mask]


def compute_row_fastest_time(raw_dt: np.ndarray, phys_floor: float) -> dict:
    """1段分の生dt配列から、物理下限除外→最速四分位→IQR頑健化を経た最終値を返す。"""
    total_n = len(raw_dt)
    phys_valid = raw_dt[raw_dt >= phys_floor]
    phys_excluded_n = total_n - len(phys_valid)
    if len(phys_valid) == 0:
        return {"total_n": total_n, "phys_excluded_n": phys_excluded_n, "quartile_n": 0,
                "iqr_excluded_n": 0, "final_n": 0, "final_mean_sec": float("nan")}
    sorted_valid = np.sort(phys_valid)
    q_n = max(1, int(round(len(sorted_valid) * FASTEST_QUARTILE_FRAC)))
    quartile = sorted_valid[:q_n]
    kept, excluded = _trimmed_by_iqr(quartile)
    final_mean = float(np.mean(kept)) if len(kept) > 0 else float("nan")
    return {"total_n": total_n, "phys_excluded_n": phys_excluded_n, "quartile_n": q_n,
            "iqr_excluded_n": len(excluded), "final_n": len(kept), "final_mean_sec": final_mean}


def pool_thin_rows(dt_by_row: dict[int, list[float]]) -> dict[str, np.ndarray]:
    """件数の薄い段を隣接段とプールする (row_index昇順=盤面上から下、
    まとめて処理し MIN_RAW_SAMPLES_PER_ROW に届くまで束ねる)。

    戻り値のキーは "5" (単独段) または "5-7" (プール範囲) の文字列表記。
    """
    pooled: dict[str, np.ndarray] = {}
    rows_sorted = sorted(dt_by_row.keys())
    buffer_rows: list[int] = []
    buffer_vals: list[float] = []
    for row in rows_sorted:
        buffer_rows.append(row)
        buffer_vals.extend(dt_by_row[row])
        if len(buffer_vals) >= MIN_RAW_SAMPLES_PER_ROW:
            key = str(buffer_rows[0]) if len(buffer_rows) == 1 else f"{buffer_rows[0]}-{buffer_rows[-1]}"
            pooled[key] = np.array(buffer_vals, dtype=float)
            buffer_rows, buffer_vals = [], []
    if buffer_vals:  # 末尾の残りは直前グループへ追記 (無ければ単独グループのまま残す)
        key = str(buffer_rows[0]) if len(buffer_rows) == 1 else f"{buffer_rows[0]}-{buffer_rows[-1]}"
        if pooled:
            last_key = list(pooled.keys())[-1]
            merged_key = f"{last_key}+{key}"
            pooled[merged_key] = np.concatenate([pooled.pop(last_key), np.array(buffer_vals)])
        else:
            pooled[key] = np.array(buffer_vals, dtype=float)
    return pooled


def _plot_distribution(dt_by_row: dict[int, list[float]], phys_floor: float, out_path: Path) -> None:
    """段別dt分布 (箱ひげ) + 物理下限ラインのPNGを保存する。"""
    meiryo_path = "/mnt/c/Windows/Fonts/meiryo.ttc"
    if Path(meiryo_path).exists():
        font_manager.fontManager.addfont(meiryo_path)
        plt.rcParams["font.family"] = "Meiryo"
    rows_sorted = sorted(dt_by_row.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot([dt_by_row[r] for r in rows_sorted], positions=rows_sorted,
               widths=0.6, showfliers=False)
    ax.axhline(phys_floor, color="red", linestyle="--", linewidth=1.0,
               label=f"物理下限 (検出値={phys_floor:.2f}秒)")
    ax.set_xlabel("row_index (盤面座標、0=最上段/隠し段側、大きいほど下段)")
    ax.set_ylabel("設置間隔 dt (秒)")
    ax.set_title("段別 (row_index別) 設置間隔分布 (外れ値非表示、66動画プール)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] 保存: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = collect_all_events(NPZ_DIR)
    all_dt = np.array([e.dt_sec for e in events], dtype=float)
    phys_floor, evidence = detect_physical_min_dt_sec(all_dt)

    print(f"\n=== 物理下限検出 (全体プール n={len(all_dt)}) ===")
    print(f"  下限={phys_floor:.3f}秒 (閾値件数={evidence['threshold_count']:.1f})")
    print(f"  検出根拠 (下限付近のヒストグラム、{HISTOGRAM_BIN_SEC}秒刻み):")
    for i, (edge, cnt) in enumerate(zip(evidence["bins_sec"], evidence["counts"])):
        flag = " <- 主要モード開始" if abs(edge - phys_floor) < 1e-9 else ""
        print(f"    [{edge:.2f}, {edge + HISTOGRAM_BIN_SEC:.2f}): n={cnt}{flag}")

    dt_by_row: dict[int, list[float]] = {}
    for e in events:
        dt_by_row.setdefault(e.row_index, []).append(e.dt_sec)
    _plot_distribution(dt_by_row, phys_floor, OUT_DIR / "placement_speed_by_row_boxplot.png")

    pooled = pool_thin_rows(dt_by_row)
    print(f"\n=== 段別最速設置時間 (物理下限除外→最速{FASTEST_QUARTILE_FRAC:.0%}→IQR頑健化) ===")
    print(f"{'段(row_index)':>16} {'全n':>6} {'物理下限除外n':>12} {'四分位n':>8}"
          f" {'IQR除外n':>8} {'最終n':>6} {'最速平均秒':>10}")
    for key in sorted(pooled.keys(), key=lambda k: int(k.split("-")[0].split("+")[0])):
        stat = compute_row_fastest_time(pooled[key], phys_floor)
        print(f"{key:>16} {stat['total_n']:6d} {stat['phys_excluded_n']:12d}"
              f" {stat['quartile_n']:8d} {stat['iqr_excluded_n']:8d} {stat['final_n']:6d}"
              f" {stat['final_mean_sec']:10.3f}")


if __name__ == "__main__":
    main()
