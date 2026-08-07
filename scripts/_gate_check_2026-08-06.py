"""バーストガード全域無悪化ゲート 補助集計 (使い捨て・2026-08-06)。

CYCLE_FINDINGS.md 4.2-quater/quinquies の I1/C1 系メトリクスを、
data/verify/burst_guard_2026-08-05/on_v2_full/*.npz (ガードON, boards_lean形式)
と data/indicators_v2/boards_lean_regen_2026-07-31/*.npz (アンカー, ガードOFF)
から算出可能な範囲でプロキシ計算する。

注意 (このスクリプトの限界、report本文に転記済):
- boards_lean npz は「STABLE かつ直前と異なる盤面」のみを記録する dedup 形式。
  non_stable_consecutive_frames (I1-2) はフレーム欠落と重複除外を区別できず算出不可。
- postprocess_corruption (D1) は raw_cnn/raw_hsv/confirmed の3値が必要で、
  boards_lean npz には confirmed 値のみしかなく算出不可。
- on_v2_full は各動画で anchor よりはるかに短い t_sec 範囲のみ処理済 (burst イベント
  最終検出時刻+数秒までで --max-sec 打ち切り)。本スクリプトは anchor 側を on_v2_full
  の t_sec 最大値でトリムした "matched" 版と、トリムなし "full" 版の両方を出す。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ON_DIR = Path("data/verify/burst_guard_2026-08-05/on_v2_full")
ANCHOR_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")

COLOR_EMPTY = 0
COLOR_UNKNOWN = 10
MIDGAME_T_SEC = 30.0
MIDGAME_COL_MIN_FRAMES = 30
PER_COL_UNKNOWN_WARNING = 0.15
PER_COL_UNKNOWN_CRITICAL = 0.30
MIDGAME_COL_EMPTY_CRITICAL = 0.99
AVG_PUYO_COUNT_CRITICAL_RATIO = 0.85

# c22 は既知の長時間劣化 (列凍結) 疑いあり動画 → 層別報告対象
KNOWN_DEGRADED_VIDEOS = {"c22"}


def _per_col_unknown_rate(grids: np.ndarray) -> dict[int, float]:
    """grids: (n, 13, 6) → col別 UNKNOWN(10) 比率。"""
    out = {}
    n = grids.shape[0]
    if n == 0:
        return {c: float("nan") for c in range(6)}
    for c in range(6):
        col = grids[:, :, c]
        out[c] = float(np.mean(col == COLOR_UNKNOWN))
    return out


def _per_col_midgame_empty_rate(
    grids: np.ndarray, t_sec: np.ndarray
) -> tuple[dict[int, float], int]:
    """中盤 (t>=30s) の col別 EMPTY(0) 比率。frame数 (行数) も返す。"""
    mask = t_sec >= MIDGAME_T_SEC
    n_frames = int(mask.sum())
    if n_frames < MIDGAME_COL_MIN_FRAMES:
        return {c: float("nan") for c in range(6)}, n_frames
    sub = grids[mask]
    out = {}
    for c in range(6):
        col = sub[:, :, c]
        out[c] = float(np.mean(col == COLOR_EMPTY))
    return out, n_frames


def _avg_puyo_count_per_row(grids: np.ndarray) -> float:
    """1行 (=1 STABLE snapshot) あたりの平均ぷよ数 (EMPTY/UNKNOWN 除く。OJAMA含む)。"""
    if grids.shape[0] == 0:
        return float("nan")
    puyo_mask = (grids != COLOR_EMPTY) & (grids != COLOR_UNKNOWN)
    per_row = puyo_mask.reshape(grids.shape[0], -1).sum(axis=1)
    return float(per_row.mean())


def _load_side_arrays(npz_path: Path, side: str, t_max: float | None = None):
    d = np.load(npz_path, allow_pickle=True)
    side_mask = d["side"] == side
    grids = d["grids"][side_mask]
    t_sec = d["t_sec"][side_mask]
    if t_max is not None:
        tmask = t_sec <= t_max
        grids = grids[tmask]
        t_sec = t_sec[tmask]
    return grids, t_sec


def analyze_video(vid_file: str) -> dict:
    on_path = ON_DIR / vid_file
    anchor_path = ANCHOR_DIR / vid_file
    d_on = np.load(on_path, allow_pickle=True)
    on_t_max = float(d_on["t_sec"].max())
    on_t_min = float(d_on["t_sec"].min())
    d_anchor_full = np.load(anchor_path, allow_pickle=True)
    anchor_t_max_full = float(d_anchor_full["t_sec"].max())

    result: dict = {
        "video": vid_file.replace(".npz", ""),
        "on_t_range": [on_t_min, on_t_max],
        "anchor_t_max_full": anchor_t_max_full,
        "coverage_ratio": on_t_max / anchor_t_max_full if anchor_t_max_full > 0 else float("nan"),
        "sides": {},
    }

    for side in ("1P", "2P"):
        on_grids, on_t = _load_side_arrays(on_path, side)
        anchor_m_grids, anchor_m_t = _load_side_arrays(anchor_path, side, t_max=on_t_max)
        anchor_f_grids, anchor_f_t = _load_side_arrays(anchor_path, side)

        on_unknown = _per_col_unknown_rate(on_grids)
        anchor_m_unknown = _per_col_unknown_rate(anchor_m_grids)
        anchor_f_unknown = _per_col_unknown_rate(anchor_f_grids)

        on_mid, on_mid_n = _per_col_midgame_empty_rate(on_grids, on_t)
        anchor_m_mid, anchor_m_mid_n = _per_col_midgame_empty_rate(anchor_m_grids, anchor_m_t)
        anchor_f_mid, anchor_f_mid_n = _per_col_midgame_empty_rate(anchor_f_grids, anchor_f_t)

        result["sides"][side] = {
            "n_rows_on": int(on_grids.shape[0]),
            "n_rows_anchor_matched": int(anchor_m_grids.shape[0]),
            "n_rows_anchor_full": int(anchor_f_grids.shape[0]),
            "per_col_unknown_rate_on": on_unknown,
            "per_col_unknown_rate_anchor_matched": anchor_m_unknown,
            "per_col_unknown_rate_anchor_full": anchor_f_unknown,
            "per_col_midgame_empty_rate_on": on_mid,
            "per_col_midgame_empty_rate_anchor_matched": anchor_m_mid,
            "per_col_midgame_empty_rate_anchor_full": anchor_f_mid,
            "midgame_n_frames_on": on_mid_n,
            "midgame_n_frames_anchor_matched": anchor_m_mid_n,
            "midgame_n_frames_anchor_full": anchor_f_mid_n,
            "avg_puyo_count_on": _avg_puyo_count_per_row(on_grids),
            "avg_puyo_count_anchor_matched": _avg_puyo_count_per_row(anchor_m_grids),
            "avg_puyo_count_anchor_full": _avg_puyo_count_per_row(anchor_f_grids),
        }
    return result


def main() -> None:
    files = sorted(p.name for p in ON_DIR.glob("*.npz"))
    all_results = []
    for f in files:
        try:
            r = analyze_video(f)
            all_results.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {f}: {e}")
    out_path = Path("data/verify/burst_guard_2026-08-05/_gate_check_2026-08-06_result.json")
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"wrote {out_path} ({len(all_results)} videos)")


if __name__ == "__main__":
    main()
