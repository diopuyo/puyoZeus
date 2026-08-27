"""STABLE持続ゲート A/B 分析 (2026-08-18、一時スクリプト)。

_ab_stable_persistence_gate_probe_2026-08-18.py が書き出した pkl (1動画1回の
実行ログ: STABLE候補列・bstate時系列・生diff系列) を読み込み、任意の
(窓長, 閾値, 最上段除外有無) 設定について post-hoc に:
  1. 行数・OFF基準に対する回復率
  2. 記録された盤面のうち「相手が連鎖中等 (非STABLE)」「自分が直近連鎖中」の
     割合 (汚染指標)
  3. 生diff値の分布 (OFF基準で実際に記録された時刻に絞った分布、
     最上段除外あり/なし比較)
を計算する。src/ 側の定数は一切変更しない (読み込み専用インポート)。
"""
from __future__ import annotations

import bisect
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 自分が直近連鎖中とみなす遡り秒数 (2026-08-18 診断用パラメータ、本番定数ではない)。
SELF_RECENT_CHAIN_LOOKBACK_SEC: float = 1.0

SIDES = ("1P", "2P")
OTHER_SIDE = {"1P": "2P", "2P": "1P"}


@dataclass(frozen=True)
class GateConfig:
    """A/B 比較対象の1設定。"""

    name: str
    window_sec: float | None   # None = ゲート無効 (OFF)
    diff_threshold: float | None
    use_no_top: bool = False


CONFIGS: tuple[GateConfig, ...] = (
    GateConfig("OFF (旧構成相当)", None, None),
    GateConfig("現行 0.25s/1.0", 0.25, 1.0),
    GateConfig("窓0.15s/閾値1.0", 0.15, 1.0),
    GateConfig("窓0.10s/閾値1.0", 0.10, 1.0),
    GateConfig("窓0.25s/閾値1.5", 0.25, 1.5),
    GateConfig("窓0.25s/閾値2.0", 0.25, 2.0),
    GateConfig("現行+最上段除外", 0.25, 1.0, use_no_top=True),
)


def _load(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _diff_lookup_arrays(diffs: list[tuple[float, float, float]]):
    """(t_sec, diff_full, diff_no_top) の系列を t_sec 昇順の3配列に変換する。"""
    if not diffs:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(diffs, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1], arr[order, 2]


def _gate_pass_series(t_sorted: np.ndarray, d_sorted: np.ndarray,
                       window_sec: float, threshold: float) -> np.ndarray:
    """各時刻 t について「直近 window_sec 秒の diff が全て threshold 未満」かを返す。

    src.board_motion.is_raw_pixel_stable と同じ意味論 (空履歴は安全側 True)。
    """
    n = len(t_sorted)
    out = np.ones(n, dtype=bool)
    j = 0
    running_max = -np.inf
    # 単調増加窓のため二重ポインタで最大値を再計算 (小規模データなので
    # 素朴なO(n log n)でも十分高速、可読性優先)。
    for i in range(n):
        lo = t_sorted[i] - window_sec
        j = bisect.bisect_left(t_sorted, lo, 0, i + 1)
        window_vals = d_sorted[j:i + 1]
        out[i] = bool(np.all(window_vals < threshold)) if window_vals.size else True
    return out


def _replay_dedup(candidate_tsec: list[float], candidate_grid: list[bytes],
                   gate_pass_lookup) -> list[tuple[float, bytes]]:
    """候補列に (gate_pass AND dedup) を適用して実際に emit される行を再生する。"""
    kept: list[tuple[float, bytes]] = []
    last_grid: bytes | None = None
    for t_sec, grid in zip(candidate_tsec, candidate_grid):
        if not gate_pass_lookup(t_sec):
            continue
        if grid == last_grid:
            continue
        kept.append((t_sec, grid))
        last_grid = grid
    return kept


def _make_gate_lookup(config: GateConfig, t_sorted: np.ndarray,
                       diff_full_sorted: np.ndarray, diff_no_top_sorted: np.ndarray):
    """config から「時刻 t における gate_pass」を返す関数を作る (OFFなら常にTrue)。"""
    if config.window_sec is None:
        return lambda t: True
    d_sorted = diff_no_top_sorted if config.use_no_top else diff_full_sorted
    passes = _gate_pass_series(t_sorted, d_sorted, config.window_sec, config.diff_threshold)
    # candidate の t_sec は diff系列と同じフレームループ由来なのでほぼ厳密一致するが、
    # 浮動小数の丸め差を吸収するため最近傍検索にする。
    def _lookup(t: float) -> bool:
        if len(t_sorted) == 0:
            return True
        idx = int(np.searchsorted(t_sorted, t))
        idx = min(idx, len(t_sorted) - 1)
        if idx > 0 and abs(t_sorted[idx - 1] - t) < abs(t_sorted[idx] - t):
            idx -= 1
        return bool(passes[idx])
    return _lookup


def _bstate_at(bstate_tsec: np.ndarray, bstate_val: np.ndarray, t: float) -> str:
    """bstate 時系列から時刻 t 直近のエントリを返す (見つからなければ 'unknown')。"""
    if len(bstate_tsec) == 0:
        return "unknown"
    idx = int(np.searchsorted(bstate_tsec, t))
    idx = min(idx, len(bstate_tsec) - 1)
    if idx > 0 and abs(bstate_tsec[idx - 1] - t) < abs(bstate_tsec[idx] - t):
        idx -= 1
    return str(bstate_val[idx])


def _self_recent_chain(bstate_tsec: np.ndarray, bstate_val: np.ndarray, t: float) -> bool:
    """直近 SELF_RECENT_CHAIN_LOOKBACK_SEC 秒以内に自分が CHAIN 状態だったか。"""
    lo = t - SELF_RECENT_CHAIN_LOOKBACK_SEC
    j = int(bisect.bisect_left(bstate_tsec, lo))
    k = int(bisect.bisect_right(bstate_tsec, t))
    return bool(np.any(bstate_val[j:k] == "chain"))


def analyze_video(pkl_path: Path) -> dict:
    data = _load(pkl_path)
    video_id = data["video_id"]
    result: dict = {"video_id": video_id, "configs": {}}

    # side別 diff系列 (t_sec昇順)
    diff_arrays = {s: _diff_lookup_arrays(data["diffs"][s]) for s in SIDES}
    # side別 bstate系列 (t_sec昇順)
    bstate_arrays = {}
    for s in SIDES:
        bs = sorted(data["bstate"][s], key=lambda x: x[0])
        bstate_arrays[s] = (
            np.array([x[0] for x in bs], dtype=np.float64),
            np.array([x[1] for x in bs], dtype="<U16"),
        )

    for config in CONFIGS:
        cfg_result: dict = {"total_rows": 0, "per_side": {}}
        all_kept_diffs_full: list[float] = []
        all_kept_diffs_no_top: list[float] = []
        n_opp_active = 0
        n_self_recent_chain = 0
        n_total_kept = 0
        for side in SIDES:
            t_sorted, diff_full_sorted, diff_no_top_sorted = diff_arrays[side]
            gate_lookup = _make_gate_lookup(config, t_sorted, diff_full_sorted, diff_no_top_sorted)
            cands = sorted(data["candidates"][side], key=lambda x: x[0])
            cand_t = [c[0] for c in cands]
            cand_grid = [c[2] for c in cands]
            kept = _replay_dedup(cand_t, cand_grid, gate_lookup)
            cfg_result["per_side"][side] = len(kept)
            cfg_result["total_rows"] += len(kept)

            opp_bt, opp_bv = bstate_arrays[OTHER_SIDE[side]]
            self_bt, self_bv = bstate_arrays[side]
            for t_sec, _grid in kept:
                n_total_kept += 1
                if _bstate_at(opp_bt, opp_bv, t_sec) != "stable":
                    n_opp_active += 1
                if _self_recent_chain(self_bt, self_bv, t_sec):
                    n_self_recent_chain += 1
                # diff分布 (この設定で実際に記録された時刻の生diff値)
                idx = int(np.searchsorted(t_sorted, t_sec))
                idx = min(idx, len(t_sorted) - 1) if len(t_sorted) else 0
                if len(t_sorted):
                    all_kept_diffs_full.append(float(diff_full_sorted[idx]))
                    all_kept_diffs_no_top.append(float(diff_no_top_sorted[idx]))

        cfg_result["opp_active_pct"] = (
            100.0 * n_opp_active / n_total_kept if n_total_kept else float("nan")
        )
        cfg_result["self_recent_chain_pct"] = (
            100.0 * n_self_recent_chain / n_total_kept if n_total_kept else float("nan")
        )
        cfg_result["diff_full_at_kept"] = all_kept_diffs_full
        cfg_result["diff_no_top_at_kept"] = all_kept_diffs_no_top
        result["configs"][config.name] = cfg_result

    return result


def _pctile_str(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.array(values)
    return (f"n={len(arr)} mean={arr.mean():.3f} p50={np.percentile(arr,50):.3f} "
            f"p90={np.percentile(arr,90):.3f} p99={np.percentile(arr,99):.3f} "
            f"max={arr.max():.3f}")


def main() -> int:
    verify_dir = Path("data/verify/ab_stable_persistence_gate_2026-08-18")
    pkl_paths = sorted(verify_dir.glob("probe_*.pkl"))
    if not pkl_paths:
        print("no probe pkl found", file=sys.stderr)
        return 1

    per_video = [analyze_video(p) for p in pkl_paths]

    off_name = CONFIGS[0].name
    print("=" * 100)
    print("動画別 行数・回復率・汚染率")
    print("=" * 100)
    for v in per_video:
        off_rows = v["configs"][off_name]["total_rows"]
        print(f"\n[{v['video_id']}] OFF行数={off_rows}")
        for cfg in CONFIGS:
            c = v["configs"][cfg.name]
            recovery = 100.0 * c["total_rows"] / off_rows if off_rows else float("nan")
            print(
                f"  {cfg.name:22s} rows={c['total_rows']:5d} "
                f"(1P={c['per_side']['1P']:4d} 2P={c['per_side']['2P']:4d}) "
                f"回復率={recovery:6.1f}%  相手非STABLE率={c['opp_active_pct']:5.1f}%  "
                f"自分直近連鎖率={c['self_recent_chain_pct']:5.1f}%"
            )

    print("\n" + "=" * 100)
    print("生diff分布 (OFF基準で記録された時刻に絞る、全動画プール)")
    print("=" * 100)
    pooled_full = []
    pooled_no_top = []
    for v in per_video:
        pooled_full.extend(v["configs"][off_name]["diff_full_at_kept"])
        pooled_no_top.extend(v["configs"][off_name]["diff_no_top_at_kept"])
    print(f"  full ROI     : {_pctile_str(pooled_full)}")
    print(f"  top行除外あり: {_pctile_str(pooled_no_top)}")

    print("\n" + "=" * 100)
    print("設定別 集計 (全動画合算)")
    print("=" * 100)
    off_total = sum(v["configs"][off_name]["total_rows"] for v in per_video)
    for cfg in CONFIGS:
        total_rows = sum(v["configs"][cfg.name]["total_rows"] for v in per_video)
        recovery = 100.0 * total_rows / off_total if off_total else float("nan")
        opp_pcts = [v["configs"][cfg.name]["opp_active_pct"] for v in per_video]
        self_pcts = [v["configs"][cfg.name]["self_recent_chain_pct"] for v in per_video]
        print(
            f"  {cfg.name:22s} total_rows={total_rows:6d} 回復率={recovery:6.1f}%  "
            f"相手非STABLE率(動画平均)={np.nanmean(opp_pcts):5.1f}%  "
            f"自分直近連鎖率(動画平均)={np.nanmean(self_pcts):5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
