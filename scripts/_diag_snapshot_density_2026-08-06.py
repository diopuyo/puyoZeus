"""診断2: v4全編npzのスナップショット欠落分類 (2026-08-06、使い捨て診断)。

物差し52盤面のうち突合不能5盤面 (c11 1P f18664 / c17 1P f89636 /
c17 2P f89724 / c11 2P f89678 / c15 2P f15646) について、ラベル時刻
±3秒の v4 npz スナップショット密度を OFF基準 (baseline npz) と比較し、
「ガードで確定が止まっていた」のか「試合境界等の正当な欠落」のかを
npz解析のみ (再走行不要) で分類する。

## 既存資産の再利用 (コピペ禁止指示への対応)
`scripts/measure_effect_gate_c_2026-08-04.py` の `_load_npz_index` /
`_find_by_frame_idx_exact` を importlib 動的import で再利用する
(ファイル名にハイフンを含むため通常の `from ... import` は不可)。
突合方式 (frame_idx完全一致→±0.35秒最近傍フォールバック) は
`scripts/_measure_yardstick_v4_2026-08-05.py` の `NEAREST_MATCH_TOLERANCE_SEC`
をそのまま踏襲する。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_snapshot_density_2026-08-06
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")

# =============================================================================
# 定数
# =============================================================================

BASELINE_NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_allframes_ref_2026-07-30")
V4_NPZ_DIR: Path = Path("data/verify/board_labels_v4_prod_2026-08-05")
VIDEO_DIR: Path = Path("/home/ryouj/frames")

# 突合可能の判定許容 (scripts/_measure_yardstick_v4_2026-08-05.py と同一値)。
NEAREST_MATCH_TOLERANCE_SEC: float = 0.35
# 密度比較窓 (coordinator指定、ラベル時刻±3秒)。
DENSITY_WINDOW_SEC: float = 3.0


@dataclass(frozen=True)
class Target:
    """診断対象の1盤面。"""

    video_stem: str
    side: str
    frame_idx: int


# 突合不能5盤面 (診断2) + c22診断1の対照確認用。
TARGETS: tuple[Target, ...] = (
    Target("c11", "1P", 18664),
    Target("c17", "1P", 89636),
    Target("c17", "2P", 89724),
    Target("c11", "2P", 89678),
    Target("c15", "2P", 15646),
    Target("c22", "2P", 188154),  # 診断1: v4に本当にスナップショットがあるか確認
)


def _video_fps(video_stem: str) -> float:
    """動画の実fpsを取得する (frame_idx→t_sec換算用)。"""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"video_{video_stem}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def _load_all_baseline_indices(video_stem: str) -> "list[object]":
    """baseline npz (試合分割 {stem}_gN.npz) を全て読み込む。"""
    indices = []
    for npz_path in sorted(BASELINE_NPZ_DIR.glob(f"{video_stem}_g*.npz")):
        idx = _MC._load_npz_index(npz_path)
        if idx is not None:
            indices.append(idx)
    return indices


def _load_v4_index(video_stem: str) -> "object | None":
    """v4 npz (全編1ファイル {stem}.npz) を読み込む。"""
    return _MC._load_npz_index(V4_NPZ_DIR / f"{video_stem}.npz")


def _count_snapshots_in_window(
    indices: "list[object]", side: str, center_t: float, half_window: float,
) -> "tuple[int, float | None]":
    """(side, center_t±half_window) 内のスナップショット数と最近接時刻差を返す。"""
    n = 0
    best_dt: "float | None" = None
    for idx in indices:
        mask = idx.sides == side
        cand = np.where(mask)[0]
        if len(cand) == 0:
            continue
        dt = np.abs(idx.t_secs[cand] - center_t)
        in_window = dt <= half_window
        n += int(np.sum(in_window))
        if len(dt) > 0:
            local_best = float(np.min(dt))
            if best_dt is None or local_best < best_dt:
                best_dt = local_best
    return n, best_dt


def _classify(
    baseline_n: int, v4_n: int, baseline_best_dt: "float | None",
    v4_best_dt: "float | None",
) -> str:
    """密度比較から欠落原因を分類する。"""
    baseline_has_exact = (
        baseline_best_dt is not None and baseline_best_dt <= NEAREST_MATCH_TOLERANCE_SEC
    )
    v4_has_exact = (
        v4_best_dt is not None and v4_best_dt <= NEAREST_MATCH_TOLERANCE_SEC
    )
    if baseline_n == 0 and v4_n == 0:
        return "正当な欠落 (両方とも±3秒無データ = 試合境界/シーン切替の可能性)"
    if baseline_has_exact and not v4_has_exact:
        return "ガード起因の欠落疑い (OFFは±0.35秒一致あり、v4は無し)"
    if not baseline_has_exact and not v4_has_exact and baseline_n > 0 and v4_n > 0:
        return "両方とも粗い (ラベル瞬間近傍が両方とも間引かれている、収集方式起因)"
    if v4_n < baseline_n * 0.3 and baseline_n >= 3:
        return "ガードによる密度低下疑い (v4のスナップショット数がOFFの30%未満)"
    return "判定不能 (追加確認要)"


def diagnose_one(target: Target) -> None:
    """1盤面分の密度比較 + 分類を出力する。"""
    fps = _video_fps(target.video_stem)
    label_t = target.frame_idx / fps
    baseline_indices = _load_all_baseline_indices(target.video_stem)
    v4_idx = _load_v4_index(target.video_stem)
    b_n, b_dt = _count_snapshots_in_window(
        baseline_indices, target.side, label_t, DENSITY_WINDOW_SEC,
    )
    v4_n, v4_dt = (
        _count_snapshots_in_window([v4_idx], target.side, label_t, DENSITY_WINDOW_SEC)
        if v4_idx is not None else (0, None)
    )
    category = _classify(b_n, v4_n, b_dt, v4_dt)
    print(
        f"{target.video_stem} {target.side} f{target.frame_idx} "
        f"(label_t={label_t:.3f}, fps={fps:.1f})"
    )
    print(
        f"  baseline: {b_n}件/±{DENSITY_WINDOW_SEC}秒, 最近接dt="
        f"{f'{b_dt:.3f}' if b_dt is not None else 'N/A'}"
    )
    print(
        f"  v4      : {v4_n}件/±{DENSITY_WINDOW_SEC}秒, 最近接dt="
        f"{f'{v4_dt:.3f}' if v4_dt is not None else 'N/A'}"
    )
    print(f"  [分類] {category}\n")


def main() -> None:
    for target in TARGETS:
        diagnose_one(target)


if __name__ == "__main__":
    main()
