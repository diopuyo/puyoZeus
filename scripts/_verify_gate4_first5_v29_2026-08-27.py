"""先頭5試合v29レビュー成果物の表示・会計不変条件を検証する。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path("data/verify/gate4_first5_review_2026-08-27")
UNRESOLVED_ABS_CAP = 43.920
REVIEW_SCENE_SEC = 301.700


def _load(path: Path) -> dict[str, np.ndarray]:
    """npzをファイル閉鎖後も使える辞書へ読む。"""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _one_second_stats(t_sec: np.ndarray, adv: np.ndarray) -> tuple[int, float]:
    """1秒窓150点以上の件数と最大変化を返す。"""
    count, maximum = 0, 0.0
    for index, now in enumerate(t_sec):
        previous = int(np.searchsorted(t_sec, now - 1.0))
        if index <= previous:
            continue
        delta = abs(float(adv[index] - adv[previous]))
        count += int(delta >= 150.0)
        maximum = max(maximum, delta)
    return count, maximum


def verify(stem: str) -> None:
    """表示・episode・grossの保存則とレビュー局面を検査する。"""
    display = _load(ROOT / f"{stem}_display.npz")
    episode = _load(ROOT / f"{stem}_episode.npz")
    timeline = _load(ROOT / f"{stem}_timeline.npz")
    if not np.array_equal(display["t_sec"], episode["t_sec"]):
        raise AssertionError("displayとepisodeの時刻窓が一致しない")
    swings, max_swing = _one_second_stats(
        display["t_sec"], display["display_adv"])
    unresolved = episode["is_unresolved"].astype(bool)
    cap_violations = int(np.count_nonzero(
        unresolved & (np.abs(display["display_adv"]) > UNRESOLVED_ABS_CAP + 1e-6)))
    active_but_resolved = int(np.count_nonzero(
        (episode["active_chain_count"].astype(int) > 0)
        & (episode["stage"].astype(str) == "RESOLVED")))
    illegal_hard = int(np.count_nonzero(
        episode["hard_override_applied"].astype(bool)
        & ~episode["allows_hard_override"].astype(bool)))
    # ledger_residual_all は「未決済の残量」そのもので、0である必要はない。
    # 正式検算式 generated = canceled + landed + outstanding の残差を測る。
    global_residual = (
        episode["total_generated"].astype(float)
        - episode["total_canceled"].astype(float)
        - episode["total_landed"].astype(float)
        - episode["ledger_residual_all"].astype(float))
    ledger_residual = int(np.count_nonzero(np.abs(global_residual) > 1e-9))
    gross_residual = int(np.count_nonzero(
        (timeline["gross_residual_p1"].astype(int) != 0)
        | (timeline["gross_residual_p2"].astype(int) != 0)))
    scene_index = int(np.argmin(abs(display["t_sec"] - REVIEW_SCENE_SEC)))
    scene_p1 = float(display["display_p1"][scene_index])
    print(
        f"rows={len(display['t_sec'])} 1sec150={swings} max1sec={max_swing:.3f} "
        f"unresolved_cap_violations={cap_violations} "
        f"active_but_resolved={active_but_resolved} illegal_hard={illegal_hard} "
        f"ledger_residual={ledger_residual}/{len(episode['t_sec'])} "
        f"gross_residual={gross_residual}/{len(timeline['t_sec'])} "
        f"scene301.700_p1={scene_p1:.4f}"
    )
    assert swings == 0
    assert cap_violations == 0
    assert active_but_resolved == 0
    assert illegal_hard == 0
    assert ledger_residual == 0
    assert gross_residual == 0
    assert scene_p1 >= 0.70


def main() -> None:
    """CLI入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stem", default="gate4_first5_cond5_v29_review_physical_chain_id_guard")
    args = parser.parse_args()
    verify(args.stem)


if __name__ == "__main__":
    main()
