"""A0 (2026-07-24) CHAIN 保持時間モデル較正の viz (user 採否レビュー用)。

feedback_viz_eval_required / feedback_human_review_at_steps 準拠:
数値サマリだけでなく目視確認できる形で before(現行 0.3s/連鎖) / after
(較正値 base=3.4s + 1.5s×連鎖数, chain_max_hold=25s) の挙動差を可視化する。

出力 (data/verify/recognition_a0_chain_hold_recalibration/ 配下):
    - state_timeline_<video>.png : CHAIN/GRAVITY_SETTLE/STABLE 等の state
      系列を before/after で上下 2 段に並べたタイムライン (連鎖 trigger は
      縦線)。
    - confirmed_diff_<video>_event<n>.png : CHAIN→STABLE 遷移直後の
      confirmed_board と ChainSimulator 期待値の diff (不一致セル赤強調)
      を before/after で並べた盤面図。
    - lag_summary_<video>.png : イベント毎の STABLE確定ラグ (連鎖発火〜
      最初の STABLE) を before/after で棒グラフ比較。

src/ は一切変更しない (読み取り専用、既存スクリプトパターンと同じ)。
recognition_physics_review.py の _capture_frames をそのまま再利用し、
ロジックの二重実装を避ける。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._a0_viz_chain_hold_recalibration \
        --video-stem c62 --start-sec 895 --max-sec 65
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, COLOR_EMPTY  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from scripts.recognition_physics_review import (  # noqa: E402
    _capture_frames, _new_chain_triggers, _find_first_stable_after,
    _FrameRecord,
)

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_a0_chain_hold_recalibration"

# 較正値 (計装 a287c587 実測、A0 提案値)。
CALIBRATED_BASE_SEC: float = 3.4
CALIBRATED_PER_STEP_SEC: float = 1.5
CALIBRATED_MAX_HOLD_SEC: float = 25.0

# state 別の表示色 (matplotlib color name)。
STATE_COLORS: dict[str, str] = {
    "STABLE": "#8fd18f",          # 緑系 (安定)
    "TSUMO_FALL": "#a8c8e8",      # 薄青
    "CHAIN": "#e8a0a0",           # 赤系 (連鎖中)
    "GRAVITY_SETTLE": "#e8cfa0",  # 橙系 (重力整合待ち)
    "OJAMA_FALL": "#c9a8e8",      # 紫系
    "MENU": "#cccccc",
}
DEFAULT_STATE_COLOR: str = "#eeeeee"

# 盤面図の色パレット (Board 色コード -> RGB)。
CELL_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    COLOR_EMPTY: (1.0, 1.0, 1.0),
    1: (0.85, 0.2, 0.2),   # 赤
    2: (0.2, 0.4, 0.85),   # 青
    3: (0.2, 0.7, 0.3),    # 緑
    4: (0.9, 0.8, 0.15),   # 黄
    5: (0.6, 0.25, 0.75),  # 紫
    9: (0.55, 0.55, 0.55), # おじゃま
    10: (0.3, 0.3, 0.3),   # UNKNOWN
}


def _run_both(
    video_stem: str, start_sec: float, max_sec: float, side: str = "1P",
) -> tuple[list[_FrameRecord], list[_FrameRecord]]:
    """default / calibrated 両方で 1 側の記録を取得する。"""
    print(f"[viz] {video_stem} start={start_sec} max={max_sec}s: default 実行中...")
    before = _capture_frames(video_stem, start_sec, max_sec)[side]
    print(f"[viz] {video_stem}: calibrated (base={CALIBRATED_BASE_SEC} "
          f"per_step={CALIBRATED_PER_STEP_SEC} max_hold={CALIBRATED_MAX_HOLD_SEC}) 実行中...")
    after = _capture_frames(
        video_stem, start_sec, max_sec,
        chain_hold_base_sec=CALIBRATED_BASE_SEC,
        chain_hold_per_step_sec=CALIBRATED_PER_STEP_SEC,
        chain_max_hold_sec=CALIBRATED_MAX_HOLD_SEC,
    )[side]
    return before, after


def _plot_state_timeline(
    video_stem: str, side: str,
    before: list[_FrameRecord], after: list[_FrameRecord],
    out_path: Path,
) -> None:
    """state 系列 (CHAIN/GRAVITY_SETTLE/STABLE 等) を before/after 2 段で描画する。"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    for ax, records, label in (
        (axes[0], before, "before (現行 0.3s/連鎖)"),
        (axes[1], after, f"after (較正 base={CALIBRATED_BASE_SEC}s+"
                          f"{CALIBRATED_PER_STEP_SEC}s×連鎖数, max_hold="
                          f"{CALIBRATED_MAX_HOLD_SEC}s)"),
    ):
        for rec in records:
            color = STATE_COLORS.get(rec.state, DEFAULT_STATE_COLOR)
            ax.axvspan(rec.t_sec, rec.t_sec + 1.0 / 60.0, color=color, lw=0)
        for idx in _new_chain_triggers(records):
            ax.axvline(records[idx].t_sec, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_ylabel(label, fontsize=9)
        ax.set_yticks([])
    axes[1].set_xlabel("time (sec)")
    handles = [
        mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8)
    fig.suptitle(
        f"A0 CHAIN保持時間較正 state タイムライン: {video_stem} {side} "
        "(点線=連鎖trigger)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] 保存: {out_path}")


def _grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Board grid (int) を RGB 画像 (BOARD_ROWS, BOARD_COLS, 3) に変換する。"""
    rgb = np.ones((BOARD_ROWS, BOARD_COLS, 3), dtype=np.float64)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rgb[r, c] = CELL_COLOR_MAP.get(int(grid[r, c]), (0.5, 0.5, 0.5))
    return rgb


def _plot_confirmed_diff(
    video_stem: str, side: str, event_idx: int,
    trigger_t: float, expected: np.ndarray,
    actual_before: np.ndarray | None, actual_after: np.ndarray | None,
    out_path: Path,
) -> None:
    """CHAIN→STABLE 遷移直後の confirmed_board と期待値の diff を
    before/after 並べて描画する (不一致セルを赤枠で強調)。"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.2))
    titles = ("期待値 (ChainSimulator.final_board)", "before 実測 confirmed",
              "after 実測 confirmed")
    grids = (expected, actual_before, actual_after)
    for ax, title, grid in zip(axes, titles, grids):
        if grid is None:
            ax.set_title(f"{title}\n(取得不可)", fontsize=8)
            ax.axis("off")
            continue
        ax.imshow(_grid_to_rgb(grid))
        mismatch = grid != expected
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if mismatch[r, c]:
                    ax.add_patch(mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, fill=False,
                        edgecolor="red", linewidth=2.2,
                    ))
        n_mismatch = int(mismatch.sum())
        ax.set_title(f"{title}\n不一致 {n_mismatch} セル", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{video_stem} {side} event#{event_idx} (trigger t={trigger_t:.2f}s) "
        "連鎖後confirmed diff (赤枠=不一致)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] 保存: {out_path}")


def _lag_and_diff_events(
    records: list[_FrameRecord], sim: ChainSimulator,
) -> list[dict]:
    """trigger 毎に (trigger_t, chain_count, first_stable_lag, expected, actual) を集める。"""
    out: list[dict] = []
    for idx in _new_chain_triggers(records):
        rec = records[idx]
        try:
            before_board = Board.from_list(rec.chain_before_grid.tolist())
            sim_result = sim.simulate(before_board)
        except Exception:
            continue
        if sim_result.chain_count < 1:
            continue
        first_stable_idx = _find_first_stable_after(records, idx)
        if first_stable_idx is None:
            continue
        out.append({
            "trigger_t": rec.chain_trigger_sec,
            "chain_count": sim_result.chain_count,
            "lag_sec": records[first_stable_idx].t_sec - rec.chain_trigger_sec,
            "expected": sim_result.final_board._grid,
            "actual": records[first_stable_idx].grid,
        })
    return out


def _plot_lag_summary(
    video_stem: str, side: str,
    before_events: list[dict], after_events: list[dict],
    out_path: Path,
) -> None:
    """イベント毎の STABLE確定ラグを before/after 棒グラフで比較する。"""
    n = max(len(before_events), len(after_events))
    if n == 0:
        print(f"[viz] {video_stem} {side}: 連鎖イベントなし、lag_summary skip")
        return
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 4))
    x = np.arange(n)
    width = 0.35
    before_lags = [e["lag_sec"] for e in before_events] + [0.0] * (n - len(before_events))
    after_lags = [e["lag_sec"] for e in after_events] + [0.0] * (n - len(after_events))
    ax.bar(x - width / 2, before_lags, width, label="before (0.3s/連鎖)", color="#7fa8d9")
    ax.bar(x + width / 2, after_lags, width, label="after (較正値)", color="#d97f7f")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ev{i}" for i in range(n)])
    ax.set_ylabel("STABLE確定ラグ (秒)")
    ax.set_title(f"{video_stem} {side}: 連鎖発火〜最初のSTABLEまでの遅延 (低いほど速い確定)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] 保存: {out_path}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-stem", default="c62")
    ap.add_argument("--start-sec", type=float, default=895.0)
    ap.add_argument("--max-sec", type=float, default=65.0)
    ap.add_argument("--side", default="1P")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = ChainSimulator()

    before, after = _run_both(
        args.video_stem, args.start_sec, args.max_sec, side=args.side,
    )

    _plot_state_timeline(
        args.video_stem, args.side, before, after,
        OUTPUT_DIR / f"state_timeline_{args.video_stem}_{args.side}.png",
    )

    before_events = _lag_and_diff_events(before, sim)
    after_events = _lag_and_diff_events(after, sim)

    _plot_lag_summary(
        args.video_stem, args.side, before_events, after_events,
        OUTPUT_DIR / f"lag_summary_{args.video_stem}_{args.side}.png",
    )

    n_events = max(len(before_events), len(after_events))
    for i in range(n_events):
        b = before_events[i] if i < len(before_events) else None
        a = after_events[i] if i < len(after_events) else None
        expected = (b or a)["expected"]
        trigger_t = (b or a)["trigger_t"]
        _plot_confirmed_diff(
            args.video_stem, args.side, i, trigger_t, expected,
            b["actual"] if b is not None else None,
            a["actual"] if a is not None else None,
            OUTPUT_DIR / f"confirmed_diff_{args.video_stem}_{args.side}_event{i}.png",
        )

    print(f"\n[DONE] 出力先: {OUTPUT_DIR}")
    print(f"  before 連鎖イベント数: {len(before_events)} / after: {len(after_events)}")
    for i in range(n_events):
        b = before_events[i]["lag_sec"] if i < len(before_events) else None
        a = after_events[i]["lag_sec"] if i < len(after_events) else None
        print(f"  event#{i}: before_lag={b} after_lag={a}")


if __name__ == "__main__":
    main()
