"""再アンカリング後の物差し精度再計算 (タスク#5、2026-08-14)。

`_reanchor_yardstick_labels_2026-08-14.py` の結果 (`reanchor_table.tsv`) を
読み、`decision == "anchored"` の行 **のみ** を使って task5 3構成
(a_baseline/b_smw/c_ojamafall) のセル正解率を再計算する。

**52盤面3744セルのうち、再アンカリングで真に対応が取れたのは c13 の
6盤面432セルのみ**(他13動画・46盤面は ±10秒探索で明確な一致が無く
「アンカー不能」)。よって本測定は 432 セル規模の限定測定であり、
52盤面規模の代替にはならない。この限界を隠さず報告する
(feedback_viz_eval_required / fail-silent警戒)。

正解グリッドは `_measure_yardstick_v4_2026-08-05.py` の
`_reconstruct_correct_grid` (baseline npz + wrong_cells上書き) をそのまま
再利用する (コピペ禁止規約)。v4側の突合は `frame_idx` 完全一致 → 失敗時は
再アンカリングされた `best_t_redl` を中心にした最近傍時刻 (許容
NEAREST_MATCH_TOLERANCE_SEC 秒) のフォールバック。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_yardstick_task5_reanchored_2026-08-14
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MY = importlib.import_module("scripts._measure_yardstick_v4_2026-08-05")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")

REANCHOR_TABLE: Path = Path("data/verify/yardstick_reanchor_2026-08-14/reanchor_table.tsv")

TASK5_DIRS: "dict[str, Path]" = {
    "a_baseline": Path("data/verify/board_labels_task5_a_baseline_2026-08-14"),
    "b_smw": Path("data/verify/board_labels_task5_b_smw_2026-08-14"),
    "c_ojamafall": Path("data/verify/board_labels_task5_c_ojamafall_2026-08-14"),
}

VISIBLE_ROWS = _MY.VISIBLE_ROWS
BOARD_COLS = _MY.BOARD_COLS

# ラベル瞬間の盤面を探す最近傍許容秒数 (_measure_yardstick_v4 と同値を採用)
NEAREST_MATCH_TOLERANCE_SEC: float = 0.35


@dataclass(frozen=True)
class AnchoredLabel:
    video_stem: str
    side: str
    frame_idx: int
    source_dir: str
    best_t_redl: float


def load_anchored_labels() -> "list[AnchoredLabel]":
    """reanchor_table.tsv から decision=='anchored' の行だけを読む。"""
    out: list[AnchoredLabel] = []
    with REANCHOR_TABLE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["decision"] != "anchored":
                continue
            out.append(AnchoredLabel(
                video_stem=row["video"], side=row["side"],
                frame_idx=int(row["frame_idx"]), source_dir=row["source_dir"],
                best_t_redl=float(row["best_t_redl"]),
            ))
    return out


def _to_yardstick_row(label: AnchoredLabel) -> "_MY.YardstickRow":
    """AnchoredLabel から正解再構成用の YardstickRow を組み立てる (wrong_cellsは元labels.tsvから再取得)。"""
    for row in _MY.load_yardstick_rows():
        if (row.video_stem == label.video_stem and row.side == label.side
                and row.frame_idx == label.frame_idx and row.source_dir == label.source_dir):
            return row
    raise ValueError(f"labels.tsv に該当行が見つからない: {label}")


def _find_v4_grid(npz_dir: Path, label: AnchoredLabel) -> "np.ndarray | None":
    """task5 npz から該当盤面を探す (frame_idx完全一致優先、失敗時は再アンカリング時刻の最近傍)。"""
    for npz_path in sorted(
        list(npz_dir.glob(f"{label.video_stem}_g*.npz"))
        + list(npz_dir.glob(f"{label.video_stem}.npz"))
    ):
        idx = _MC._load_npz_index(npz_path)
        if idx is None:
            continue
        match = _MC._find_by_frame_idx_exact(idx, label.side, label.frame_idx)
        if match is not None:
            return match[0]
    v4_grid = None
    for npz_path in sorted(
        list(npz_dir.glob(f"{label.video_stem}_g*.npz"))
        + list(npz_dir.glob(f"{label.video_stem}.npz"))
    ):
        idx = _MC._load_npz_index(npz_path)
        if idx is None:
            continue
        near = _MY._find_nearest_in_time(idx, label.side, label.best_t_redl)
        if near is not None:
            v4_grid = near
            break
    return v4_grid


def main() -> None:
    anchored = load_anchored_labels()
    print(f"[1/2] アンカー済ラベル: {len(anchored)}/52 (残りはアンカー不能で除外)")
    for tag, npz_dir in TASK5_DIRS.items():
        total_cells = 0
        total_errors = 0
        v4_missing = 0
        detail_lines: list[str] = []
        for label in anchored:
            row = _to_yardstick_row(label)
            rec = _MY._reconstruct_correct_grid(row, _MY.BASELINE_NPZ_DIR)
            if rec is None:
                raise RuntimeError(f"正解再構成失敗 (想定外、アンカー済のはず): {label}")
            correct_grid, _label_t = rec
            v4_grid = _find_v4_grid(npz_dir, label)
            if v4_grid is None:
                v4_missing += 1
                detail_lines.append(f"  [no_match] {label.video_stem} {label.side} f{label.frame_idx}")
                continue
            n_errors = 0
            for r in VISIBLE_ROWS:
                for c in range(BOARD_COLS):
                    if int(correct_grid[r, c]) != int(v4_grid[r, c]):
                        n_errors += 1
            total_cells += len(VISIBLE_ROWS) * BOARD_COLS
            total_errors += n_errors
            if n_errors:
                detail_lines.append(
                    f"  [{n_errors}セル誤り] {label.video_stem} {label.side} f{label.frame_idx}",
                )
        acc = (total_cells - total_errors) / total_cells * 100 if total_cells else float("nan")
        print(
            f"[{tag}] 突合={len(anchored) - v4_missing}/{len(anchored)} "
            f"(v4_missing={v4_missing}) セル正解率={total_cells - total_errors}/{total_cells}"
            f"={acc:.4f}%",
        )
        for line in detail_lines:
            print(line)


if __name__ == "__main__":
    main()
