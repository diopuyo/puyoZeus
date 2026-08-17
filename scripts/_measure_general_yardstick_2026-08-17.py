"""一般分布ラベルセット (55盤面) の突合測定 (ラベル規約v2、2026-08-17)。

## 目的
`data/verify/board_labels_general_2026-08-17/labels.tsv` (user記入済み) を
正解として、任意の認識構成で収集した npz の一般分布正解率を測る。

## ラベル規約v2による突合方式 (W8根治が実際に効く形)
1. `anchor_sig_path` (主キー) の中心フレームシグネチャを参照画像として、
   測定対象動画の `t_sec` (補助キー、参考値) 付近を NCC 再走査し、真の
   時刻/フレームを再特定する (`_label_anchor_lib_2026-08-17.
   reanchor_by_signature`)。動画が再DLされて内容が変わっていても機能する。
2. 再特定した時刻の ±`NEAREST_MATCH_TOLERANCE_SEC` 以内で最近傍の
   STABLE snapshot を測定対象 npz から取得する。
3. NCC が `NCC_CONFIDENT_THRESHOLD` 未満なら `anchor_unresolved` として
   明示的に除外する (黙って落とさない)。
4. 最近傍フォールバックを使った行には **`fallback_used=True`** の注記を
   必ず付ける (2026-08-17発見の34セル偽陽性教訓、frame_idx完全一致より
   時刻ずれの影響を受けやすいため)。

## 使い方 (未実行、userラベル完成後)
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_general_yardstick_2026-08-17 \\
        --candidate-npz-dir data/verify/board_labels_v4F_yardstick_2026-08-17 \\
        --candidate-video-dir "$HOME/frames"
"""
from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
_LIB = importlib.import_module("scripts._label_anchor_lib_2026-08-17")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

LABELS_DIR: Path = Path("data/verify/board_labels_general_2026-08-17")
LABELS_TSV: Path = LABELS_DIR / "labels.tsv"
VISIBLE_ROWS: range = range(1, 13)  # row1-12 (row0隠し段は対象外、既存規約と統一)
BOARD_COLS: int = 6
OK_MARKER: str = "ok"

# 再アンカリング探索窓・NCC閾値は既存資産 (_reanchor_yardstick_labels_2026-08-14) と統一
SEARCH_WINDOW_SEC: float = _LIB._RA.SEARCH_WINDOW_SEC
NCC_CONFIDENT_THRESHOLD: float = _LIB._RA.NCC_CONFIDENT_THRESHOLD
# 再アンカリング後、npz側の最近傍snapshotを探す許容窓 (W8(c)の34セル偽陽性教訓により
# 小さめに保つ。境界を跨ぐ場合は fallback_used フラグで必ず明示する)
NEAREST_MATCH_TOLERANCE_SEC: float = 0.35


@dataclass(frozen=True)
class LabelRow:
    """labels.tsv 1行分 (ラベル規約v2形式)。"""

    sheet: str
    anchor_hash: str
    anchor_sig_path: Path
    video: str
    side: str
    frame_idx: int
    t_sec: float
    wrong_cells: "dict[tuple[int, int], int]"


@dataclass
class MatchResult:
    """1盤面分の突合結果 (fail-silent回避のため状態を明示区別する)。"""

    row: LabelRow
    status: str  # "matched" / "anchor_unresolved" / "no_candidate_video" / "no_snapshot"
    n_errors: "int | None"
    fallback_used: bool
    ncc_score: float


# =============================================================================
# 1. labels.tsv 読込
# =============================================================================


def _parse_wrong_cells(spec: str) -> "dict[tuple[int, int], int]":
    spec = spec.strip()
    if spec == "" or spec.lower() == OK_MARKER:
        return {}
    out: "dict[tuple[int, int], int]" = {}
    for token in spec.split(","):
        m = re.match(r"r(\d+)c(\d+)=(\d+)", token.strip())
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    return out


def load_label_rows() -> "list[LabelRow]":
    """labels.tsv (ラベル規約v2形式) を読み込む。"""
    lines = [l for l in LABELS_TSV.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    rows: "list[LabelRow]" = []
    for r in csv.DictReader(lines, delimiter="\t"):
        rows.append(LabelRow(
            r["sheet"], r["anchor_hash"], Path(r["anchor_sig_path"]), r["video"],
            r["side"], int(r["frame_idx"]), float(r["t_sec"]),
            _parse_wrong_cells(r["wrong_cells"]),
        ))
    return rows


# =============================================================================
# 2. 1盤面の突合 (再アンカリング → 最近傍snapshot → セルdiff)
# =============================================================================


def _video_path(row: LabelRow, video_dir: Path) -> Path:
    stem = row.video if row.video.startswith("video_") else f"video_{row.video}"
    return video_dir / f"{stem}.mp4"


def match_one_row(row: LabelRow, candidate_npz_dir: Path, candidate_video_dir: Path) -> MatchResult:
    """1行分: 再アンカリング → 候補npzから最近傍snapshot取得 → セルdiff。"""
    key, candidate_grid = _LIB.load_anchor_sidecar(row.anchor_sig_path)
    correct = candidate_grid.copy() if candidate_grid is not None else None
    if correct is None:
        return MatchResult(row, "no_baseline_grid", None, False, float("nan"))
    for (r, c), v in row.wrong_cells.items():
        correct[r, c] = v
    video_path = _video_path(row, candidate_video_dir)
    if not video_path.exists():
        return MatchResult(row, "no_candidate_video", None, False, float("nan"))
    best_t, best_score = _LIB.reanchor_by_signature(
        video_path, _LIB._RA.region_for_side(row.side), key, row.t_sec, SEARCH_WINDOW_SEC,
    )
    if best_score < NCC_CONFIDENT_THRESHOLD:
        return MatchResult(row, "anchor_unresolved", None, False, best_score)
    idx = _MC._load_npz_index(_find_npz_for_video(candidate_npz_dir, row.video))
    if idx is None:
        return MatchResult(row, "no_candidate_video", None, False, best_score)
    exact = _MC._find_by_frame_idx_exact(idx, row.side, row.frame_idx)
    fallback_used = exact is None
    grid = exact[0] if exact is not None else _nearest_grid(idx, row.side, best_t)
    if grid is None:
        return MatchResult(row, "no_snapshot", None, fallback_used, best_score)
    n_err = sum(
        1 for r in VISIBLE_ROWS for c in range(BOARD_COLS)
        if int(correct[r, c]) != int(grid[r, c])
    )
    return MatchResult(row, "matched", n_err, fallback_used, best_score)


def _find_npz_for_video(npz_dir: Path, video: str) -> Path:
    """{video}_g*.npz または {video}.npz のいずれかを返す (無ければ存在しないPathを返す)。"""
    stem = video.replace("video_", "")
    matches = sorted(npz_dir.glob(f"{stem}_g*.npz")) + sorted(npz_dir.glob(f"{stem}.npz"))
    return matches[0] if matches else npz_dir / f"{stem}__missing__.npz"


def _nearest_grid(idx: "object", side: str, t_sec: float) -> "np.ndarray | None":
    """side一致かつ最近傍のsnapshot (許容窓外はNone)。"""
    cand = np.where(idx.sides == side)[0]
    if len(cand) == 0:
        return None
    dt = np.abs(idx.t_secs[cand] - t_sec)
    best = int(np.argmin(dt))
    if float(dt[best]) > NEAREST_MATCH_TOLERANCE_SEC:
        return None
    return idx.grids[cand[best]]


# =============================================================================
# 3. 集計 + レポート
# =============================================================================


def build_report(results: "list[MatchResult]") -> str:
    """55盤面の正解率レポート (fallback_used件数・未解決件数を必ず明示)。"""
    matched = [r for r in results if r.status == "matched"]
    n_cells = len(matched) * len(VISIBLE_ROWS) * BOARD_COLS
    n_err = sum(r.n_errors for r in matched)
    acc = (n_cells - n_err) / n_cells * 100 if n_cells else 0.0
    n_fallback = sum(1 for r in matched if r.fallback_used)
    lines = [
        f"突合: {len(matched)}/{len(results)} 盤面",
        f"セル正解率: {n_cells - n_err}/{n_cells} = {acc:.4f}%",
        f"最近傍フォールバック使用: {n_fallback}/{len(matched)} 盤面 "
        f"(注記: おじゃま一括着弾境界を跨ぐと偽陽性化しうる、W8(c)参照)",
    ]
    for status in ("anchor_unresolved", "no_candidate_video", "no_snapshot", "no_baseline_grid"):
        n = sum(1 for r in results if r.status == status)
        if n:
            lines.append(f"  {status}: {n}件")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-npz-dir", type=Path, required=True)
    ap.add_argument("--candidate-video-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_label_rows()
    print(f"[1/2] labels.tsv読込: {len(rows)}盤面")
    results = [match_one_row(r, args.candidate_npz_dir, args.candidate_video_dir) for r in rows]
    print("[2/2] " + build_report(results))


if __name__ == "__main__":
    main()
