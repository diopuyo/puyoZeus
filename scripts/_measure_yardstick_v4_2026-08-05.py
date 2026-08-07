"""物差し (盤面単位人手ラベル、52盤面3744セル) のv4構成再測定 (2026-08-05 準備)。

`project_yardstick_first_results` (2026-07-31/08-01) で確立した「盤面単位
人手ラベル」測定系を、バーストガード確定構成 v4 (Stage1.5+0.954+1.5b) で
再測定するための突合スクリプト。**旧labels.tsvは正解グリッドを直接持たず
"旧基準npz (boards_lean_allframes_ref_2026-07-30、v4以前=無burst-guard)
に対する wrong_cells 差分" として記録されている**ため、正解グリッド自体は
「旧基準npzのグリッド + wrong_cells上書き」で再構成する必要がある
(measure_effect_gate_c_2026-08-04.py の `correct_grid` 直接記録方式とは
データ構造が異なる点に注意、これが今回確認した最大の「罠」)。

## 突合対象
- `data/verify/board_labels_2026-07-31/labels.tsv` (v1、12盤面864セル)
- `data/verify/board_labels_2026-07-31_v3/labels.tsv` (v3、40盤面2880セル)
- 合計52盤面3744セル (v2は f03b0bd でサンプリング手法自体を修正した中間
  バッチのため最終集計には含めない、`project_yardstick_first_results` 準拠)

## 既存資産の再利用 (コピペ禁止指示への対応)
`scripts/measure_effect_gate_c_2026-08-04.py` (importlib動的import) の
`_load_npz_index` / `_find_by_frame_idx_exact` をそのまま再利用する。

## 使い方 (v4 npz 着弾後)
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_yardstick_v4_2026-08-05 \
        --v4-npz-dir data/verify/board_labels_v4_yardstick_2026-08-05

## 自己検証モード (今夜の準備、CPU負荷なし)
`--v4-npz-dir` に旧基準npzディレクトリそのものを渡すと、旧基準グリッドを
「v4扱い」で自分自身と突合することになり、誤り0/3744 (100%一致) になる
はず。これは新規収集不要でロジックの正しさを検証する自己検証であり、
v4の実測結果ではない (--self-check フラグで明示)。
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

# ファイル名にハイフンを含むため通常の `from ... import` は不可 (動的import)。
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

BASELINE_NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_allframes_ref_2026-07-30")
LABEL_DIRS: "tuple[Path, ...]" = (
    Path("data/verify/board_labels_2026-07-31"),
    Path("data/verify/board_labels_2026-07-31_v3"),
)
BOARD_ROWS: int = 13
BOARD_COLS: int = 6
# row0 (隠し段) はラベルシートに写っていない (ゲーム画面に表示されない) ため
# 人手ラベルの正解対象外。RESULTS.md 「1盤面=72セル」= row1-12 のみが対象
# (自己検証runで実際に確認: row0を含めた78セル基準では 52盤面3744セルの
# 歴史的数値=99.5%系に一致しない、含めない72セル基準で v1=99.54%/
# v3=99.51%と正確に一致した)。加えて v4 の Stage1.5b (§11) は row0 の
# infer_hidden_row を明示的に変更する設計なので、row0 を「誤り」として
# 数えると意図的な修正を誤判定してしまう二重の理由で除外必須。
VISIBLE_ROWS: range = range(1, BOARD_ROWS)  # row1-12 (72セル/盤面)
OK_MARKER: str = "ok"


# =============================================================================
# データ構造
# =============================================================================


@dataclass(frozen=True)
class YardstickRow:
    """labels.tsv 1行分。"""

    source_dir: str
    video_stem: str
    side: str
    frame_idx: int
    wrong_cells: "dict[tuple[int, int], int]"  # 空dict = 全セル正解(ok)


@dataclass
class BoardResult:
    """1盤面分の突合結果。"""

    row: YardstickRow
    baseline_found: bool
    v4_found: bool
    n_errors: "int | None"
    mismatched_cells: "list[tuple[int, int, int, int]]"  # (r,c,correct,v4)


# =============================================================================
# 1. labels.tsv 読込 (wrong_cells 記法 "r3c2=1,r5c0=0" をパース)
# =============================================================================


def _parse_wrong_cells(spec: str) -> "dict[tuple[int, int], int]":
    """"r3c2=1,r5c0=0" 形式をパースする ("ok"/空文字は誤りなし=空dict)。"""
    spec = spec.strip()
    if spec == "" or spec.lower() == OK_MARKER:
        return {}
    out: "dict[tuple[int, int], int]" = {}
    for token in spec.split(","):
        m = re.match(r"r(\d+)c(\d+)=(\d+)", token.strip())
        if not m:
            continue
        r, c, v = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out[(r, c)] = v
    return out


def load_yardstick_rows() -> list[YardstickRow]:
    """v1+v3 labels.tsv を統合して読み込む (v2は中間バッチのため対象外)。"""
    rows: list[YardstickRow] = []
    for label_dir in LABEL_DIRS:
        tsv = label_dir / "labels.tsv"
        lines = [l for l in tsv.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
        for r in csv.DictReader(lines, delimiter="\t"):
            rows.append(YardstickRow(
                source_dir=label_dir.name,
                video_stem=r["video"].replace("video_", ""),
                side=r["side"],
                frame_idx=int(r["frame"]),
                wrong_cells=_parse_wrong_cells(r["wrong_cells"]),
            ))
    return rows


# =============================================================================
# 2. 正解グリッド再構成 (旧基準npzグリッド + wrong_cells上書き)
# =============================================================================


def _reconstruct_correct_grid(
    row: YardstickRow, baseline_dir: Path,
) -> "tuple[np.ndarray, float] | None":
    """旧基準npzから該当frameのグリッドを取り、wrong_cellsで上書きして正解を作る。

    Returns:
        (正解グリッド, ラベル時点のt_sec)。t_sec はv4側の実効盤面フォールバック
        (frame_idx一致失敗時) のアンカーに使う。
    """
    for npz_path in sorted(baseline_dir.glob(f"{row.video_stem}_g*.npz")):
        idx = _MC._load_npz_index(npz_path)
        if idx is None:
            continue
        match = _MC._find_by_frame_idx_exact(idx, row.side, row.frame_idx)
        if match is None:
            continue
        grid, t_sec = match
        correct = grid.copy()
        for (r, c), v in row.wrong_cells.items():
            correct[r, c] = v
        return correct, float(t_sec)
    return None


# =============================================================================
# 3. v4突合 (同一frame_idxでの厳密一致、fpsやsample-intervalを変えていない前提)
# =============================================================================


# 最近傍突合の許容時刻差 (秒)。STABLEスナップショットは毎秒数枚あるため、
# ラベル瞬間の盤面は±この範囲に必ず存在するはず。これを超える最近傍は
# 「その瞬間の盤面が無い」= no_match として正直に落とす (実効盤面方式は
# ラベルが連鎖境界直後の場合に数秒前の盤面=別内容を掴む誤りがあり不採用、
# 2026-08-05実測: c15 2P f15646 で26セルの偽誤りを生成した)。
NEAREST_MATCH_TOLERANCE_SEC: float = 0.35


def _find_nearest_in_time(
    idx: "object", side: str, label_t: float,
) -> "np.ndarray | None":
    """ラベル時刻に最も近いスナップショットのグリッド (±許容内のみ) を返す。"""
    mask = (idx.sides == side)
    cand = np.where(mask)[0]
    if len(cand) == 0:
        return None
    dt = np.abs(idx.t_secs[cand] - label_t)
    best = int(np.argmin(dt))
    if float(dt[best]) > NEAREST_MATCH_TOLERANCE_SEC:
        return None
    return idx.grids[cand[best]]


def compare_one_row(row: YardstickRow, baseline_dir: Path, v4_dir: Path) -> BoardResult:
    """1盤面分: 正解再構成 → v4グリッド取得 → セル単位diff。

    v4側の突合は frame_idx 完全一致を優先し、失敗時は「ラベル時刻±0.35秒の
    最近傍スナップショット」にフォールバックする (2026-08-05追記: 窓付き収集は
    コールドスタートで確定タイミングの足並みが全編ずれ、exact一致がほぼ
    失敗する。最近傍±許容はその瞬間の実内容を測る)。
    """
    rec = _reconstruct_correct_grid(row, baseline_dir)
    if rec is None:
        return BoardResult(row, False, False, None, [])
    correct, label_t = rec
    v4_grid = None
    for npz_path in sorted(
        list(v4_dir.glob(f"{row.video_stem}_g*.npz"))
        + list(v4_dir.glob(f"{row.video_stem}.npz"))
    ):  # 窓付き ({stem}_gN.npz) と全編 ({stem}.npz) の両命名に対応
        idx = _MC._load_npz_index(npz_path)
        if idx is None:
            continue
        match = _MC._find_by_frame_idx_exact(idx, row.side, row.frame_idx)
        if match is not None:
            v4_grid = match[0]
            break
        near = _find_nearest_in_time(idx, row.side, label_t)
        if near is not None and v4_grid is None:
            v4_grid = near  # exact優先のためbreakしない (他npzのexactを探し続ける)
    if v4_grid is None:
        return BoardResult(row, True, False, None, [])
    mismatches = [
        (r, c, int(correct[r, c]), int(v4_grid[r, c]))
        for r in VISIBLE_ROWS for c in range(BOARD_COLS)
        if int(correct[r, c]) != int(v4_grid[r, c])
    ]
    return BoardResult(row, True, True, len(mismatches), mismatches)


# =============================================================================
# 4. 集計 + レポート
# =============================================================================


def build_report(results: list[BoardResult]) -> str:
    """52盤面3744セルの正解率レポート (fail-silent回避、no_match件数を必ず明示)。"""
    baseline_missing = sum(1 for r in results if not r.baseline_found)
    v4_missing = sum(1 for r in results if r.baseline_found and not r.v4_found)
    matched = [r for r in results if r.n_errors is not None]
    total_cells = len(matched) * len(VISIBLE_ROWS) * BOARD_COLS
    total_errors = sum(r.n_errors for r in matched)
    acc = (total_cells - total_errors) / total_cells * 100 if total_cells else 0.0
    lines = [
        f"突合盤面: {len(matched)}/{len(results)} "
        f"(baseline_missing={baseline_missing} / v4_missing={v4_missing})",
        f"セル正解率: {total_cells - total_errors}/{total_cells} = {acc:.4f}%",
    ]
    for r in matched:
        if r.n_errors:
            lines.append(
                f"  {r.row.video_stem} {r.row.side} f{r.row.frame_idx} "
                f"[{r.row.source_dir}]: {r.n_errors}セル誤り {r.mismatched_cells}"
            )
    return "\n".join(lines)


# =============================================================================
# 5. main
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v4-npz-dir", type=Path, required=True)
    ap.add_argument(
        "--self-check", action="store_true",
        help="v4-npz-dir に旧基準npzを渡した自己検証モードであることを明示する",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        print("[自己検証モード] v4扱いのnpzは旧基準そのもの → 誤り0/3744が期待値")
    rows = load_yardstick_rows()
    print(f"[1/2] labels.tsv読込: {len(rows)}盤面 (v1+v3統合)")
    results = [compare_one_row(r, BASELINE_NPZ_DIR, args.v4_npz_dir) for r in rows]
    print("[2/2] " + build_report(results))


if __name__ == "__main__":
    main()
