"""外部正解の物差し: 人手ラベル候補の規模を測る (2026-07-31)。

## なぜ物差しが必要か

看板指標「セル正解率95.4%」は `measure_stable_cell_acc.py` が
raw_cnn / raw_hsv / confirmed の**3者多数決を正解とする自己無矛盾性チェック**で、
**CNN と HSV が同じ誤りに合意すると原理的に検出できない**
(スクリプト自身の docstring が fail-silent を明記している)。
→ 「認識精度99.99%」という目標が、それを測れない指標の上に立っている。

## なぜランダム抽出ではダメか

99.99% を検証するには数千ラベルが必要で、しかも**盲点 (CNN と HSV が同じ誤りに
合意する箇所) はランダム抽出では滅多に当たらない**。人手ラベルは user の時間なので
1 ラベルあたりの情報量を最大化する必要がある。

## 方針: CNN/HSV の両方から独立した判定材料で盲点を狙い撃つ

**物理則**がそれ。浮きぷよ (空セルの上にぷよ) は物理的に不可能なので、
**CNN と HSV が何に合意していようと少なくとも一方のセルが誤り**と確定する。
これは cnn/hsv の一致・不一致とは独立な信号なので、相関する誤りを捕まえられる。

本スクリプトは基準データ (全フレーム収集) について、
以下の類型の候補件数を数えて**ラベル付けの規模を見積もる**:

  1. 浮きぷよ (空セルの上に色ぷよ) — 物理的に不可能
  2. 隠し段以外での「空セルの上におじゃま」— 同じく不可能
  3. 前後フレームで色が変化 (空を経由せず色→別色) — 連鎖なしでは不可能

いずれも**「少なくとも1セルが誤り」を確定できる**ので、
人手ラベルは「どちらが誤りか」の判定だけで済み、1件あたりの情報量が高い。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_yardstick_candidates_2026-07-31 \
        --limit 40
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

# 盤面の形状 (13行 x 6列。row 0 は隠し段)
BOARD_ROWS: int = 13
BOARD_COLS: int = 6
HIDDEN_ROWS: int = 1
# 色コード
COLOR_EMPTY: int = 0
COLOR_OJAMA: int = 9
COLOR_UNKNOWN: int = 10
# 有効な色ぷよ
VALID_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})


def _floating_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """浮きぷよ (直下が空なのにぷよがある) セルを返す。

    隠し段 (row 0) は画面外で推論値なので対象外にする。
    最下段は直下が無いので対象外。
    """
    out: list[tuple[int, int]] = []
    for c in range(BOARD_COLS):
        for r in range(HIDDEN_ROWS, BOARD_ROWS - 1):
            v = int(grid[r, c])
            below = int(grid[r + 1, c])
            if v in VALID_COLORS or v == COLOR_OJAMA:
                if below == COLOR_EMPTY:
                    out.append((r, c))
    return out


def _color_to_color_changes(
    prev: np.ndarray, cur: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """空を経由せず色→別色に変化したセルを返す (連鎖なしでは不可能)。

    Returns:
        (row, col, 前の色, 後の色) のリスト。
    """
    out: list[tuple[int, int, int, int]] = []
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            a, b = int(prev[r, c]), int(cur[r, c])
            if a == b:
                continue
            if a in VALID_COLORS and b in VALID_COLORS:
                out.append((r, c, a, b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz-dir", type=Path,
        default=Path("data/indicators_v2/boards_lean_allframes_ref_2026-07-30"),
    )
    ap.add_argument("--limit", type=int, default=40, help="先頭N件のnpz (0=全件)")
    args = ap.parse_args()

    files = sorted(args.npz_dir.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"npz が無い: {args.npz_dir}")
        return
    print(f"対象 {len(files)} npz ({args.npz_dir.name})\n")

    n_rows = 0
    n_float_rows = 0
    float_cells = 0
    float_by_video: Counter = Counter()
    n_c2c_rows = 0
    c2c_cells = 0
    unknown_cells = 0
    total_cells = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        grids = np.asarray(d["grids"])
        sides = np.asarray(d["side"]).astype(str)
        vids = np.asarray(d["video_id"]).astype(str)
        n_rows += len(grids)
        total_cells += len(grids) * (BOARD_ROWS - HIDDEN_ROWS) * BOARD_COLS
        # 浮きぷよ
        prev_by_side: dict[str, np.ndarray] = {}
        for i in range(len(grids)):
            g = grids[i]
            fl = _floating_cells(g)
            if fl:
                n_float_rows += 1
                float_cells += len(fl)
                float_by_video[str(vids[i])] += len(fl)
            unknown_cells += int(
                (g[HIDDEN_ROWS:, :] == COLOR_UNKNOWN).sum(),
            )
            side = str(sides[i])
            pv = prev_by_side.get(side)
            if pv is not None:
                ch = _color_to_color_changes(pv, g)
                if ch:
                    n_c2c_rows += 1
                    c2c_cells += len(ch)
            prev_by_side[side] = g

    print(f"{'項目':<40}{'値':>14}")
    print("-" * 54)
    print(f"{'STABLE snapshot 行数':<40}{n_rows:>14}")
    print(f"{'判定対象セル総数 (隠し段除く)':<40}{total_cells:>14}")
    print(f"{'UNKNOWN セル':<40}{unknown_cells:>14}"
          f" ({100.0 * unknown_cells / max(1, total_cells):.3f}%)")
    print()
    print(f"{'★浮きぷよを含む行':<40}{n_float_rows:>14}"
          f" ({100.0 * n_float_rows / max(1, n_rows):.2f}%)")
    print(f"{'★浮きぷよセル総数':<40}{float_cells:>14}"
          f" ({100.0 * float_cells / max(1, total_cells):.3f}%)")
    print(f"{'★色→別色 変化を含む行':<40}{n_c2c_rows:>14}"
          f" ({100.0 * n_c2c_rows / max(1, n_rows):.2f}%)")
    print(f"{'★色→別色 変化セル総数':<40}{c2c_cells:>14}")
    print()
    print("浮きぷよが多い動画 上位:")
    for v, n in float_by_video.most_common(8):
        print(f"  {v:<16}{n:>8}")
    print(
        "\n→ これらは **CNN/HSV が何に合意していようと少なくとも1セルが誤り**。"
        "\n→ 人手ラベルは「どちらが誤りか」の判定だけで済むので情報量が高い。"
        "\n→ 件数が多すぎる場合は層別サンプリングで規模を絞る。"
    )


if __name__ == "__main__":
    main()
