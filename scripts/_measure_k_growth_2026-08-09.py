"""K を増やしたとき火力がどう伸びるかを実測し、外挿の成否を確かめる (2026-08-09).

user 提案:
> K4 まではできたとして、 それ以降は **ある程度ツモが平均化される**ことも
> 加味すると別の方法で返せるかどうかの確認をとる道もある。
> 例えば長い連鎖で K12 が必要だとして、 それを近似的に高速処理する方法

## 統計的な根拠
深く読むほど個々のツモ順は結果に効かなくなる (大数の法則)。 12 手も打てば
出る色はほぼ均等になり、 「何が来るか」より **「何個置けるか」** が支配的になる。
よって K が大きい領域では、 全探索でなく **平均と分散から確率を出す**近似が
成立しうる。

## 測ること
1. K=1..8 で到達火力 (お邪魔換算) の **平均と分散** がどう伸びるか
2. その伸びが **外挿できる形か** (線形 / 対数 / 平方根のどれに乗るか)
3. 分散が K とともにどう振る舞うか (正規近似が使えるか)
4. **空き容量** との関係 (置ける数が支配的なら、 容量で説明できるはず)

読み取り専用。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, Board  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import scripts.mc_counter_estimator as mc  # noqa: E402

OUT_TSV = _ROOT / "data" / "verify" / "k_growth_2026-08-09.tsv"
# 1 手あたりの時間 (時間予算 → 手数 の換算に使う)
SEC_PER_HAND: float = 0.733
N_ROLLOUTS: int = 40
N_BOARDS: int = 6


def _mk(rows: int, seed: int) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice([1, 2, 3, 4]))
    return Board.from_list(g)


def _free_cells(b: Board) -> int:
    return int(sum(
        1 for r in range(1, BOARD_ROWS) for c in range(BOARD_COLS)
        if b.get(r, c) == COLOR_EMPTY
    ))


def main() -> int:
    boards = [(_mk(rows, seed=rows * 7), rows) for rows in (3, 5, 7, 9, 10, 11)]
    rows_out = ["fill_rows\tfree_cells\tk\tmean_ojama\tstd_ojama\tp25\tp75\thands\tms"]
    print(f"{'積み':>4s} {'空き':>5s} {'K':>3s} {'平均':>8s} {'標準偏差':>9s} "
          f"{'p25':>7s} {'p75':>7s} {'実手数':>7s} {'時間':>8s}")
    print("-" * 68)
    data: dict[int, list[tuple[int, float, float]]] = {}
    for b, rows in boards:
        free = _free_cells(b)
        for k in (1, 2, 3, 4, 6, 8, 12):
            budget = k * SEC_PER_HAND
            t0 = time.perf_counter()
            d = mc.estimate_counter_distribution(
                b, budget, thresholds_ojama=(12.0,), n_rollouts=N_ROLLOUTS,
            )
            ms = (time.perf_counter() - t0) * 1000
            std = max(0.0, (d.p75 - d.p25) / 1.349)  # 四分位範囲から標準偏差を近似
            print(f"{rows:4d} {free:5d} {k:3d} {d.mean:8.1f} {std:9.1f} "
                  f"{d.p25:7.1f} {d.p75:7.1f} {d.mean_hands_used:7.1f} {ms:7.0f}ms")
            rows_out.append(
                f"{rows}\t{free}\t{k}\t{d.mean:.2f}\t{std:.2f}\t{d.p25:.2f}\t"
                f"{d.p75:.2f}\t{d.mean_hands_used:.2f}\t{ms:.0f}"
            )
            data.setdefault(rows, []).append((k, d.mean, std))
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(rows_out) + "\n", encoding="utf-8")

    print()
    print("=== 外挿できるか (K に対する伸び方の当てはまり R^2) ===")
    print(f"{'積み':>4s} {'線形':>8s} {'対数':>8s} {'平方根':>8s}  最良")
    for rows, series in data.items():
        ks = np.array([x[0] for x in series], dtype=float)
        ms_ = np.array([x[1] for x in series], dtype=float)
        if ms_.max() <= 0:
            print(f"{rows:4d}   (火力ゼロのため判定不可)")
            continue
        fits = {}
        for name, xs in (("線形", ks), ("対数", np.log(ks)), ("平方根", np.sqrt(ks))):
            A = np.vstack([xs, np.ones(len(xs))]).T
            coef, res, *_ = np.linalg.lstsq(A, ms_, rcond=None)
            pred = A @ coef
            ss_res = float(((ms_ - pred) ** 2).sum())
            ss_tot = float(((ms_ - ms_.mean()) ** 2).sum())
            fits[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        best = max(fits, key=lambda k_: fits[k_])
        print(f"{rows:4d} {fits['線形']:8.3f} {fits['対数']:8.3f} "
              f"{fits['平方根']:8.3f}  {best}")
    print()
    print(f"出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
