"""真犯人 (dig_resistance の seed なし drop_ojama) の単体数値確定マイクロテスト。

trace (grid 付き) から実ゲームの盤面を復元し、同一盤面に対して
dig_resistance / ukeyasusa を N 回呼んで値の分布を出す。
さらに ChainSimulator.drop_ojama を seed=0 固定にモンキーパッチした場合に
分布が1点に潰れること (=乱数が唯一の揺れ源であること) を確認する。
本体コードは変更しない (パッチは本スクリプトのプロセス内のみ)。

使い方:
  python scripts/_diag_adv_nondet_microtest_2026-08-25.py trace_r3.jsonl 162.4 222.0 --n 300
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402


def hf(h: str) -> float:
    try:
        return float.fromhex(h)
    except ValueError:
        return float("nan")


def load_boards(trace: str, t_target: float) -> "tuple[Board, Board, float]":
    """snap.t_sec が t_target に最も近い score レコードの盤面ペアを復元する。"""
    best = None
    with open(trace, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] != "score" or "grid1" not in rec:
                continue
            ts = hf(rec["snap"].get("t_sec", "nan"))
            if best is None or abs(ts - t_target) < abs(best[0] - t_target):
                best = (ts, rec)
    ts, rec = best
    shape = tuple(rec["grid_shape"])
    dtype = np.dtype(rec["grid_dtype"])
    boards = []
    for key in ("grid1", "grid2"):
        arr = np.frombuffer(
            base64.b64decode(rec[key]), dtype=dtype).reshape(shape).copy()
        b = Board()
        b._grid = arr.astype(np.uint8)
        boards.append(b)
    return boards[0], boards[1], ts


def dist(fn, board: Board, n: int) -> Counter:
    c: Counter = Counter()
    for _ in range(n):
        c[round(fn(board).score, 6)] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("t_targets", nargs="+", type=float)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    for tt in a.t_targets:
        b1, b2, ts = load_boards(a.trace, tt)
        print(f"\n===== t={ts:.3f} (目標 {tt}) =====")
        for side, b in (("1P", b1), ("2P", b2)):
            fill = b.count_puyos()
            print(f"-- {side} (ぷよ数 {fill}/78) --")
            for name, fn in (("dig_resistance", iv.dig_resistance),
                             ("ukeyasusa", iv.ukeyasusa)):
                c = dist(fn, b, a.n)
                print(f"  {name}: {a.n}回中 {dict(sorted(c.items()))}")
    # 乱数固定パッチで分布が1点になることの確認 (drop_ojama の seed を強制)
    print("\n===== drop_ojama を seed=0 固定にパッチ (このプロセス内のみ) =====")
    orig = ChainSimulator.drop_ojama

    def patched(self, board, ojama_count, seed=None):  # noqa: ANN001
        return orig(self, board, ojama_count, seed=0)

    ChainSimulator.drop_ojama = patched
    try:
        for tt in a.t_targets:
            b1, b2, ts = load_boards(a.trace, tt)
            for side, b in (("1P", b1), ("2P", b2)):
                c1 = dist(iv.dig_resistance, b, 50)
                c2 = dist(iv.ukeyasusa, b, 50)
                print(f"  t={ts:.3f} {side}: dig={dict(c1)} ukey={dict(c2)}")
    finally:
        ChainSimulator.drop_ojama = orig


if __name__ == "__main__":
    main()
