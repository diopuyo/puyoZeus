"""実行間揺らぎが「判断メトリクス」をどれだけ動かすかの定量化 (読み込み専用)。

同一条件2 run の dump npz から、A/B判断に使われる代表メトリクスを各 run で
計算し、run間の食い違いを出す:

  1. 符号 (1P有利/2P有利) が run 間で食い違う行数 (adv_raw / adv_ema)
  2. EVEN判定 (|adv|<=3.0) の分類が食い違う行数
  3. 急変イベント数 (|adv_ema(t)-adv_ema(t-0.5s)| >= 30 / 50、定義明示のproxy)
  4. ±100張り付き率 (|adv_ema|>=99.5)
  5. 決着方向 (各gameの最終行 adv_ema の符号) の食い違い

使い方:
  python scripts/_diag_adv_nondet_decision_impact_2026-08-25.py A.npz B.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EVEN_TH = 3.0
JUMP_WINDOW_SEC = 0.5
JUMP_THRESHOLDS = (30.0, 50.0)
PIN_TH = 99.5


def _jump_count(t: np.ndarray, v: np.ndarray, gi: np.ndarray, th: float) -> int:
    """|v(t)-v(t-0.5s)| >= th のイベント数 (game境界をまたがない、連続超過は1件)。"""
    n = 0
    in_event = False
    for g in sorted(set(gi.tolist())):
        m = gi == g
        tg, vg = t[m], v[m]
        # 0.5秒前の値: 線形探索 (行間隔ほぼ一定なので単純に)
        j = 0
        in_event = False
        for i in range(len(tg)):
            while tg[i] - tg[j] > JUMP_WINDOW_SEC:
                j += 1
            jumped = abs(vg[i] - vg[j]) >= th
            if jumped and not in_event:
                n += 1
            in_event = jumped
    return n


def main() -> None:
    pa, pb = sys.argv[1], sys.argv[2]
    a = np.load(pa, allow_pickle=True)
    b = np.load(pb, allow_pickle=True)
    t = a["t_sec"]
    gi = a["game_idx"]
    n_rows = len(t)
    print(f"A={Path(pa).name}  B={Path(pb).name}  母数={n_rows}行")
    for key in ("adv_raw", "adv_ema"):
        va, vb = a[key].astype(float), b[key].astype(float)
        sign_diff = int((np.sign(va) != np.sign(vb)).sum())
        cls_a = np.where(np.abs(va) <= EVEN_TH, 0, np.sign(va))
        cls_b = np.where(np.abs(vb) <= EVEN_TH, 0, np.sign(vb))
        cls_diff = int((cls_a != cls_b).sum())
        print(f"\n[{key}]")
        print(f"  符号食い違い: {sign_diff}/{n_rows} 行")
        print(f"  EVEN込み3値分類の食い違い: {cls_diff}/{n_rows} 行")
        for th in JUMP_THRESHOLDS:
            ja = _jump_count(t, va, gi, th)
            jb = _jump_count(t, vb, gi, th)
            print(f"  急変(0.5秒で>={th:.0f}): A={ja}回 B={jb}回 差={ja-jb:+d}")
        pin_a = float((np.abs(va) >= PIN_TH).mean() * 100)
        pin_b = float((np.abs(vb) >= PIN_TH).mean() * 100)
        print(f"  ±100張り付き率(>= {PIN_TH}): A={pin_a:.2f}% B={pin_b:.2f}%")
        # 決着方向: 各gameの最終行の符号
        for g in sorted(set(gi.tolist())):
            m = gi == g
            sa, sb = np.sign(va[m][-1]), np.sign(vb[m][-1])
            if sa != sb:
                print(f"  [決着方向食い違い] game{g}: A={sa:+.0f} B={sb:+.0f}")


if __name__ == "__main__":
    main()
