"""60fps全フレーム版とstride2版のSTABLE遷移タイミングのズレを診断する (2026-08-12)。

_compare_fps_stride_ab_2026-08-12.py で「対応なし(取りこぼし疑い)」が
601件中388件と多かった理由の追跡用。dedup済みSTABLE snapshotは実イベント
(設置/連鎖)駆動で発生間隔が不定なため、絶対時刻±0.05sでの対応判定は
「同一イベントをどれだけ早く/遅くSTABLE確定するか」のズレを過検出する。
イベント順序 (何番目のsnapshotか) で対応させ、時刻差の分布を見る。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SIDES = ("1P", "2P")


def main() -> None:
    """CLI 引数 (full_npz, stride_npz) でイベント順序ベースのズレを表示する。"""
    full_path, stride_path = Path(sys.argv[1]), Path(sys.argv[2])
    zf = np.load(full_path, allow_pickle=True)
    zs = np.load(stride_path, allow_pickle=True)
    for side in SIDES:
        mf = np.asarray(zf["side"]).astype(str) == side
        ms = np.asarray(zs["side"]).astype(str) == side
        tf = np.sort(np.asarray(zf["t_sec"], dtype=float)[mf])
        ts = np.sort(np.asarray(zs["t_sec"], dtype=float)[ms])
        print(f"=== side={side} full60fps件数={len(tf)} stride2件数={len(ts)} ===")
        n = min(len(tf), len(ts))
        lags = tf[:n] - ts[:n]
        print(
            f"  先頭{n}件(順序対応)の t差(full-stride2): "
            f"中央値{np.median(lags):+.3f}s 平均{np.mean(lags):+.3f}s "
            f"最小{lags.min():+.3f}s 最大{lags.max():+.3f}s"
        )
        print(f"  full60fps先頭5件: {[round(float(x), 3) for x in tf[:5]]}")
        print(f"  stride2先頭5件  : {[round(float(x), 3) for x in ts[:5]]}")
        print(f"  full60fps末尾5件: {[round(float(x), 3) for x in tf[-5:]]}")
        print(f"  stride2末尾5件  : {[round(float(x), 3) for x in ts[-5:]]}")


if __name__ == "__main__":
    main()
