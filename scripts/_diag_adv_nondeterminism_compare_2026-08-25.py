"""有利不利スコアの実行間非決定性 — dump npz ペア突合診断 (読み込み専用)。

同一コード・同一フラグで生成した timeline dump npz を2つ突合し、
  - キー別の不一致行数 / 母数 / 最大差
  - adv_raw 不一致行の時間分布・game別分布・連発長 (キャッシュ持続)
  - 不一致行の直前 HeavyAdvCache 更新境界 (every=9) との位相関係
を出す。コードは変更しない (計装は本スクリプトのみ)。

使い方:
  python scripts/_diag_adv_nondeterminism_compare_2026-08-25.py A.npz B.npz [--csv out.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 連続区間 (開始index, 長さ) のリスト。"""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    a = np.load(args.a, allow_pickle=True)
    b = np.load(args.b, allow_pickle=True)
    keys = sorted(set(a.files) & set(b.files))
    only_a = set(a.files) - set(b.files)
    only_b = set(b.files) - set(a.files)
    if only_a or only_b:
        print(f"[キー差] only_a={sorted(only_a)} only_b={sorted(only_b)}")
    n_rows = None
    print(f"=== {Path(args.a).name} vs {Path(args.b).name} ===")
    identical_keys = []
    for k in keys:
        va, vb = a[k], b[k]
        if va.shape != vb.shape:
            print(f"[shape不一致] {k}: {va.shape} != {vb.shape}")
            continue
        n_rows = va.shape[0] if va.ndim else 1
        if va.dtype.kind in "fiub" and vb.dtype.kind in "fiub":
            diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
            n_ne = int((diff > 0).sum())
            if n_ne:
                print(f"[不一致] {k}: {n_ne}/{diff.size} 行  最大差 {diff.max():.6g}")
            else:
                identical_keys.append(k)
        else:
            ne = np.array([str(x) != str(y) for x, y in zip(va.ravel(), vb.ravel())])
            n_ne = int(ne.sum())
            if n_ne:
                print(f"[不一致] {k}: {n_ne}/{va.size} 要素 (文字列)")
            else:
                identical_keys.append(k)
    print(f"[一致キー] {len(identical_keys)}/{len(keys)}: {identical_keys}")

    if "adv_raw" not in keys:
        return
    da = a["adv_raw"].astype(np.float64)
    db = b["adv_raw"].astype(np.float64)
    ne = np.abs(da - db) > 0
    t = a["t_sec"] if "t_sec" in keys else np.arange(len(da))
    gi = a["game_idx"] if "game_idx" in keys else np.zeros(len(da), int)
    print(f"\n=== adv_raw 詳細 ===  不一致 {int(ne.sum())}/{len(da)} 行 "
          f"最大差 {np.abs(da-db).max():.4f}")
    if ne.sum():
        d = np.abs(da - db)[ne]
        print(f"差分分布: p50={np.percentile(d,50):.4f} p90={np.percentile(d,90):.4f} "
              f"p99={np.percentile(d,99):.4f} max={d.max():.4f}")
        for g in sorted(set(gi[ne].tolist())):
            m = ne & (gi == g)
            print(f"  game{g}: {int(m.sum())}行  t=[{t[m].min():.2f}, {t[m].max():.2f}]")
        runs = _runs(ne)
        lens = [ln for _, ln in runs]
        print(f"連続区間: {len(runs)}個  長さ p50={np.percentile(lens,50):.1f} "
              f"max={max(lens)} (行)")
        # 先頭数区間を表示
        for i0, ln in runs[:12]:
            print(f"    行{i0} t={t[i0]:.3f} g{gi[i0]} 長さ{ln} "
                  f"advA={da[i0]:+.3f} advB={db[i0]:+.3f} 差{abs(da[i0]-db[i0]):.3f}")
    if args.csv and ne.sum():
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "t_sec", "game_idx", "adv_a", "adv_b", "absdiff"])
            for i in np.where(ne)[0]:
                w.writerow([int(i), float(t[i]), int(gi[i]),
                            float(da[i]), float(db[i]), float(abs(da[i]-db[i]))])
        print(f"[保存] {args.csv}")


if __name__ == "__main__":
    main()
