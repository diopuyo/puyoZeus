"""Gate 3R-6 案A の必須測定 (読み込み専用・比較のみ)。

  [1] bit-identical 証明: first5games_planA_off.npz (実装後・フラグOFF) の
      全配列が 既存 first5games_on.npz (実装前・同一構成) と完全一致するか
  [2] ON の副作用範囲: first5games_planA_on.npz と OFF の差分が
      is_dead1/is_dead2 のみか。adv_raw / adv_ema / p1 の最大差を数値で出す
  [3] 真の窒息の維持 (最重要): 2P 実試合2 (t=221.867-223.4、おじゃま満杯) と
      2P 実試合4 (t=324.3-325.567) の各行の is_dead2 を OFF/ON 並記
  [4] 誤判定 run (1P 実試合2 t=164.033-164.733) の OFF/ON 並記
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

BASE = Path("data/verify/gate3r6_diag_2026-08-25/first5games_on.npz")
OFF = Path("data/verify/gate3r6_planA_2026-08-25/first5games_planA_off.npz")
ON = Path("data/verify/gate3r6_planA_2026-08-25/first5games_planA_on.npz")

# 真の窒息 run / 誤判定 run の実測窓 (isdead_runs_on_games1to5.csv より)
WINDOWS = [
    ("[3] 真の窒息 2P g2 (見逃したら不合格)", "2", 221.6, 223.6),
    ("[3] 真の窒息 2P g4 (終端 OJAMA_FALL)", "2", 324.0, 325.8),
    ("[4] 誤判定 1P g2 (残存 0.7s)", "1", 163.8, 165.2),
]


def compare_all_keys(a_path: Path, b_path: Path, label: str) -> None:
    """2つの dump npz の全キーを突合し、不一致キーと数値差を表示する。"""
    a = np.load(a_path, allow_pickle=True)
    b = np.load(b_path, allow_pickle=True)
    keys_a, keys_b = set(a.files), set(b.files)
    print(f"=== {label} ===")
    if keys_a != keys_b:
        print(f"  キー集合が不一致: only_a={keys_a - keys_b} only_b={keys_b - keys_a}")
    mismatch = 0
    for k in sorted(keys_a & keys_b):
        va, vb = a[k], b[k]
        if va.shape != vb.shape:
            print(f"  [不一致] {k}: shape {va.shape} != {vb.shape}")
            mismatch += 1
            continue
        if va.dtype.kind in "fiu" and vb.dtype.kind in "fiu":
            diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
            if diff.size and diff.max() > 0:
                print(f"  [不一致] {k}: 差のある行 {(diff > 0).sum()}/{diff.size}"
                      f" 最大差 {diff.max():.6g}")
                mismatch += 1
        else:
            neq = sum(1 for x, y in zip(va.ravel(), vb.ravel()) if x != y)
            if neq:
                print(f"  [不一致] {k}: 不一致要素 {neq}/{va.size}")
                mismatch += 1
    n_keys = len(keys_a & keys_b)
    print(f"  -> 不一致キー {mismatch}/{n_keys} (0/{n_keys} なら完全一致)")


def show_windows(off_path: Path, on_path: Path) -> None:
    """真の窒息/誤判定の各窓で is_dead を OFF/ON 並記する。"""
    off = np.load(off_path, allow_pickle=True)
    on = np.load(on_path, allow_pickle=True)
    t_off, t_on = off["t_sec"], on["t_sec"]
    for label, side, lo, hi in WINDOWS:
        print(f"=== {label} (t=[{lo},{hi}]) ===")
        m_off = (t_off >= lo) & (t_off <= hi)
        m_on = (t_on >= lo) & (t_on <= hi)
        if m_off.sum() != m_on.sum():
            print(f"  行数不一致 off={m_off.sum()} on={m_on.sum()} (要調査)")
        rows = zip(t_on[m_on], on["game_idx"][m_on],
                   off[f"is_dead{side}"][m_off], on[f"is_dead{side}"][m_on],
                   on[f"state{side}"][m_on])
        for tt, gi, d_off, d_on, st in rows:
            mark = "" if d_off == d_on else "  <- 変化"
            print(f"  {tt:8.3f} g{gi} own={str(st):14s} "
                  f"OFF={int(d_off)} ON={int(d_on)}{mark}")


def main() -> None:
    if BASE.exists() and OFF.exists():
        compare_all_keys(BASE, OFF, "[1] 実装前(既存diag) vs 実装後OFF = bit-identical 証明")
    if OFF.exists() and ON.exists():
        compare_all_keys(OFF, ON, "[2] OFF vs ON = 副作用範囲 (is_dead1/2 のみのはず)")
        show_windows(OFF, ON)


if __name__ == "__main__":
    main()
