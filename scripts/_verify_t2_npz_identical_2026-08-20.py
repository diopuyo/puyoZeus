"""T2: 2つの npz が全列 bit 一致かを検証する (2026-08-20)。

native HSV 分類 (Rust) の採用可否を決める最終ゲート。合成パッチでの一致
(T1、4,732枚×4構成で不一致0) は確認済みだが、**本番経路を通したときに同じ
盤面・同じラベルが出るか**は別問題。状態機械・CNN smoothing 履歴・おじゃま
会計などの下流が絡むため、npz の全キーを突き合わせる。

受け入れ条件: **全キーで array_equal**。1 セルでも違えば採用不可。

fail-silent 対策: キーの集合が違う / 行数が違う / 片方にしか無いキーがある
といった「比較にならない」状態を成功と誤読しないよう、明示的に落とす。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _cmp(a: Path, b: Path) -> bool:
    """2 npz の全キーを突き合わせる。一致すれば True。"""
    da = np.load(a, allow_pickle=True)
    db = np.load(b, allow_pickle=True)
    ka, kb = set(da.keys()), set(db.keys())
    print(f"\n=== {a.name} ===")
    print(f"  OFF: {a.parent.name}   ON: {b.parent.name}")
    if ka != kb:
        print(f"  ✗ キー集合が違う: OFF のみ={sorted(ka - kb)} / ON のみ={sorted(kb - ka)}")
        return False
    print(f"  キー数 {len(ka)}")

    ok = True
    for k in sorted(ka):
        va, vb = np.asarray(da[k]), np.asarray(db[k])
        if va.shape != vb.shape:
            print(f"  ✗ {k}: 形状が違う {va.shape} vs {vb.shape}")
            ok = False
            continue
        if va.dtype.kind == "f":
            # NaN を含む列 (won 等) は NaN 位置も含めて一致を要求する
            same = np.array_equal(va, vb, equal_nan=True)
        else:
            same = np.array_equal(va, vb)
        if not same:
            diff = int(np.sum(va != vb)) if va.dtype.kind != "f" else int(
                np.sum(~((va == vb) | (np.isnan(va) & np.isnan(vb))))
            )
            print(f"  ✗ {k}: 不一致 {diff} / {va.size} 要素 ({diff/max(1,va.size)*100:.4f}%)")
            ok = False
    if ok:
        rows = int(np.asarray(da[sorted(ka)[0]]).shape[0])
        print(f"  ✓ 全 {len(ka)} キーが bit 一致 (行数 {rows})")
    return ok


def main() -> int:
    """usage: [--off-dir DIR] [--on-dir DIR] [target ...]"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-dir", type=Path,
                    default=Path("data/indicators_v2/boards_lean_t2_off_2026-08-20"))
    ap.add_argument("--on-dir", type=Path,
                    default=Path("data/indicators_v2/boards_lean_t2_on_2026-08-20"))
    ap.add_argument("targets", nargs="*", help="target_id (省略で OFF 側の全件)")
    args = ap.parse_args()

    off_files = sorted(args.off_dir.glob("*.npz"))
    if args.targets:
        off_files = [args.off_dir / f"{t}.npz" for t in args.targets]
    if not off_files:
        print(f"[error] 比較対象が無い: {args.off_dir}")
        return 1

    all_ok = True
    n_cmp = 0
    for a in off_files:
        b = args.on_dir / a.name
        if not a.exists():
            print(f"[skip] OFF 側が無い: {a}")
            continue
        if not b.exists():
            print(f"[skip] ON 側が無い (収集中?): {b}")
            continue
        n_cmp += 1
        if not _cmp(a, b):
            all_ok = False

    print()
    if n_cmp == 0:
        print("=== 比較できたファイルが0件。収集の完了を待つこと ===")
        return 1
    if all_ok:
        print(f"=== T2 合格: {n_cmp} 本すべて bit 一致 → 採用可 ===")
        return 0
    print(f"=== T2 不合格: 不一致あり → 採用不可 ===")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
