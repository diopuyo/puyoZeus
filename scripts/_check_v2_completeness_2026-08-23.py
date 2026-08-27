"""根治版 (v2) の8区間が完走したかを判定する (2026-08-23)。

## なぜ必要か

前回の再走査で seg05 が要求終点 4379.5秒に対して 4340.8秒で打ち切られていた
(約38.7秒・660フレーム不足)。原因は `visualize_advantage_overlay.py:4893-4895`
の `cap.read()` 失敗時に**エラーログなしで break** する実装。

不足したまま比較すると「解消した」ように見えてしまうので、
**実測終点が要求終点に到達しているか**を先に確認する。

## 判定の閾値

**行数では判定しない。** dump は「再評価のたびに1行追記」する仕様なので、
断片化が直れば再評価の回数自体が変わり、行数の一致は原理的に成立しない。

閾値10.0秒は、**正常完走した8区間 (2026-08-21 の基準) の実測ばらつき
0.03〜7.53秒から決めた**。シーンからの逆算ではない。
前回のバグは38.67秒不足だったので確実に判別できる。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_TOLERANCE_SEC = 10.0
# ファイル名 seg01_0_893.7.npz から開始・終了を取る
_NAME_RE = re.compile(r"seg(\d+)_([0-9.]+)_([0-9.]+)\.npz$")


def main() -> int:
    """8区間の完走を判定する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    files = sorted((PROJECT_ROOT / args.dir).glob("seg*.npz"))
    if not files:
        print(f"[error] npz が無い: {args.dir}")
        return 1

    print(f"=== 完走判定 {args.dir.name} (許容 {_TOLERANCE_SEC}秒) ===")
    print(f"{'区間':>6} {'要求終点':>10} {'実測終点':>10} {'不足':>8} {'行数':>8} 判定")
    ng = 0
    total_rows = 0
    for f in files:
        m = _NAME_RE.search(f.name)
        if m is None:
            print(f"  {f.name}: **ファイル名を解析できない**")
            ng += 1
            continue
        seg, _start, end = m.group(1), float(m.group(2)), float(m.group(3))
        z = np.load(f, allow_pickle=True)
        t = np.asarray(z["t_sec"], dtype=float)
        last = float(t.max()) if len(t) else 0.0
        gap = end - last
        rows = len(t)
        total_rows += rows
        ok = gap <= _TOLERANCE_SEC
        if not ok:
            ng += 1
        print(f"{seg:>6} {end:10.1f} {last:10.1f} {gap:8.2f} {rows:8,} "
              f"{'OK' if ok else '**不足**'}")

    print(f"\n総行数: {total_rows:,} (参考値。行数は再評価回数に依存するので"
          f"基準との一致は想定しない)")
    if ng == 0:
        print("=> **8区間すべて完走** 合格")
        return 0
    print(f"=> **{ng} 区間が不足**。該当区間を再実行すること")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
