"""既存の学習データから K 依存性を分析する (2026-08-09).

user 提案「K12 を近似的に高速処理する方法」の検証。
重い MC を回さなくても、 **既存指標 near_future_fire_k1..k5 が K 別の火力を
持っている** ので、 そこから伸び方を調べられる (ライブラリアンの教訓)。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

CSV = _ROOT / "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"

def main() -> int:
    df = pd.read_csv(CSV, usecols=lambda c: c.startswith("near_future_fire_k")
                     or c in ("board_puyo_total",))
    cols = [f"near_future_fire_k{k}" for k in range(1, 6)]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna()
    print(f"サンプル {len(sub)} 行 / K = {len(cols)} 段階")
    print()
    print(f"{'K':>3s} {'平均':>9s} {'標準偏差':>10s} {'中央値':>9s} {'p90':>9s}")
    means, stds = [], []
    for i, c in enumerate(cols, start=1):
        v = sub[c].values
        means.append(float(v.mean())); stds.append(float(v.std()))
        print(f"{i:3d} {v.mean():9.4f} {v.std():10.4f} "
              f"{np.median(v):9.4f} {np.percentile(v, 90):9.4f}")
    m = np.array(means); ks = np.arange(1, len(m) + 1)
    print()
    print("増分:", np.round(np.diff(m), 4))
    print()
    print("=== 伸び方の当てはまり (R^2) ===")
    best = None
    for name, xs in (("線形", ks), ("対数", np.log(ks)), ("平方根", np.sqrt(ks))):
        A = np.vstack([xs, np.ones(len(xs))]).T
        coef, *_ = np.linalg.lstsq(A, m, rcond=None)
        pred = A @ coef
        r2 = 1 - ((m - pred) ** 2).sum() / ((m - m.mean()) ** 2).sum()
        print(f"  {name}: R^2={r2:.4f}  (係数 {coef[0]:+.4f}, 切片 {coef[1]:+.4f})")
        if best is None or r2 > best[1]: best = (name, r2, coef, xs)
    print()
    name, r2, coef, xs = best
    print(f"最良: {name} (R^2={r2:.4f})")
    print("外挿値 (この形が K>5 でも続くと仮定した場合):")
    for k in (6, 8, 12, 16):
        x = {"線形": k, "対数": np.log(k), "平方根": np.sqrt(k)}[name]
        print(f"  K={k:2d}: {coef[0] * x + coef[1]:.4f}")
    print()
    print("注意: これは **外挿** であり実測ではない。 実際には盤面の空き容量が")
    print("      効くため、 どこかで飽和するはず。 上限の検証が別途必要。")
    print()
    print("=== 標準偏差の推移 (正規近似が使えるか) ===")
    s = np.array(stds)
    print("  標準偏差:", np.round(s, 4))
    print("  変動係数 (std/mean):", np.round(s / np.maximum(m, 1e-9), 3))
    print("  → 変動係数が K とともに下がるなら、 深いほど平均で語れる")
    print("    (= user 指摘『ツモが平均化される』の定量的な裏付け)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
