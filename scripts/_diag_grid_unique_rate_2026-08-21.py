"""盤面のユニーク率を測る (2026-08-21、応手の到達確率を繋ぐ前提)。

## なぜ必要か

応手の到達確率 (`counter_reach_probability_fast` / `mc_counter_estimator`) は
1件あたり 170〜528ms かかる。62本の学習データは 34万行あるので、全行を素で
計算すると直列16時間を超えて非現実的。

ただし盤面は「ぷよが止まって落ち着いた瞬間」だけを記録しているため、**同じ
盤面が連続する行が多い**と見込める (手が変わるまで盤面は変わらない)。
同一盤面をキャッシュすれば実効計算数は「手替わりの回数」に落ちる。

fable の設計ではこれが着工判断の前提になっている:
  ユニーク評価数が 8.5万以下 → 全行フル計算 (直列4時間、動画単位14並列で20分級)
  超える場合 → 手替わり行のみ計算して間に合わせる (carry)

## 何を測るか

1. 盤面のユニーク数 (バイト列のハッシュで数える)
2. 閾値をバケット化した場合の (盤面, 閾値) 組み合わせ数
   — 応手の到達確率は「どれだけのおじゃまを返せるか」の閾値も引数に取るため、
   キャッシュキーは盤面だけでは足りない。6個 (=1段) 刻みで量子化する
   (おじゃま換算の誤差±数個は元々あるので実害は小さい)
3. 1動画あたりの平均・最大 (並列実行時の粒度を決めるため)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 閾値の量子化幅 (6個 = 1段。おじゃまは6列に均等配分されるため段が自然な単位)
_THRESHOLD_BUCKET = 6


def _grid_key(grid: np.ndarray) -> bytes:
    """盤面のハッシュキー (同一内容なら同一キー)。"""
    return hashlib.blake2b(
        np.ascontiguousarray(grid, dtype=np.int8).tobytes(), digest_size=16,
    ).digest()


def main() -> int:
    """npz 群を走査してユニーク率を出す。"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz-dir", type=Path,
        default=Path("data/indicators_v2/boards_lean_model50v2_2026-08-20"),
    )
    args = ap.parse_args()

    files = sorted(args.npz_dir.glob("*.npz"))
    if not files:
        print(f"[error] npz が無い: {args.npz_dir}")
        return 1

    total_rows = 0
    all_grids: set[bytes] = set()
    all_pairs: set[tuple[bytes, int]] = set()
    per_video: list[tuple[str, int, int, int]] = []

    for f in files:
        z = np.load(f, allow_pickle=True)
        grids = z["grids"]
        # 閾値の材料: 相手の最大連鎖のおじゃま換算 ≒ 予告量で近似する
        # (実際の閾値は相手盤面から出すが、ここでは量子化後の種類数を
        #  見積もるのが目的なので既存列で代用する)
        forecast = (
            np.asarray(z["ojama_forecast"], dtype=float)
            if "ojama_forecast" in z else np.zeros(len(grids))
        )
        gk: set[bytes] = set()
        pk: set[tuple[bytes, int]] = set()
        for i in range(len(grids)):
            k = _grid_key(grids[i])
            gk.add(k)
            fv = forecast[i] if i < len(forecast) else 0.0
            if not np.isfinite(fv):
                fv = 0.0
            bucket = int(fv * 72.0 // _THRESHOLD_BUCKET)  # 0-1正規化を個数へ戻す
            pk.add((k, bucket))
        total_rows += len(grids)
        all_grids |= gk
        all_pairs |= pk
        per_video.append((f.stem, len(grids), len(gk), len(pk)))

    print(f"=== {args.npz_dir.name} ({len(files)} 本) ===")
    print(f"総行数            {total_rows:,}")
    print(f"ユニーク盤面      {len(all_grids):,}  ({len(all_grids)/total_rows*100:.1f}%)")
    print(f"(盤面,閾値) の組  {len(all_pairs):,}  ({len(all_pairs)/total_rows*100:.1f}%)")
    print()
    per_video.sort(key=lambda x: -x[3])
    print(f"{'動画':>8} {'行数':>8} {'ユニーク盤面':>12} {'(盤面,閾値)':>12}")
    print("-" * 46)
    for name, n, g, pr in per_video[:5]:
        print(f"{name:>8} {n:>8,} {g:>12,} {pr:>12,}")
    print(f"{'...':>8}")
    avg_pair = sum(x[3] for x in per_video) / len(per_video)
    print(f"1動画あたり平均 (盤面,閾値) 組: {avg_pair:,.0f}")
    print()
    # 判定
    print("--- 着工判断 (fable 設計の基準) ---")
    n_eval = len(all_pairs)
    print(f"評価が必要な回数: {n_eval:,}")
    for label, ms in (("fast (170ms)", 170), ("precise (528ms)", 528)):
        hours = n_eval * ms / 1000 / 3600
        par = hours / 14
        print(f"  {label:<18} 直列 {hours:6.1f} 時間 / 14並列 {par*60:6.0f} 分")
    if n_eval <= 85000:
        print("  → 8.5万以下: **全行フル計算が現実的**")
    else:
        print("  → 8.5万超: 手替わり行のみ計算して carry する設計が必要")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
