"""納品レンダ全8セグメント dump で ±100 張り付きの頻度・反転回数を定量化する。

定義:
  - 張り付き区間: |adv_ema| >= 99.5 が連続する区間 (dump 行ベース、行間ギャップ
    0.5s 超は区間を切る。ギャップ=決着ホールド中で dump に表示値が無い時間帯)。
  - 反転: 同一試合内で張り付き区間の符号が前回の張り付き区間と逆。
  - 生値と符号逆: |adv_ema| >= 99.5 かつ adv_raw と符号が逆 (adv_raw の
    |値| >= 3.0 = EVEN閾値、微小生値は判定不能として除外)。
注意: dump の adv_ema は決着ホールド中の実表示値を含まない (settled 停止で行が
出ない) ため、本統計は「dump に写った範囲」の保守的な下限。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DIR = Path("data/verify/zenchi_render_2026-08-21")
TH = 99.5
GAP = 0.5
EVEN = 3.0


def main() -> None:
    """8セグメントの張り付き区間を集計する。"""
    all_runs: list[tuple[str, int, float, float, float, float]] = []
    total_dur = 0.0
    wrong_sign_time = 0.0
    total_time = 0.0
    for p in sorted(DIR.glob("seg*.npz")):
        f = np.load(p, allow_pickle=True)
        t = f["t_sec"]; ema = f["adv_ema"]; raw = f["adv_raw"]; g = f["game_idx"]
        total_time += float(t[-1] - t[0])
        # 符号逆時間 (行間隔で重み付け)
        dt = np.diff(t, append=t[-1])
        dt = np.clip(dt, 0, GAP)
        mask = (np.abs(ema) >= TH) & (np.abs(raw) >= EVEN) & (np.sign(ema) != np.sign(raw))
        wrong_sign_time += float(dt[mask].sum())
        # 張り付き区間の抽出
        on = np.abs(ema) >= TH
        i = 0
        n = len(t)
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while (j + 1 < n and on[j + 1] and g[j + 1] == g[i]
                   and t[j + 1] - t[j] <= GAP):
                j += 1
            dur = float(t[j] - t[i])
            sign = float(np.sign(ema[i]))
            all_runs.append((p.stem[:5], int(g[i]), float(t[i]), float(t[j]), dur, sign))
            total_dur += dur
            i = j + 1
    print(f"総dump時間: {total_time:.1f}s  張り付き総時間: {total_dur:.1f}s "
          f"({100*total_dur/total_time:.1f}%)  区間数: {len(all_runs)}")
    print(f"生値と符号逆の張り付き時間: {wrong_sign_time:.1f}s "
          f"({100*wrong_sign_time/total_time:.1f}%)")
    durs = np.array([r[4] for r in all_runs])
    if len(durs):
        print(f"区間長: min={durs.min():.1f} med={np.median(durs):.1f} "
              f"p90={np.percentile(durs,90):.1f} max={durs.max():.1f}")
    # 試合内の符号反転回数
    flips: dict[tuple[str, int], int] = {}
    last_sign: dict[tuple[str, int], float] = {}
    for seg, gi, t0, t1, dur, sign in all_runs:
        key = (seg, gi)
        if key in last_sign and last_sign[key] != sign:
            flips[key] = flips.get(key, 0) + 1
        last_sign[key] = sign
    n_games_with_pm = len(last_sign)
    print(f"±100張り付きが出た試合(セグ×game): {n_games_with_pm}")
    fl = sorted(flips.items(), key=lambda kv: -kv[1])
    print(f"試合内で符号反転あり: {len(fl)}試合、反転回数分布: "
          f"{[v for _, v in fl]}")
    for (seg, gi), v in fl[:10]:
        print(f"  {seg} game{gi}: {v}回反転")
    # 長い張り付き上位
    print("--- 張り付き区間 上位15 (長さ順) ---")
    for seg, gi, t0, t1, dur, sign in sorted(all_runs, key=lambda r: -r[4])[:15]:
        print(f"  {seg} game{gi} t={t0:.1f}-{t1:.1f} ({dur:.1f}s) sign={'+' if sign>0 else '-'}")


if __name__ == "__main__":
    main()
