# -*- coding: utf-8 -*-
"""±100張り付き根治 (2026-08-24) の before/after ペア測定。

同一動画・同一コード・フラグ OFF/ON の2つの dump ディレクトリを比較する
(feedback_paired_comparison_fixed_population_2026-08-20: 母集団を固定した
ペア比較でなければ改善が悪化に見える)。

測定4指標 (memory project_pm100_display_flip_2026-08-24 の実測と同定義):
  ① ±100張り付き時間 (|adv_ema| >= 99.5、行間ギャップ0.5s超で区間を切る)
  ② うち生値 (adv_raw) と符号が逆の時間
  ③ 試合内で張り付き符号が反転した試合数
  ④ 1秒以内に150pt以上の急変回数

加えて受け入れ条件4 (真の致死の見逃し禁止):
  各試合の最終フレームの is_dead1/2 から勝者を推定し (前セッション
  boundary_winner.py と同一方式)、最終表示 (adv_ema) の向きが勝者と一致
  しているかを OFF/ON で全数比較する。

使い方:
  python -m scripts._analyze_pm100_pair_2026-08-24 <dumps_off_dir> <dumps_on_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STICK_TH = 99.5     # 張り付き判定 (親測定と同じ)
GAP_SEC = 0.5       # 行間ギャップ (決着ホールド等で dump 行が無い時間)
EVEN_TH = 3.0       # 生値の向き判定不能帯 (EVEN閾値)
SWING_PT = 150.0    # 急変の大きさ
SWING_SEC = 1.0     # 急変の時間窓
FINAL_WEAK_TH = 30.0  # 最終表示が「鈍い」とみなす閾値 (勝敗確定局面の期待強度)


def _load_dir(d: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(d.glob("seg*.npz")):
        f = np.load(p, allow_pickle=True)
        out.append((p.stem[:5], {k: f[k] for k in f.files}))
    return out


def _sticky_runs(t, ema, g):
    """張り付き区間 [(game, t0, t1, dur, sign)] を返す。"""
    runs = []
    on = np.abs(ema) >= STICK_TH
    i, n = 0, len(t)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and on[j + 1] and g[j + 1] == g[i]
               and t[j + 1] - t[j] <= GAP_SEC):
            j += 1
        runs.append((int(g[i]), float(t[i]), float(t[j]),
                     float(t[j] - t[i]), float(np.sign(ema[i]))))
        i = j + 1
    return runs


def _count_swings(t, ema, g) -> int:
    """1秒以内に150pt以上動いた回数 (同一試合内、greedy で二重計上回避)。"""
    n = len(t)
    count = 0
    i = 0
    while i < n - 1:
        j = i + 1
        hit = -1
        while j < n and t[j] - t[i] <= SWING_SEC and g[j] == g[i]:
            if abs(float(ema[j]) - float(ema[i])) >= SWING_PT:
                hit = j
                break
            j += 1
        if hit >= 0:
            count += 1
            i = hit  # 同じ急変を重複カウントしない
        else:
            i += 1
    return count


def _metrics(segs) -> dict:
    total_time = 0.0
    stick_time = 0.0
    wrong_time = 0.0
    swings = 0
    all_runs = []
    for seg, d in segs:
        t, ema, raw, g = d["t_sec"], d["adv_ema"], d["adv_raw"], d["game_idx"]
        total_time += float(t[-1] - t[0])
        dt = np.clip(np.diff(t, append=t[-1]), 0, GAP_SEC)
        mask = ((np.abs(ema) >= STICK_TH) & (np.abs(raw) >= EVEN_TH)
                & (np.sign(ema) != np.sign(raw)))
        wrong_time += float(dt[mask].sum())
        runs = _sticky_runs(t, ema, g)
        stick_time += sum(r[3] for r in runs)
        all_runs += [(seg,) + r for r in runs]
        swings += _count_swings(t, ema, g)
    # 試合内符号反転
    flips: dict[tuple[str, int], int] = {}
    last_sign: dict[tuple[str, int], float] = {}
    for seg, gi, t0, t1, dur, sign in all_runs:
        key = (seg, gi)
        if key in last_sign and last_sign[key] != sign:
            flips[key] = flips.get(key, 0) + 1
        last_sign[key] = sign
    return {
        "total_time": total_time, "stick_time": stick_time,
        "wrong_time": wrong_time, "swings": swings,
        "runs": all_runs, "flip_games": flips,
    }


TRUTH_TAIL_SEC = 2.0  # 勝者推定に使う末尾窓 (最終行だけだと確定が疎になる)


def _final_verdicts(segs) -> dict[tuple[str, int], dict]:
    """試合ごとの最終表示と is_dead ベースの勝者推定。

    勝者推定: 各試合の dump 末尾 TRUTH_TAIL_SEC 秒の窓で is_dead1/2 を見る。
    片側だけが dead を示せばその逆側を勝者とする (前セッション
    boundary_winner.py の最終行方式を窓に拡張してカバレッジを上げたもの。
    is_dead は凍結盤面に対する誤検知が既知 [memory 8/23] だが、試合末尾の
    窓では敗者側が実際に埋まっているため信頼できる。両側 dead は不確定)。
    最終表示: 末尾 1.0 秒の adv_ema 平均 (単一行のノイズを避ける)。
    """
    out: dict[tuple[str, int], dict] = {}
    for seg, d in segs:
        t, ema, g = d["t_sec"], d["adv_ema"], d["game_idx"]
        d1 = d.get("is_dead1")
        d2 = d.get("is_dead2")
        for gi in np.unique(g):
            m = g == gi
            tt, ee = t[m], ema[m]
            t_end = float(tt[-1])
            tail = tt >= t_end - TRUTH_TAIL_SEC
            truth = None
            if d1 is not None and d2 is not None:
                dd1 = bool(np.any(d1[m][tail]))
                dd2 = bool(np.any(d2[m][tail]))
                if dd1 != dd2:
                    truth = "2P" if dd1 else "1P"  # 死んだ側の逆が勝者
            disp_tail = tt >= t_end - 1.0
            out[(seg, int(gi))] = {
                "t_end": t_end,
                "final_adv": float(np.mean(ee[disp_tail])),
                "truth": truth,
            }
    return out


def main() -> None:
    off_dir, on_dir = Path(sys.argv[1]), Path(sys.argv[2])
    off = _load_dir(off_dir)
    on = _load_dir(on_dir)
    m_off, m_on = _metrics(off), _metrics(on)

    def _row(label, m):
        pct = 100 * m["stick_time"] / m["total_time"]
        wpct = 100 * m["wrong_time"] / m["total_time"]
        print(f"{label}: 総時間 {m['total_time']:.1f}s | "
              f"①張り付き {m['stick_time']:.1f}s ({pct:.1f}%) 区間{len(m['runs'])} | "
              f"②符号逆 {m['wrong_time']:.1f}s ({wpct:.1f}%) | "
              f"③反転試合 {len(m['flip_games'])} (延べ{sum(m['flip_games'].values())}回) | "
              f"④急変 {m['swings']}回")

    print("== 4指標 (同一動画ペア比較) ==")
    _row("OFF", m_off)
    _row("ON ", m_on)

    # 受け入れ条件4: 最終表示の向き
    v_off, v_on = _final_verdicts(off), _final_verdicts(on)
    keys = sorted(set(v_off) & set(v_on))
    n_truth = wrong_off = wrong_on = 0
    weakened = []
    sign_changed = []
    for k in keys:
        fo, fn = v_off[k], v_on[k]
        if np.sign(fo["final_adv"]) != np.sign(fn["final_adv"]):
            sign_changed.append((k, fo["final_adv"], fn["final_adv"]))
        truth = fn["truth"]
        if truth is None:
            continue
        n_truth += 1
        want = 1.0 if truth == "1P" else -1.0
        if np.sign(fo["final_adv"]) != want:
            wrong_off += 1
        if np.sign(fn["final_adv"]) != want:
            wrong_on += 1
        if (abs(fo["final_adv"]) >= FINAL_WEAK_TH
                and abs(fn["final_adv"]) < FINAL_WEAK_TH):
            weakened.append((k, fo["final_adv"], fn["final_adv"], truth))
    print(f"\n== 受け入れ条件4: 決着直前の表示 (試合数 {len(keys)}、勝者確定 {n_truth}) ==")
    print(f"最終表示の向きが勝者と逆: OFF {wrong_off} 試合 / ON {wrong_on} 試合")
    print(f"ONで最終表示が鈍った (|adv| {FINAL_WEAK_TH}以上→未満): {len(weakened)} 試合")
    for k, a, b, tr in weakened:
        print(f"  {k[0]} game{k[1]}: OFF {a:+.1f} -> ON {b:+.1f} (勝者 {tr})")
    print(f"OFF/ONで最終表示の符号が変わった試合: {len(sign_changed)}")
    for k, a, b in sign_changed[:20]:
        print(f"  {k[0]} game{k[1]}: OFF {a:+.1f} -> ON {b:+.1f}")

    # seg01 game2 (user指摘場面) の反転確認
    print("\n== seg01 game2 (t=141-224、user指摘場面) ==")
    for label, segs in (("OFF", off), ("ON", on)):
        for seg, d in segs:
            if seg != "seg01":
                continue
            t, ema = d["t_sec"], d["adv_ema"]
            m = (t >= 141.0) & (t <= 224.0)
            tt, ee = t[m], ema[m]
            flips = 0
            details = []
            for i in range(1, len(tt)):
                if (abs(ee[i - 1]) >= STICK_TH and abs(ee[i]) >= STICK_TH
                        and np.sign(ee[i]) != np.sign(ee[i - 1])):
                    flips += 1
                    details.append(f"t={tt[i]:.2f}")
                # ±99.5 未満を経由した実質反転 (1秒以内に両極)
            print(f"{label} {seg}: 窓内の隣接行±100反転 {flips}回 {details}")
            # t=211.4 付近の軌跡
            m2 = (tt >= 210.5) & (tt <= 213.0)
            tr = [f"{a:.2f}:{b:+.0f}" for a, b in zip(tt[m2], ee[m2])]
            print(f"  t=210.5-213.0 軌跡: {' '.join(tr)}")


if __name__ == "__main__":
    main()
