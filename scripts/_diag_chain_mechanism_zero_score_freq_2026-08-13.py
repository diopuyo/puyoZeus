"""chain_event.total_score=0 persist 事象の頻度診断 (2026-08-13、使い捨て)。

既収集済 (data/indicators_v2/boards_lean_phase_l_2026-08-11/*.npz、127本) の
chain_trigger_sec / chain_mechanism 列を側+ゲーム単位で dedup し、
mechanism 別の出現頻度と「score=0 で生成される機構 (formula/landing) が
次イベント (新規 pseudo イベント or baseline による実スコア解決) までに
何秒保持されるか」「baseline (実スコア) の後続が一定時間内に来るか」を集計する。

前提の注意 (2026-08-13実測): chain_mechanism は「この STABLE snapshot 時点で
有効な検知結果」のキャリーフォワード列であり、同一連鎖の hold 期間中に
重複記録される。dedup (trigger_sec が直前と同値なら同一連鎖の重複として除外)
を必ず行うこと ( _measure_unconfirmed_prechain_2026-08-13.py と同方式)。
"""
from __future__ import annotations

import glob
import statistics
from pathlib import Path

import numpy as np

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
TRIGGER_DEDUP_EPS = 1e-4
FOLLOWUP_WINDOW_SEC = 10.0
ZERO_SCORE_MECHANISMS = ("formula", "landing")


def _dedup_events_by_group(files: list[Path]) -> dict:
    """side+game_idx をキーに (trigger_sec, mechanism) の重複除去済み列を返す。"""
    groups: dict[tuple[str, str, int], list[tuple[float, str]]] = {}
    for path in files:
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if "chain_trigger_sec" not in d.files:
            continue
        trig = d["chain_trigger_sec"].astype(np.float64)
        mech = d["chain_mechanism"] if "chain_mechanism" in d.files else None
        side_arr, game_idx = d["side"], d["game_idx"]
        idx_by_key: dict[tuple[str, int], list[int]] = {}
        for i in range(len(trig)):
            key = (str(side_arr[i]), int(game_idx[i]))
            idx_by_key.setdefault(key, []).append(i)
        for (side, gidx), idxs in idx_by_key.items():
            last = None
            events: list[tuple[float, str]] = []
            for i in idxs:
                tv = trig[i]
                if np.isnan(tv):
                    continue
                if last is not None and abs(tv - last) < TRIGGER_DEDUP_EPS:
                    continue
                last = tv
                m = str(mech[i]) if mech is not None else "(none)"
                events.append((tv, m))
            if events:
                groups[(path.name, side, gidx)] = events
    return groups


def main() -> int:
    files = sorted(NPZ_DIR.glob("*.npz"))
    print(f"対象ファイル数: {len(files)}")
    groups = _dedup_events_by_group(files)

    mech_counts: dict[str, int] = {}
    total_events = 0
    zero_hold_secs: list[float] = []
    followup_gaps: list[float] = []
    has_followup = 0
    no_followup = 0
    per_mech_gap: dict[str, list[float]] = {m: [] for m in ZERO_SCORE_MECHANISMS}

    for events in groups.values():
        for pos, (tv, m) in enumerate(events):
            total_events += 1
            mech_counts[m] = mech_counts.get(m, 0) + 1
            if m not in ZERO_SCORE_MECHANISMS:
                continue
            if pos + 1 < len(events):
                gap = events[pos + 1][0] - tv
                zero_hold_secs.append(gap)
                per_mech_gap[m].append(gap)
            else:
                per_mech_gap[m].append(float("inf"))
            followed = False
            for j in range(pos + 1, len(events)):
                if events[j][0] - tv > FOLLOWUP_WINDOW_SEC:
                    break
                if events[j][1] == "baseline":
                    followed = True
                    followup_gaps.append(events[j][0] - tv)
                    break
            if followed:
                has_followup += 1
            else:
                no_followup += 1

    print(f"\n=== dedup済み chain-trigger イベント総数: {total_events} ===")
    for m, c in sorted(mech_counts.items(), key=lambda kv: -kv[1]):
        print(f"  mechanism={m:12s} n={c:6d} ({c / total_events * 100:.2f}%)")
    zero_born = sum(mech_counts.get(m, 0) for m in ZERO_SCORE_MECHANISMS)
    print(
        f"\nscore=0固定で生成される機構 (formula+landing) の割合: "
        f"{zero_born}/{total_events} = {zero_born / total_events * 100:.2f}%"
    )

    total_zero = has_followup + no_followup
    print(f"\n=== formula/landing 事象 (score=0固定生成) 総数={total_zero} ===")
    print(
        f"  {FOLLOWUP_WINDOW_SEC:.0f}秒以内にbaseline機構(実スコア)が後続: "
        f"{has_followup} ({has_followup / total_zero * 100:.2f}%)"
    )
    print(
        f"  後続なし (score=0のまま別経路で終了 = 実スコア未確定の疑い): "
        f"{no_followup} ({no_followup / total_zero * 100:.2f}%)"
    )
    if followup_gaps:
        fg = sorted(followup_gaps)
        print(
            f"  後続ありの場合のgap: median={statistics.median(fg):.2f}s "
            f"p90={fg[int(len(fg) * 0.9)]:.2f}s max={max(fg):.2f}s"
        )

    if zero_hold_secs:
        zh = sorted(zero_hold_secs)
        over4 = sum(1 for v in zh if v >= 4.0)
        print(
            f"\nscore=0保持の下限 (次イベントまでの間隔): n={len(zh)} "
            f"median={statistics.median(zh):.2f}s p90={zh[int(len(zh) * 0.9)]:.2f}s "
            f"max={max(zh):.2f}s"
        )
        print(f"  4秒以上: {over4}/{len(zh)} = {over4 / len(zh) * 100:.2f}%")

    print("\n--- mechanism別 (formula vs landing) の内訳 ---")
    for m, gaps in per_mech_gap.items():
        finite = sorted(g for g in gaps if np.isfinite(g))
        inf_n = sum(1 for g in gaps if not np.isfinite(g))
        print(f"mechanism={m}: n={len(gaps)} (末尾イベントで後続無し={inf_n})")
        if finite:
            over4 = sum(1 for v in finite if v >= 4.0)
            print(
                f"  次イベントまでgap: median={statistics.median(finite):.2f}s "
                f"p90={finite[int(len(finite) * 0.9)]:.2f}s max={max(finite):.2f}s"
            )
            print(f"  4秒以上: {over4}/{len(finite)} = {over4 / len(finite) * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
