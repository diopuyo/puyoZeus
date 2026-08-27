"""32件の「新規」エピソードが分割・移動なのか真に新規なのかを判定する診断 (2026-08-23)。

scripts/_compare_kill_override_fix_episodes_2026-08-22.py と全く同じ
collect_episodes/overlaps ロジックを複製し (本体コード非変更方針)、
「新規」判定された after エピソードそれぞれについて、
同一ファイル内の最も近い before エピソードとの時間差 (中心時刻ベース) を
計算する。閾値 (既定 ±5.0 秒) 以内なら「分割・移動」候補、
それ以外は「真に新規」候補として分類する。

コードは変更しない。計装専用。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

DEFAULT_BEFORE_DIR = PROJECT_ROOT / "data/verify/zenchi_render_2026-08-21"
DEFAULT_AFTER_DIR = PROJECT_ROOT / "data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22"

KILL_ROOM_FLOOR = 4
KILL_RATIO_MIN = 0.6
KILL_RATIO_FULL = 1.5
KILL_MIN_PENDING = 40


def kill_g(inc1: float, inc2: float, room1: float, room2: float) -> float:
    l1 = inc1 / max(KILL_ROOM_FLOOR, room1) if inc1 >= KILL_MIN_PENDING else 0.0
    l2 = inc2 / max(KILL_ROOM_FLOOR, room2) if inc2 >= KILL_MIN_PENDING else 0.0
    lead = l1 - l2
    mag = abs(lead)
    if mag < KILL_RATIO_MIN:
        return 0.0
    return min(1.0, (mag - KILL_RATIO_MIN) / (KILL_RATIO_FULL - KILL_RATIO_MIN))


def collect_episodes(dump_dir: Path) -> list[dict]:
    files = sorted(glob.glob(str(dump_dir / "seg*.npz")))
    episodes: list[dict] = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = d["t_sec"]
        adv_raw, adv_ema = d["adv_raw"], d["adv_ema"]
        pend1, pend2 = d["pending_p1"].astype(float), d["pending_p2"].astype(float)
        room1, room2 = d["room1"].astype(float), d["room2"].astype(float)
        n = len(t)
        g = np.array([kill_g(pend1[i], pend2[i], room1[i], room2[i]) for i in range(n)])
        conflict = (
            (g >= 0.9) & (np.abs(adv_raw) >= 5.0) & (np.abs(adv_ema) >= 50.0)
            & (np.sign(adv_raw) != np.sign(adv_ema))
        )
        idx = np.where(conflict)[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0]
        for grp in np.split(idx, splits + 1):
            episodes.append(dict(
                file=Path(f).name, t0=float(t[grp[0]]), t1=float(t[grp[-1]]),
            ))
    return episodes


def overlaps(a: dict, b: dict) -> bool:
    return a["file"] == b["file"] and a["t0"] <= b["t1"] and b["t0"] <= a["t1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-dir", type=Path, default=DEFAULT_BEFORE_DIR)
    ap.add_argument("--after-dir", type=Path, default=DEFAULT_AFTER_DIR)
    ap.add_argument("--window", type=float, default=5.0, help="分割・移動とみなす中心時刻差の閾値(秒)")
    a = ap.parse_args()

    before = collect_episodes(a.before_dir)
    after = collect_episodes(a.after_dir)

    new = [ep for ep in after if not any(overlaps(e, ep) for e in before)]
    print(f"修正前episode数={len(before)} 修正後episode数={len(after)} 新規={len(new)}")

    print("\n--- 新規32件の最近傍before対応表 ---")
    print("file\tafter_t0\tafter_t1\tafter_dur\tnearest_before_t0\tnearest_before_t1\tcenter_diff_sec\t分類")
    split_or_move = 0
    truly_new = 0
    rows = []
    for ep in new:
        same_file_before = [e for e in before if e["file"] == ep["file"]]
        center_ep = (ep["t0"] + ep["t1"]) / 2
        if same_file_before:
            # 中心時刻が最も近いbeforeを探す
            best = min(same_file_before, key=lambda e: abs((e["t0"] + e["t1"]) / 2 - center_ep))
            center_before = (best["t0"] + best["t1"]) / 2
            diff = abs(center_before - center_ep)
            # 追加基準: 時間窓として隣接/重複に近いか (端点距離)
            edge_gap = max(0.0, max(ep["t0"] - best["t1"], best["t0"] - ep["t1"]))
        else:
            best = None
            diff = float("inf")
            edge_gap = float("inf")
        classification = "分割・移動" if diff <= a.window else "真に新規"
        if classification == "分割・移動":
            split_or_move += 1
        else:
            truly_new += 1
        rows.append((ep, best, diff, edge_gap, classification))
        b0 = f"{best['t0']:.2f}" if best else "-"
        b1 = f"{best['t1']:.2f}" if best else "-"
        print(f"{ep['file']}\t{ep['t0']:.2f}\t{ep['t1']:.2f}\t{ep['t1']-ep['t0']:.2f}\t{b0}\t{b1}\t{diff:.2f}\t{classification}")

    print(f"\n分割・移動 (中心差<= {a.window}秒): {split_or_move} 件")
    print(f"真に新規候補 (中心差 > {a.window}秒 または同一ファイルにbeforeなし): {truly_new} 件")

    print("\n--- 真に新規候補の詳細 (edge_gapも表示) ---")
    for ep, best, diff, edge_gap, cls in rows:
        if cls == "真に新規":
            b = f"({best['t0']:.2f}-{best['t1']:.2f})" if best else "(同ファイルにbeforeなし)"
            print(f"{ep['file']} t0={ep['t0']:.2f} t1={ep['t1']:.2f} dur={ep['t1']-ep['t0']:.2f} "
                  f"最近傍before={b} center_diff={diff:.2f} edge_gap={edge_gap:.2f}")


if __name__ == "__main__":
    main()
