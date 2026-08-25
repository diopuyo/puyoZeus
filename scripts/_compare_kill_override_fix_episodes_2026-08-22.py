"""修正① (kill_override 連鎖完走後是正) 適用前後で符号衝突エピソードが
どう変わったかを比較する (2026-08-22)。

前提: 以下2つのディレクトリに同一区間分割の dump (npz) が揃っていること。
  - 修正前: data/verify/zenchi_render_2026-08-21 (112エピソード基準値)
  - 修正後: data/verify/zenchi_render_kill_override_fix_2026-08-22
    (scripts/_rescan_zenchi_kill_override_fix_2026-08-22.sh の出力)

判定は scripts/_diag_kill_raw_display_conflict_2026-08-22.py と全く同じ式
(kill_g) を使う。ファイル名にハイフンを含むためモジュールとして import
できず、8行の純関数を意図的に複製している (診断スクリプト間の既存の
慣行、本体コード scripts/visualize_advantage_overlay.py:kill_override が
単一情報源であることに変わりはない)。

出力: 各エピソードを (file, t0, t1) の時間窓の重なりで対応付け、
  - 解消 (修正前にあり、修正後の同時刻帯に衝突が無い)
  - 残存 (修正前にもあり、修正後の同時刻帯にも衝突が残る)
  - 新規 (修正後にのみ現れる、想定外=要調査)
に分類する。t=6717.5 等の実測済み7エピソードも個別に追跡する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

# [2026-08-22 是正] 従来はこの2定数を直接 main() が参照しており、
# CLI 引数化されていなかった (coordinator指摘: --dir を渡しても無視され
# v1 を見続けていた事故)。既定値は従来と完全に同一に保つ (backwards
# compat、無引数実行で今までと同じ112エピソードが再現されることを
# 必ず確認してから使うこと)。
DEFAULT_BEFORE_DIR = PROJECT_ROOT / "data/verify/zenchi_render_2026-08-21"
DEFAULT_AFTER_DIR = PROJECT_ROOT / "data/verify/zenchi_render_kill_override_fix_2026-08-22"

# scripts/_diag_kill_raw_display_conflict_2026-08-22.py の定数/関数と完全同一
# (ファイル名にハイフンを含むため import 不可、意図的な複製、上記docstring参照)。
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

# 実測済み7エピソード (logs/killoverride_wrong_2026-08-22/一覧.tsv)。
KNOWN_ANCHORS = (324.567, 807.667, 1030.567, 1031.467, 4914.533, 6717.500, 7017.933)


def collect_episodes(dump_dir: Path) -> list[dict]:
    import glob
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
    ap.add_argument(
        "--before-dir", type=Path, default=DEFAULT_BEFORE_DIR,
        help=f"修正前 (旧本番dump) の seg*.npz ディレクトリ (既定={DEFAULT_BEFORE_DIR})")
    ap.add_argument(
        "--after-dir", type=Path, default=DEFAULT_AFTER_DIR,
        help=f"修正後の seg*.npz ディレクトリ (既定={DEFAULT_AFTER_DIR})")
    a = ap.parse_args()
    print(f"[before-dir] {a.before_dir}")
    print(f"[after-dir]  {a.after_dir}")

    before = collect_episodes(a.before_dir)
    after = collect_episodes(a.after_dir)
    print(f"修正前: {len(before)} エピソード")
    print(f"修正後: {len(after)} エピソード")

    resolved = [e for e in before if not any(overlaps(e, a) for a in after)]
    persisted = [e for e in before if any(overlaps(e, a) for a in after)]
    new = [a for a in after if not any(overlaps(e, a) for e in before)]

    print(f"\n解消: {len(resolved)} 件")
    print(f"残存: {len(persisted)} 件")
    print(f"新規 (想定外・要調査): {len(new)} 件")

    if persisted:
        print("\n--- 残存エピソード一覧 (先頭30件) ---")
        for e in persisted[:30]:
            print(e)
    if new:
        print("\n--- 新規エピソード一覧 (全件) ---")
        for e in new:
            print(e)

    print("\n=== 既知7アンカーの追跡 ===")
    for t_anchor in KNOWN_ANCHORS:
        hit_before = [e for e in before if e["t0"] - 1 <= t_anchor <= e["t1"] + 1]
        hit_after = [e for e in after if e["t0"] - 1 <= t_anchor <= e["t1"] + 1]
        print(f"t={t_anchor}: 修正前衝突={bool(hit_before)} 修正後衝突={bool(hit_after)}")


if __name__ == "__main__":
    main()
