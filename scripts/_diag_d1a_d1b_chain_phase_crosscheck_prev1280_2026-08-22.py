"""前回 (修正前) 1,280件 (judgment_scan_zenchi_2026-08-22/suspects.tsv) について
同じ観点 (state1/state2 による連鎖中判定) を集計する。

重要な限界 (断定を避けるための明記):
このtsvが生成された時刻 (2026-08-22 06:04) に対し、参照する dump npz
(data/verify/zenchi_render_2026-08-21/seg*.npz) は 2026-08-22 12:37-12:40 に
再生成されたもの (kill_override への pending 入力を fctracker.inc から
確定会計ベースへ差し替えた修正、visualize_advantage_overlay.py 4755-4761行
付近のコメント参照)。したがって:
  - D1a (is_dead 直読み、盤面認識のみに依存): pending 修正の影響を受けない
    はずなので、現在の dump の state1/state2/is_dead1/is_dead2 で近似的に
    突合してよいと考えられる (ただし断定はしない)。
  - D1b (pending/room 比を使う): 現在の dump の pending は「修正後」の値
    であり、旧 suspects.tsv 生成時に実際にD1bの引き金になった「修正前」の
    pending 値とは異なる可能性がある。ここでの pending 列は参考値に留め、
    state1/state2 (連鎖中か) のみを本集計の主目的とする。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parent / "_diag_d1a_d1b_chain_phase_crosscheck_2026-08-22.py"
_spec = importlib.util.spec_from_file_location("_diag_chain_crosscheck_base", _MOD_PATH)
_base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base)

CHAIN_BROAD = _base.CHAIN_BROAD
CHAIN_NARROW = _base.CHAIN_NARROW
load_all_dumps = _base.load_all_dumps
nearest_match = _base.nearest_match
parse_side_from_evidence = _base.parse_side_from_evidence

BASE = Path(__file__).resolve().parent.parent
OLD_SUSPECTS_TSV = BASE / "data/verify/judgment_scan_zenchi_2026-08-22/suspects.tsv"
MATCH_TOL_SEC = 0.02


def main() -> None:
    print("[1/3] 旧 suspects.tsv 読み込み中...")
    lines = OLD_SUSPECTS_TSV.read_text(encoding="utf-8").splitlines()
    rows = [l.split("\t") for l in lines[1:]]
    print(f"  総行数={len(rows)}")

    print("[2/3] dump (現行、8セグメント) 読み込み中...")
    dump = load_all_dumps()

    print("[3/3] 突合中...")
    targets = []
    for r in rows:
        video_id, t_sec_s, detector, severity, stage, evidence, game_idx = r[:7]
        if detector not in ("D1a", "D1b"):
            continue
        if stage not in ("display", "both"):
            continue
        targets.append((float(t_sec_s), detector, evidence))
    print(f"  対象 (D1a/D1b display+both) = {len(targets)} 件")

    unmatched = 0
    results = []
    for t_sec, detector, evidence in targets:
        side = parse_side_from_evidence(evidence)
        idx, dt = nearest_match(dump, t_sec)
        if dt > MATCH_TOL_SEC:
            unmatched += 1
            continue
        state1 = str(dump["state1"][idx])
        state2 = str(dump["state2"][idx])
        side_state = state1 if side == "1P" else (state2 if side == "2P" else None)
        other_state = state2 if side == "1P" else (state1 if side == "2P" else None)
        results.append({"detector": detector, "t_sec": t_sec, "side_state": side_state,
                         "other_state": other_state})
    print(f"  突合失敗 (|dt|>{MATCH_TOL_SEC}s) = {unmatched} / {len(targets)}"
          " (旧dumpと現dumpでSTABLE更新の時刻が微妙にズレている場合ここに出る)")

    for detector in ("D1a", "D1b"):
        sub = [r for r in results if r["detector"] == detector]
        n = len(sub)
        if n == 0:
            print(f"{detector}: 0件 (突合成功分)")
            continue
        narrow = sum(1 for r in sub if r["side_state"] in CHAIN_NARROW)
        broad = sum(1 for r in sub if r["side_state"] in CHAIN_BROAD)
        both_stable = sum(1 for r in sub if r["side_state"] == "STABLE" and r["other_state"] == "STABLE")
        print(f"\n=== 旧{detector} (突合成功 n={n} / 全{sum(1 for t in targets if t[1]==detector)}) ===")
        print(f"  当該側が CHAIN/GRAVITY_SETTLE (狭義) = {narrow} ({narrow/n:.1%})")
        print(f"  当該側が NON_STABLE全体 (広義)        = {broad} ({broad/n:.1%})")
        print(f"  両側とも STABLE                        = {both_stable} ({both_stable/n:.1%})")


if __name__ == "__main__":
    main()
