"""予測入力の漏斗を、保存済み成果物の再生だけで測る (2026-09-17、検証の梯子 段2)。

## 何を確かめる台か
G3 の実走で「予測入力 (conditional_current_after_original_J)」が 0 件だった理由は、
**採用の門が読む窓が G2 のまま ([32494, 36900]) 残っていた配線の間違い**だと分かった。
この台は、窓を直したら何件観測できる見込みなのかを、**実走を回さずに**見積もる。

## 原則3: 新しい台はまず既知の失敗を再現する
先に v12 の既知の値を再現できることを示す。再現できない台の結果は使えない。
  - 採用 1 件 (1P frame 34796)
  - 却下 1,096 件 (全件 outside_original_directional_scope、1P 508 / 2P 588)
  - make 到達 182 件、うち made 0 件
  - make の却下内訳 131 (tail 未消費) + 51 (STABLE 以外)

## 見積もりの限界 (明記)
`fresh` と raw==confirmed は保存物から復元できない。したがって出せるのは
**上限**であって、実イベント数ではない。上限 ≫ 実測なら「窓より上流の門が塞いでいる」、
上限 ≈ 0 なら「その区間に条件が無い」と読める。

読み取り専用。既存の成果物を一切変更しない。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# v12 の既知の値。台がこれを再現できなければ、以降の見積もりは信用しない。
KNOWN_V12 = dict(
    adoptions=1,
    rejections=1096,
    rejection_sides={"1P": 508, "2P": 588},
    native_scope=(32494, 36900),
    make_invoked=182,
    made=0,
)

# 候補が成立しうるのは確定状態のときだけ (conditional_current.py:116)。
STABLE_STATE = "stable"


def load_adoption(run_dir: Path) -> dict:
    """採用と却下の実測を読む。"""
    data = json.loads((run_dir / "BASELINE_ADOPTION_STATUS.json").read_text(encoding="utf-8"))
    rejects = data.get("unwitnessed_appends") or []
    sides = collections.Counter(r.get("side") for r in rejects)
    reasons = collections.Counter(r.get("reason") for r in rejects)
    frames = [int(r["frame"]) for r in rejects if "frame" in r]
    native = None
    if rejects:
        native = (int(rejects[0]["native_first"]), int(rejects[0]["native_last"]))
    return dict(
        adoptions=len(data.get("adoptions") or []),
        adoption_frames=[int(a["adoption_frame"]) for a in (data.get("adoptions") or [])],
        adoption_sides=[str(a.get("segment_id", "")).split(":")[0]
                        for a in (data.get("adoptions") or [])],
        rejections=len(rejects),
        rejection_sides=dict(sides),
        rejection_reasons=dict(reasons),
        rejection_frame_range=(min(frames), max(frames)) if frames else None,
        native_scope=native,
    )


def load_make_calls(run_dir: Path) -> dict:
    """make() の呼出と結果を読む (21MB あるので必要な列だけ数える)。"""
    path = run_dir / "CONDITIONAL_CURRENT_CALLS.json"
    if not path.is_file():
        return dict(rows=0, make_invoked=0, made=0, sides={}, frame_range=None)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else (data.get("rows") or data.get("calls") or [])
    invoked = [r for r in rows if r.get("make_invoked")]
    made = [r for r in rows if r.get("made")]
    frames = [int(r["frame"]) for r in invoked if "frame" in r]
    return dict(
        rows=len(rows),
        make_invoked=len(invoked),
        made=len(made),
        sides=dict(collections.Counter(r.get("side") for r in invoked)),
        frame_range=(min(frames), max(frames)) if frames else None,
    )


def scan_frames(run_dir: Path) -> dict:
    """frames.jsonl を1行ずつ読み、side ごとの確定状態を数える。

    197MB あるので全部をメモリに載せない。必要なのは frame_idx / side / state だけ。
    """
    path = run_dir / "frames.jsonl"
    per_side: dict[str, dict] = {}
    total = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"kind": "frame_side"' not in line:
                continue
            row = json.loads(line)
            side = str(row.get("side"))
            frame = int(row.get("frame_idx", -1))
            state = str(row.get("state"))
            entry = per_side.setdefault(side, dict(
                steps=0, stable=0, first_frame=frame, last_frame=frame,
                states=collections.Counter(), stable_runs=[], _run=0,
            ))
            entry["steps"] += 1
            entry["last_frame"] = frame
            entry["states"][state] += 1
            if state == STABLE_STATE:
                entry["stable"] += 1
                entry["_run"] += 1
            else:
                if entry["_run"] > 0:
                    entry["stable_runs"].append(entry["_run"])
                entry["_run"] = 0
            total += 1
    for entry in per_side.values():
        if entry["_run"] > 0:
            entry["stable_runs"].append(entry["_run"])
        entry.pop("_run", None)
        entry["states"] = dict(entry["states"].most_common(6))
    return dict(total_side_steps=total, per_side=per_side)


def reproduce_known(adoption: dict, calls: dict) -> list[str]:
    """既知の失敗を再現できるかを確かめる。ずれたら名前つきで返す。"""
    problems: list[str] = []
    if adoption["adoptions"] != KNOWN_V12["adoptions"]:
        problems.append("採用件数 %s (既知 %s)" % (adoption["adoptions"], KNOWN_V12["adoptions"]))
    if adoption["rejections"] != KNOWN_V12["rejections"]:
        problems.append("却下件数 %s (既知 %s)" % (adoption["rejections"], KNOWN_V12["rejections"]))
    if adoption["rejection_sides"] != KNOWN_V12["rejection_sides"]:
        problems.append("却下の側の内訳 %s (既知 %s)"
                        % (adoption["rejection_sides"], KNOWN_V12["rejection_sides"]))
    if adoption["native_scope"] != KNOWN_V12["native_scope"]:
        problems.append("窓 %s (既知 %s)" % (adoption["native_scope"], KNOWN_V12["native_scope"]))
    if calls["make_invoked"] != KNOWN_V12["make_invoked"]:
        problems.append("make到達 %s (既知 %s)" % (calls["make_invoked"], KNOWN_V12["make_invoked"]))
    if calls["made"] != KNOWN_V12["made"]:
        problems.append("made %s (既知 %s)" % (calls["made"], KNOWN_V12["made"]))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="実走の成果物ディレクトリ")
    parser.add_argument("--skip-frames", action="store_true",
                        help="frames.jsonl の走査を省く (197MB・数分かかるため)")
    args = parser.parse_args()

    adoption = load_adoption(args.run_dir)
    calls = load_make_calls(args.run_dir)

    print("=== 既知の失敗を再現できるか (原則3) ===")
    problems = reproduce_known(adoption, calls)
    if problems:
        print("  **再現できない**。この台の見積もりは使えない:")
        for p in problems:
            print("    - " + p)
    else:
        print("  再現できた。採用%d / 却下%d / make到達%d / made%d / 窓%s"
              % (adoption["adoptions"], adoption["rejections"],
                 calls["make_invoked"], calls["made"], adoption["native_scope"]))

    print()
    print("=== 窓が塞いでいた量 (母数つき) ===")
    first, last = adoption["native_scope"] or (None, None)
    print("  採用の門が読む窓: [%s, %s] (G2 の値)" % (first, last))
    print("  却下 %d 件、理由の内訳: %s" % (adoption["rejections"], adoption["rejection_reasons"]))
    print("  却下の側の内訳: %s" % adoption["rejection_sides"])
    lo, hi = adoption["rejection_frame_range"] or (None, None)
    print("  却下された frame の範囲: %s 〜 %s" % (lo, hi))
    if first is not None and hi is not None:
        print("  → 却下の最大 frame %d < 窓の開始 %d : %s"
              % (hi, first, "全件が窓より手前" if hi < first else "窓の中にも却下がある"))
    print("  採用できた frame: %s (側 %s)"
          % (adoption["adoption_frames"], adoption["adoption_sides"]))

    print()
    print("=== make() の到達 ===")
    print("  記録行 %d / make到達 %d / made %d" % (calls["rows"], calls["make_invoked"], calls["made"]))
    print("  側の内訳: %s / frame範囲: %s" % (calls["sides"], calls["frame_range"]))
    if calls["rows"]:
        print("  到達率: %d / %d = %.2f%%"
              % (calls["make_invoked"], calls["rows"], 100 * calls["make_invoked"] / calls["rows"]))

    if not args.skip_frames:
        print()
        print("=== 窓を直したときの母集団 (上限。fresh と raw==confirmed は復元不能) ===")
        frames = scan_frames(args.run_dir)
        print("  side-step 総数 %d (母数)" % frames["total_side_steps"])
        for side, entry in sorted(frames["per_side"].items()):
            runs = entry["stable_runs"]
            print("  %s: %d step / 確定 %d (%.1f%%) / 確定が続いた区間 %d本 (最長%d step)"
                  % (side, entry["steps"], entry["stable"],
                     100 * entry["stable"] / entry["steps"] if entry["steps"] else 0.0,
                     len(runs), max(runs) if runs else 0))
            print("     状態の内訳: %s" % entry["states"])
        print()
        print("  読み方: 現状は 1P の 182 step (全体の 0.50%%) しか make に到達していない。")
        print("          窓を直すと母集団は上の side-step 全体になる。")
        print("          ただし採用は side あたり 1 回で終わる可能性があり (SLOT_COUNT=1)、")
        print("          そこは保存物からは分からない。実走の漏斗票で測る。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
