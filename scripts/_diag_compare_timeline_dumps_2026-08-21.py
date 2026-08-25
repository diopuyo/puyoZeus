"""分割レンダ(ウォームアップ付き) vs 通しレンダの判定値一致検証 (2026-08-21)。

`visualize_advantage_overlay.save_timeline_dump`/`load_timeline_dump` を
そのまま再利用する (本体は変更しない、読み込み専用)。

reference (通し, t=3300起点) と各 warmup 変種 (t=3326起点、warmup=0/5/15/26秒)
を、重なる時刻範囲 [3330, 3360] で行ごとに比較する (t_sec を key に最も近い
行を対応付け、adv_raw/b1_hash/b2_hash/p1/pending_p1/pending_p2 が一致するか)。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_advantage_overlay import load_timeline_dump  # noqa: E402

DUMP_DIR = PROJECT_ROOT / "data" / "verify" / "zenchi_warmup_2026-08-21"
COMPARE_FROM = 3330.0  # 境界(3326直後)から数秒後を比較窓の開始にする
COMPARE_TO = 3360.0

FIELDS = ["adv_raw", "adv_ema", "p1", "p1_raw", "pending_p1", "pending_p2",
          "room1", "room2", "b1_hash", "b2_hash", "score1", "score2"]


def _load(name: str):
    path = DUMP_DIR / f"{name}.npz"
    if not path.exists():
        print(f"[skip] {path} が無い")
        return None
    _, rows = load_timeline_dump(path)
    return [r for r in rows if COMPARE_FROM <= r.t_sec <= COMPARE_TO]


def _row_dict(row) -> dict:
    return {f: getattr(row, f) for f in FIELDS}


def compare(ref_rows, test_rows, label: str) -> None:
    if ref_rows is None or test_rows is None:
        print(f"[{label}] データ不足でスキップ")
        return
    n = min(len(ref_rows), len(test_rows))
    if len(ref_rows) != len(test_rows):
        print(f"[{label}] 行数不一致: ref={len(ref_rows)} test={len(test_rows)} "
              f"(先頭{n}行のみ比較)")
    mismatches = []
    for i in range(n):
        rr, tr = ref_rows[i], test_rows[i]
        if abs(rr.t_sec - tr.t_sec) > 0.05:
            mismatches.append((i, "t_sec_misaligned", rr.t_sec, tr.t_sec))
            continue
        for f in FIELDS:
            rv, tv = getattr(rr, f), getattr(tr, f)
            if isinstance(rv, float):
                if abs(rv - tv) > 1e-6:
                    mismatches.append((i, f, rv, tv))
            elif rv != tv:
                mismatches.append((i, f, rv, tv))
    print(f"\n[{label}] 比較行数={n} 不一致={len(mismatches)}")
    if mismatches:
        first_mismatch_t = ref_rows[mismatches[0][0]].t_sec
        print(f"  最初の不一致: t={first_mismatch_t:.2f}s "
              f"field={mismatches[0][1]} ref={mismatches[0][2]} test={mismatches[0][3]}")
        for m in mismatches[:5]:
            print(f"    idx={m[0]} field={m[1]} ref={m[2]} test={m[3]}")
        last_mismatch_t = ref_rows[mismatches[-1][0]].t_sec
        print(f"  最後の不一致: t={last_mismatch_t:.2f}s")
    else:
        print("  ★完全一致 (指定フィールド全て)")


def main() -> None:
    ref = _load("ref")
    print(f"ref行数(全体)={'N/A' if ref is None else len(ref)} "
          f"範囲[{COMPARE_FROM},{COMPARE_TO}]")
    for name in ("w0", "w5", "w15", "w26"):
        test = _load(name)
        compare(ref, test, name)


if __name__ == "__main__":
    main()
