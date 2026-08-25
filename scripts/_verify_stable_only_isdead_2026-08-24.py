"""is_dead 凍結盤面誤判定の遡及訂正 (`_retroactively_correct_dead_dump_rows`)
を、既存 timeline dump (8区間、data/verify/zenchi_render_slide_exit_guard_v2_
2026-08-22) に対して**本番実装そのもの**で検証するスクリプト (2026-08-24)。

再レンダリング不要 (dump は既存npzをそのまま読む・実装は
scripts.visualize_advantage_overlay._retroactively_correct_dead_dump_rows
を直接呼ぶ、独自の再実装は行わない)。

検証項目 (受け入れ条件1-4対応):
  [A] 陽性対照: 訂正前 (現行/フラグOFF相当) の「CHAINを含む凍結」再現
  [B] 是正確認: 訂正後の残存状況 (0にはならないが大幅減、理由は
      本体スクリプトのコメント参照)
  [C] 受け入れ条件2: t=6701.67-6717.03 の窓で 訂正後 is_dead1 が
      True にならないこと
  [D] 受け入れ条件4 (最重要): 全 game_idx 境界直前の値が訂正前後で
      変化しないこと (=見逃しゼロ、変化 0 件でなければならない)
  [E] 全域での is_dead=True 時間の変化 (副作用規模)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402

SEG_DIR = Path("data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22")
ALIVE_STATES = {"CHAIN", "GRAVITY_SETTLE", "TSUMO_FALL", "OJAMA_FALL"}


def load_all_rows() -> list[vao.TimelineDumpRow]:
    """8区間 npz を時刻順に読み、1本の TimelineDumpRow 列に結合する。"""
    rows: list[vao.TimelineDumpRow] = []
    for p in sorted(SEG_DIR.glob("*.npz")):
        _video_id, seg_rows = vao.load_timeline_dump(p)
        rows.extend(seg_rows)
    return rows


def extract_runs(rows: list[vao.TimelineDumpRow], side: str) -> list[dict]:
    is_dead = [getattr(r, f"is_dead{side}") for r in rows]
    state = [getattr(r, f"state{side}") for r in rows]
    game_idx = [r.game_idx for r in rows]
    t = [r.t_sec for r in rows]
    runs = []
    n = len(rows)
    i = 0
    while i < n:
        if not is_dead[i]:
            i += 1
            continue
        j = i
        while j < n and is_dead[j]:
            j += 1
        runs.append(dict(
            t_start=t[i], t_end=t[j - 1], duration=t[j - 1] - t[i],
            includes_alive_state=any(s in ALIVE_STATES for s in state[i:j]),
            game_idx_changed=game_idx[i] != game_idx[j - 1],
        ))
        i = j
    return runs


def main() -> None:
    rows = load_all_rows()
    print(f"読み込み総行数: {len(rows)}")

    corrected = vao._retroactively_correct_dead_dump_rows(rows)

    print("=" * 70)
    print("[A] 訂正前 (フラグOFF相当) の CHAINを含む凍結")
    for side in ("1", "2"):
        runs = extract_runs(rows, side)
        bad = [r for r in runs if r["includes_alive_state"] and not r["game_idx_changed"]]
        print(f"  {side}P: 疑わしい区間 = {len(bad)}件 "
              f"合計{sum(r['duration'] for r in bad):.1f}秒")

    print("=" * 70)
    print("[B] 訂正後 (本番実装) の残存状況")
    for side in ("1", "2"):
        runs = extract_runs(corrected, side)
        bad = [r for r in runs if r["includes_alive_state"] and not r["game_idx_changed"]]
        print(f"  {side}P: 残存疑わしい区間 = {len(bad)}件 "
              f"合計{sum(r['duration'] for r in bad):.1f}秒")

    print("=" * 70)
    print("[C] 受け入れ条件2: t=6701.67-6717.03 で 訂正後 is_dead1 が"
          " True にならないこと")
    hits = [r for r in corrected if 6701.67 <= r.t_sec <= 6717.03 and r.is_dead1]
    print(f"  違反行数: {len(hits)} (0が期待値)")
    assert len(hits) == 0, "受け入れ条件2 違反"

    print("=" * 70)
    print("[D] 受け入れ条件4 (最重要): game_idx 境界直前の値が訂正前後で不変")
    diff_count = 0
    for i in range(len(rows) - 1):
        if rows[i].game_idx != rows[i + 1].game_idx:
            for side in ("1", "2"):
                old_v = getattr(rows[i], f"is_dead{side}")
                new_v = getattr(corrected[i], f"is_dead{side}")
                if old_v != new_v:
                    diff_count += 1
                    print(f"    !! t={rows[i].t_sec:.2f} game_idx "
                          f"{rows[i].game_idx}->{rows[i+1].game_idx} side={side}P: "
                          f"旧={old_v} 新={new_v}")
    print(f"  境界直前で値が変化した件数: {diff_count} (0でなければならない)")
    assert diff_count == 0, "受け入れ条件4 (死亡見逃しゼロ) 違反"

    print("=" * 70)
    print("[E] 全域での is_dead=True 行数の変化 (副作用規模)")
    for side in ("1", "2"):
        old_n = sum(1 for r in rows if getattr(r, f"is_dead{side}"))
        new_n = sum(1 for r in corrected if getattr(r, f"is_dead{side}"))
        print(f"  {side}P: True行数 旧={old_n} -> 新={new_n} (削減={old_n - new_n}行)")

    print("=" * 70)
    print("全ての assert を通過。受け入れ条件2・4 (最重要) は本番実装で成立を確認済み。")


if __name__ == "__main__":
    main()
