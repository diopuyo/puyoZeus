"""残存 is_dead 持続 (137件, 478.5秒) の内訳分類 (2026-08-24)。

読み込み専用の診断スクリプト。data/verify/zenchi_render_slide_exit_guard_v2_
2026-08-22/*.npz を `scripts.visualize_advantage_overlay
._retroactively_correct_dead_dump_rows` (本番実装そのもの) で訂正した後、
残る is_dead=True 連続区間 (「疑わしい」= 持続>=1秒・own state に生存状態を
含む・試合未終了) を機構別に粗く分類する。

分類軸 (①-④はコード上判別可能な範囲、確定的な分類ではなく「有力候補」):
  A. 自分側 CHAIN を含む   -> 本タスクの主題 (STABLE誤認からのCHAIN検知遅延)
     が疑われる。state に CHAIN が出現するが、それでも is_dead=True が
     残っている = 訂正ロジック (次のSTABLE値で遡及) が「次のSTABLE到達点」
     をさらに超えて残存しているケース、または CHAIN 区間そのものを
     is_dead=True で覆っているケース。
  B. 自分側 OJAMA_FALL/TSUMO_FALL/GRAVITY_SETTLE のみ (CHAINなし)
     -> 別の「NON-STABLE中に確定盤面が古いまま」系 (既知のW30等)。
  C. 相手側だけ生存状態 (own は終始 STABLE) -> 自分側の凍結board自体が
     真に満杯 (=本当に窒息に近い/満杯) で、相手の連鎖中に是正が及んで
     いないケース。
  D. game_idx 変化なしだが両者 STABLE のみ (alive_frac=0 のはずだが
     念のため計上) -> 抽出条件と矛盾、要目視。

出力: logs/_diag_residual_isdead_classify_2026-08-24/classified_runs.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402

SEG_DIR = Path("data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22")
OUT_DIR = Path("logs/_diag_residual_isdead_classify_2026-08-24")
ALIVE_STATES = {"CHAIN", "GRAVITY_SETTLE", "TSUMO_FALL", "OJAMA_FALL"}


def load_all_rows() -> list["vao.TimelineDumpRow"]:
    rows: list["vao.TimelineDumpRow"] = []
    for p in sorted(SEG_DIR.glob("*.npz")):
        _video_id, seg_rows = vao.load_timeline_dump(p)
        rows.extend(seg_rows)
    return rows


def classify_run(own_states: set, other_states: set) -> str:
    """own/other side の観測 state 集合から分類ラベルを返す。"""
    if "CHAIN" in own_states:
        return "A_own_chain"
    if own_states & {"OJAMA_FALL", "TSUMO_FALL", "GRAVITY_SETTLE"}:
        return "B_own_nonchain_action"
    if other_states & ALIVE_STATES:
        return "C_other_side_active_only"
    return "D_other_unexplained"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_all_rows()
    corrected = vao._retroactively_correct_dead_dump_rows(rows)
    n = len(corrected)

    out_rows: list[dict] = []
    for side, is_dead_key, state_key, other_key in (
        ("1", "is_dead1", "state1", "state2"),
        ("2", "is_dead2", "state2", "state1"),
    ):
        is_dead = [getattr(r, is_dead_key) for r in corrected]
        state = [getattr(r, state_key) for r in corrected]
        other_state = [getattr(r, other_key) for r in corrected]
        game_idx = [r.game_idx for r in corrected]
        t = [r.t_sec for r in corrected]
        i = 0
        while i < n:
            if not is_dead[i]:
                i += 1
                continue
            j = i
            while j < n and is_dead[j]:
                j += 1
            own_states = set(state[i:j])
            other_states = set(other_state[i:j])
            alive_own = any(s in ALIVE_STATES for s in own_states)
            gi_changed = game_idx[i] != game_idx[j - 1]
            dur = t[j - 1] - t[i]
            if not gi_changed and alive_own:
                label = classify_run(own_states, other_states)
                out_rows.append(dict(
                    side=side, t_start=round(t[i], 3), t_end=round(t[j - 1], 3),
                    duration=round(dur, 3), n_rows=j - i,
                    own_states=",".join(sorted(own_states)),
                    other_states=",".join(sorted(other_states)),
                    label=label,
                ))
            i = j

    out_csv = OUT_DIR / "classified_runs.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "side", "t_start", "t_end", "duration", "n_rows",
            "own_states", "other_states", "label",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"分類件数: {len(out_rows)}")
    from collections import Counter
    label_counts = Counter(r["label"] for r in out_rows)
    label_durs: dict[str, float] = {}
    for r in out_rows:
        label_durs[r["label"]] = label_durs.get(r["label"], 0.0) + r["duration"]
    for label in sorted(label_counts):
        print(f"  {label}: {label_counts[label]}件 "
              f"合計{label_durs[label]:.1f}秒")
    print(f"[保存] {out_csv}")


if __name__ == "__main__":
    main()
