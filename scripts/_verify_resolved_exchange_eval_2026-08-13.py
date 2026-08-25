"""#9 (両者発火後の勝率乱高下) 対処の効果検証 (使い捨て、コミット対象外)。

デモレビュー sceneA (source t=192〜204秒、両者発火。
scripts/_demo_review_scene_check_2026-08-13.sh と同一区間) を
--resolved-exchange-eval OFF/ON の2条件で実行し、実際に画面へ表示される
disp_adv (generate() の debug_history_out フック) の分散・値域を比較する。

OFF条件のフラグ構成は scripts/_gen_demo_fixed_2026-08-13.sh (改修後デモ1本目、
検収合格済み) と同一にする — #9 はその検収後に user が指摘した残存論点のため、
「合格済み構成 + 新フラグ」を新条件とするのが正しい比較 (フラグ差分だけを見る)。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from scripts.visualize_advantage_overlay import generate  # noqa: E402

# --- 診断用: ResolvedExchangeTracker.update の遷移を print する (使い捨て) ---
_orig_update = vao.ResolvedExchangeTracker.update


def _traced_update(self, r_p1, r_p2, snap, elapsed_sec):
    was_active = self._active
    active, just_deactivated = _orig_update(self, r_p1, r_p2, snap, elapsed_sec)
    if (not was_active and active) or just_deactivated or (was_active and active):
        pass
    if not was_active and active:
        print(f"    [activate] adv={self.hold_adv:.1f} p1={self.hold_p1:.3f}")
    if just_deactivated:
        print(f"    [deactivate] adv={self.hold_adv:.1f} p1={self.hold_p1:.3f}")
    return active, just_deactivated


vao.ResolvedExchangeTracker.update = _traced_update

_orig_resolve = vao.ResolvedExchangeTracker._resolve


def _traced_resolve(self, snap, elapsed_sec, score1, score2):
    _orig_resolve(self, snap, elapsed_sec, score1, score2)
    print(f"    [resolve] score1={score1:.0f} score2={score2:.0f} "
          f"-> adv={self.hold_adv:.1f}")


vao.ResolvedExchangeTracker._resolve = _traced_resolve

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
START_SEC = 192.0
END_SEC = 204.0
WARMUP_SEC = 10.0

# scripts/_gen_demo_fixed_2026-08-13.sh と同一 (検収合格済み構成)。
BASE_KWARGS = dict(
    enable_early_fire_reaction=True,
    enable_per_side_settled=True,
    disable_score_lead_bias=True,
    disable_pressure=True,
    enable_counter_remaining_time=True,
    enable_counter_defender_only=True,
    stable_majority_window=True,
    enable_ojama_fall_placement_override=True,
    enable_ojama_fall_entry_hardening=True,
    enable_ojama_fall_scoped_exit=True,
)


def run(resolved: bool) -> list[tuple[float, float]]:
    history: list[tuple[float, float]] = []
    generate(
        VIDEO, Path("data/verify/_unused_resolved_exchange_check.mp4"),
        max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=END_SEC, warmup_sec=WARMUP_SEC,
        render=False, debug_history_out=history,
        enable_resolved_exchange_eval=resolved,
        **BASE_KWARGS,
    )
    return history


def report(label: str, history: list[tuple[float, float]]) -> None:
    vals = [v for _, v in history]
    if not vals:
        print(f"[{label}] n=0 (データ無し)")
        return
    print(
        f"[{label}] n={len(vals)} "
        f"stdev={statistics.pstdev(vals):.3f} "
        f"range={max(vals) - min(vals):.3f} "
        f"min={min(vals):.2f} max={max(vals):.2f}"
    )
    # 隣接差分の絶対値平均 (フレーム間の跳ね幅=乱高下の直接的指標)
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    if diffs:
        print(f"    mean_abs_frame_delta={statistics.mean(diffs):.3f}")


if __name__ == "__main__":
    import os
    if os.environ.get("DEBUG_ONLY_ON") == "1":
        hist_on = run(resolved=True)
        report("ON  (--resolved-exchange-eval あり)", hist_on)
    else:
        hist_off = run(resolved=False)
        hist_on = run(resolved=True)
        report("OFF (--resolved-exchange-eval なし)", hist_off)
        report("ON  (--resolved-exchange-eval あり)", hist_on)
