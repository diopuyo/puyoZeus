"""指摘14 診断: ResolvedExchangeTracker._resolve/_amplify_decisive の内部値を
monkeypatch で記録する (本体コード改変なし)。

対象窓: t=193〜201秒 (final3_m1 内で p2=18.9% が5.2秒間 (195.33-200.53) 凍結
表示された区間、scripts/_diag_issue14_scene_2026-08-15.py の結果より特定済み)。

記録する内部値:
  - _resolve() 呼び出し時点の gen1/gen2 (score→おじゃま換算個数)
  - resolve_mutual_exchange の結果: dropped_to_p1/p2 (このターンで実際に
    盤面へ物理着地した量、OJAMA_MAX_DROP_PER_TURN=30上限) と leftover_p1/p2
    (forecastのみに残る、着地しない繰越分)
  - board_p1_after/board_p2_after の実際のおじゃまセル数 (モデルに渡る盤面)
  - _score_advantage の生 adv/p1 (amplify前)
  - _amplify_decisive が加える amp の値・defender_prob・incoming (増幅の入力)
  - 最終 hold_adv/hold_p1 (表示値)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
from src import indicators_v2 as iv  # noqa: E402
from src.exchange_virtual_board import resolve_mutual_exchange  # noqa: E402

START_SEC = 162.0  # 本番 (final3) と同一のウォームアップ起点に揃える
                    # (2026-08-15 事故: 193sから起動すると online_hsv 再較正の
                    # タイミングが変わり disp_adv が別物になった、再現条件は
                    # 常に本番と一致させること)。
END_SEC = 202.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue14_internals.mp4")

_records: list[dict] = []

_orig_resolve = ov.ResolvedExchangeTracker._resolve
_orig_amplify = ov.ResolvedExchangeTracker._amplify_decisive


def _patched_resolve(self, snap, elapsed_sec, score1, score2):
    gen1 = iv._score_to_ojama_count(score1, elapsed_sec)
    gen2 = iv._score_to_ojama_count(score2, elapsed_sec)
    result = resolve_mutual_exchange(
        self._ev1.before_board, self._ev2.before_board, gen1, gen2,
        snap.pending_p1, snap.pending_p2,
    )
    rec = {
        "kind": "resolve", "t_sec": self._t_sec,
        "score1": score1, "score2": score2,
        "gen1": gen1, "gen2": gen2,
        "pending_p1_in": snap.pending_p1, "pending_p2_in": snap.pending_p2,
        "dropped_to_p1": result.dropped_to_p1, "dropped_to_p2": result.dropped_to_p2,
        "leftover_p1": result.leftover_p1, "leftover_p2": result.leftover_p2,
        "board_p1_after_ojama": int(iv.board_ojama_count(result.board_p1_after).raw),
        "board_p2_after_ojama": int(iv.board_ojama_count(result.board_p2_after).raw),
    }
    _orig_resolve(self, snap, elapsed_sec, score1, score2)
    rec["hold_adv_after_resolve"] = self.hold_adv
    rec["hold_p1_after_resolve"] = self.hold_p1
    rec["incoming_total_p1"] = self._incoming_total_p1
    rec["incoming_total_p2"] = self._incoming_total_p2
    _records.append(rec)
    print(f"[_resolve] t={rec['t_sec']:.2f} score1={score1:.0f} score2={score2:.0f} "
          f"gen1={gen1} gen2={gen2} pend_in=({snap.pending_p1},{snap.pending_p2}) "
          f"dropped=({result.dropped_to_p1},{result.dropped_to_p2}) "
          f"leftover=({result.leftover_p1},{result.leftover_p2}) "
          f"board_after_ojama=({rec['board_p1_after_ojama']},{rec['board_p2_after_ojama']}) "
          f"incoming_total=({self._incoming_total_p1:.0f},{self._incoming_total_p2:.0f}) "
          f"-> hold_p1={self.hold_p1*100:.1f}%")


def _patched_amplify(self, adv, result):
    adv_before = adv
    adv_after, p1_after = _orig_amplify(self, adv, result)
    print(f"[_amplify_decisive] t={self._t_sec:.2f} adv_before={adv_before:+.2f} "
          f"defender_side={self.hold_defender_side} incoming={self.hold_incoming_ojama:.0f} "
          f"defender_prob={self.hold_defender_prob:.3f} "
          f"adv_after={adv_after:+.2f} p1_after={p1_after*100:.1f}%")
    return adv_after, p1_after


_orig_reeval = ov.ResolvedExchangeTracker._reevaluate_live_defender


def _patched_reeval(self, b1, b2, snap=None):
    before_hold_adv = self.hold_adv
    _orig_reeval(self, b1, b2, snap)
    if self.hold_adv == before_hold_adv:
        return  # 間引きでスキップされた (COUNTER_RECOMPUTE_INTERVAL_SEC 未満)
    defender_side, incoming = self._decisive_defender(self._result)
    live_board = b1 if defender_side == "1P" else b2
    ojama_on_live_board = int(iv.board_ojama_count(live_board).raw) if live_board else -1
    print(f"[_reevaluate_live_defender] t={self._t_sec:.2f} "
          f"defender_side={defender_side} incoming_total={incoming:.0f} "
          f"live_board_ojama_before_landing={ojama_on_live_board} "
          f"hold_adv {before_hold_adv:+.2f} -> {self.hold_adv:+.2f} "
          f"(p1 {ov.adv_to_winprob(before_hold_adv)*100:.1f}% -> {self.hold_p1*100:.1f}%)")


ov.ResolvedExchangeTracker._resolve = _patched_resolve
ov.ResolvedExchangeTracker._amplify_decisive = _patched_amplify
ov.ResolvedExchangeTracker._reevaluate_live_defender = _patched_reeval


def main() -> int:
    history: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=END_SEC,
        show_recognition=True,
        enable_early_fire_reaction=True, enable_per_side_settled=True,
        disable_score_lead_bias=True, disable_pressure=True,
        enable_counter_remaining_time=True, enable_counter_defender_only=True,
        stable_majority_window=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_resolved_live_defender=True,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
    )
    print(f"\n[history] n={len(history)}")
    last_bucket = None
    for t_abs, disp_adv in history:
        bucket = round(t_abs / 0.2)
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = ov.adv_to_winprob(disp_adv)
        print(f"  t={t_abs:7.2f} disp_adv={disp_adv:+7.2f} p1={p1*100:5.1f}% p2={100-p1*100:5.1f}%")
    print("\nDONE_ISSUE14_INTERNALS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
