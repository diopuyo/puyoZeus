"""検収セルフベリファイ (確認デモ = final2, review_demo_2026-08-12.mp4 3試合):
指摘9/10/12 の回帰確認を --no-render 単発実行でまとめて行う (read-only)。

_gen_demo_fixed_2026-08-13.sh (HEAD 78426de、指摘12修正1〜4コミット済み) と
完全同一のフラグ構成で generate(render=False) を実行し、以下を出力する:
  - 指摘12窓 (source 232-238s): ResolvedExchangeTracker._resolve のホールド
    確定値 (hold_adv/hold_p1/defender_side/incoming/defender_prob) + 表示値
    disp_adv の高解像度サンプル (0.2秒刻み)。
  - 指摘9/10窓 (source 193-201s, 両者発火ホールド): 表示値 disp_adv の
    0.2秒刻みサンプル + ホールド区間内の分散 (乱高下なしの確認)。
  - 全域チェック: dump_timeline (settled更新ごと) から極端値割合・主因分布を
    集計 (シーン修正が全域を悪化させていないかの一次スクリーニング)。

本体コード (scripts/visualize_advantage_overlay.py) は変更しない
(計装は monkeypatch のみ)。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
from scripts.visualize_advantage_overlay import load_timeline_dump  # noqa: E402

START_SEC = 162.0
END_SEC = 310.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_selfverify_final2.mp4")
DUMP_PATH = Path("data/verify/demo_fixed_2026-08-13/selfverify_final2_fullscan_2026-08-14.npz")

ISSUE12_LO, ISSUE12_HI = 232.0, 238.0
ISSUE910_LO, ISSUE910_HI = 193.0, 201.0

_CUR_T = {"t": None}
RESOLVE_LOG: list[dict] = []


def _patch_t_tracker() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    orig = RecognitionPipeline.update

    def patched(self, fi, t, frame):
        _CUR_T["t"] = t
        return orig(self, fi, t, frame)

    RecognitionPipeline.update = patched


def _patch_resolve_logger() -> None:
    orig_resolve = ov.ResolvedExchangeTracker._resolve

    def patched(self, snap, elapsed_sec, score1, score2):
        ev1_cc = self._ev1.chain_count if self._ev1 else None
        ev2_cc = self._ev2.chain_count if self._ev2 else None
        orig_resolve(self, snap, elapsed_sec, score1, score2)
        RESOLVE_LOG.append({
            "t": _CUR_T["t"], "score1": score1, "score2": score2,
            "ev1_cc": ev1_cc, "ev2_cc": ev2_cc,
            "hold_adv": self.hold_adv, "hold_p1": self.hold_p1,
            "defender_side": self.hold_defender_side,
            "incoming": self.hold_incoming_ojama,
            "defender_prob": self.hold_defender_prob,
        })

    ov.ResolvedExchangeTracker._resolve = patched


def main() -> int:
    _patch_t_tracker()
    _patch_resolve_logger()
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
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
        dump_timeline_path=DUMP_PATH,
    )

    print(f"n_history_samples={len(history)}")

    print("\n===== 指摘12窓 (source 232-238s): _resolve ホールド確定値 =====")
    for e in RESOLVE_LOG:
        t = e["t"]
        if t is None or not (ISSUE12_LO <= t <= ISSUE12_HI):
            continue
        p1pct = e["hold_p1"] * 100.0
        print(f"t={t:.2f} score1={e['score1']} score2={e['score2']} "
              f"ev1_cc={e['ev1_cc']} ev2_cc={e['ev2_cc']} "
              f"hold_adv={e['hold_adv']:+.2f} hold_p1(1P)={p1pct:.2f}% "
              f"defender={e['defender_side']} incoming={e['incoming']} "
              f"defender_prob={e['defender_prob']}")

    print("\n===== 指摘12窓 (source 232-238s): 表示値 disp_adv (0.2秒刻み) =====")
    last_bucket = None
    for t_sec, disp_adv in history:
        if not (ISSUE12_LO <= t_sec <= ISSUE12_HI):
            continue
        bucket = round(t_sec * 5) / 5.0
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = 0.5 + disp_adv / 200.0
        print(f"t={t_sec:7.2f} disp_adv={disp_adv:+7.2f} p1(1P)={p1*100:5.1f}% "
              f"p2(2P)={100 - p1*100:5.1f}%")

    print("\n===== 指摘9/10窓 (source 193-201s、両者発火ホールド): 表示値 disp_adv (0.2秒刻み) =====")
    window_vals: list[float] = []
    last_bucket = None
    for t_sec, disp_adv in history:
        if not (ISSUE910_LO <= t_sec <= ISSUE910_HI):
            continue
        bucket = round(t_sec * 5) / 5.0
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = 0.5 + disp_adv / 200.0
        window_vals.append(disp_adv)
        print(f"t={t_sec:7.2f} disp_adv={disp_adv:+7.2f} p1(1P)={p1*100:5.1f}%")
    if window_vals:
        import statistics
        print(f"\n窓内サンプル数={len(window_vals)} "
              f"disp_adv range=[{min(window_vals):+.2f}, {max(window_vals):+.2f}] "
              f"stdev={statistics.pstdev(window_vals):.3f}")

    print("\n===== 全域チェック (dump_timeline, source 162-310s) =====")
    video_id, rows = load_timeline_dump(DUMP_PATH)
    n_total = len(rows)
    n_extreme_p1 = sum(1 for r in rows if r.p1 >= 0.97)
    n_extreme_p2 = sum(1 for r in rows if r.p1 <= 0.03)
    print(f"video_id={video_id} n_settled_updates={n_total}")
    print(f"極端値(p1>=97%): {n_extreme_p1}/{n_total} ({n_extreme_p1/n_total*100:.1f}%)")
    print(f"極端値(p1<=3%): {n_extreme_p2}/{n_total} ({n_extreme_p2/n_total*100:.1f}%)")
    driver_counter = Counter(r.drivers_top1_name for r in rows)
    print("主因(top1) 列名分布 (上位10):")
    for name, cnt in driver_counter.most_common(10):
        print(f"  {name:30s} {cnt:5d} ({cnt/n_total*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
