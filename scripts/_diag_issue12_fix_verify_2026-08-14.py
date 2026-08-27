"""指摘12 修正3点 (docs/DEMO_REVIEW_2026-08-13.md #12) の検証用スクリプト。

修正1 (時間予算を#3新方式へ統一) + 修正2 (RESOLVED_AMPLIFY_SCALE 新設) を
本体コード (scripts/visualize_advantage_overlay.py) に実装済みの状態で、
「旧実装 (修正前と同じ計算式)」と「新実装 (現状のコード)」を同一動画・同一
生成条件で比較する (計装ラッパー方式、本体コードは変更しない)。

対象窓 (source絶対秒、review_demo_2026-08-12.mp4):
  - 指摘10/デモ1(t=33-38) 相当: 195.0-200.0s (1試合目、両者発火)
  - 指摘12 相当: 234.87s (2試合目、2Pの本線発火)
両方とも同一の generate() 実行 (start_sec=162, end_sec=237) でカバーできる
(2試合目は1試合目に続くため、状態連続性のため162から通しで実行する)。

「旧実装」は ResolvedExchangeTracker._amplify_decisive を、修正前と同じ式
(iv.estimate_chain_anim_duration_sec(観測連鎖数) + COUNTER_SCALE全量) に
一時的にモンキーパッチして再現する (本体コードのロールバックはしない)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

iv = ov.iv

START_SEC = 162.0
END_SEC = 237.0
WINDOWS = {
    "issue10_demo1_t33-38": (195.0, 200.0),
    "issue12_t234.87": (234.0, 236.0),
}

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue12_fix_verify.mp4")

_CUR_T = {"t": None}


def _patch_hcache_stub() -> None:
    """HeavyAdvCache.update をスタブ化 (計算コスト削減、issue12 v5診断と同じ手法)。

    ライブ per-frame ブレンド用の値だが、決着ホールド中はこの経路自体が
    丸ごとスキップされる (resolved_hold_freezes_settled) ため、本検証の対象
    (ResolvedExchangeTracker._resolve/_amplify_decisive) の値には一切影響
    しない。saturated_chain_count/ukeyasusa/reach_fire_power はいずれも
    連鎖シミュレーション込みで重いため、ここをゼロ固定にして高速化する。"""

    def patched(self, b1, b2, snap, sp1, sp2, elapsed):
        return (0.0, 0.0, [], 0.0, 0.0, 0.0, 0.0)

    ov.HeavyAdvCache.update = patched


def _tag_decisive_counter_tracker() -> None:
    """ResolvedExchangeTracker.__init__ 直後に内部 counter_tracker へ目印を付ける
    (issue12 v5診断と同じ手法。ライブ per-frame 用インスタンスと区別するため)。"""
    orig_init = ov.ResolvedExchangeTracker.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._counter_tracker._is_decisive_amplify_tracker = True

    ov.ResolvedExchangeTracker.__init__ = patched_init


def _patch_live_counter_tracker_stub() -> None:
    """ライブ per-frame 用 (main loop の counter_tracker インスタンス) の MC
    計算だけをスタブ化する (計算コスト削減)。ResolvedExchangeTracker 内部の
    決着増幅専用 MC (_tag_decisive_counter_tracker でタグ付け済み) は
    本物のまま計算する — 本検証の対象値には一切影響しない。"""
    orig = ov.CounterReachTracker._update_defender_only

    def patched(self, b1, b2, budget_sec, next1, next2, t_sec, defender_side, threshold_ojama):
        if not getattr(self, "_is_decisive_amplify_tracker", False):
            self.last_hands = 0.0
            return (0.0, float("nan"), float("nan"))
        return orig(self, b1, b2, budget_sec, next1, next2, t_sec, defender_side, threshold_ojama)

    ov.CounterReachTracker._update_defender_only = patched


def _patch_t_tracker() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    orig = RecognitionPipeline.update

    def patched(self, fi, t, frame):
        _CUR_T["t"] = t
        return orig(self, fi, t, frame)

    RecognitionPipeline.update = patched


def _legacy_amplify_decisive(self, adv, result):
    """修正前 (2026-08-13時点) の _amplify_decisive を再現する (比較用)。

    時間予算 = iv.estimate_chain_anim_duration_sec(観測連鎖数) の直呼び
    (経過時間控除なし)、強度 = 共用 COUNTER_SCALE の全量。
    """
    defender_side, incoming = self._decisive_defender(result)
    if defender_side is None:
        return adv, ov.adv_to_winprob(adv)
    attacker_event = self._ev2 if defender_side == "1P" else self._ev1
    budget = iv.estimate_chain_anim_duration_sec(float(attacker_event.chain_count))
    _, cp1, cp2 = self._counter_tracker.update(
        result.board_p1_after, result.board_p2_after, budget,
        defender_side=defender_side, threshold_ojama=incoming,
    )
    defender_prob = cp1 if defender_side == "1P" else cp2
    amp = ov._counter_defender_adv(
        defender_side, defender_prob, incoming,
        result.board_p1_after, result.board_p2_after,
        scale=ov.COUNTER_SCALE,  # 修正前は共用スケールを全量使用
    )
    adv = max(-100.0, min(100.0, adv + amp))
    return adv, ov.adv_to_winprob(adv)


def _install_logging_patch(log: list) -> None:
    orig_resolve = ov.ResolvedExchangeTracker._resolve

    def patched_resolve(self, snap, elapsed_sec, score1, score2):
        ev1_cc = self._ev1.chain_count if self._ev1 else None
        ev2_cc = self._ev2.chain_count if self._ev2 else None
        orig_resolve(self, snap, elapsed_sec, score1, score2)
        log.append({
            "t": _CUR_T["t"], "score1": score1, "score2": score2,
            "ev1_cc": ev1_cc, "ev2_cc": ev2_cc,
            "hold_adv": self.hold_adv, "hold_p1": self.hold_p1,
            "defender_side": self.hold_defender_side,
            "incoming": self.hold_incoming_ojama,
            "defender_prob": self.hold_defender_prob,
        })

    ov.ResolvedExchangeTracker._resolve = patched_resolve


def _run(variant: str, end_sec: float) -> list:
    """variant: "new" (現状コードそのまま) または "legacy" (修正前を再現)。

    end_sec: 処理コスト削減のため、比較に必要な区間だけ短縮できるように
    呼び出し側から指定する (実行のたびに再生成、本体コードは変更しない)。
    """
    log: list = []
    _install_logging_patch(log)
    if variant == "legacy":
        ov.ResolvedExchangeTracker._amplify_decisive = _legacy_amplify_decisive
    history: list = []
    ov.generate(
        VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=end_sec,
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
    )
    for row in log:
        row["history"] = list(history)
    return log


def _in_window(t: "float | None") -> "str | None":
    if t is None:
        return None
    for name, (lo, hi) in WINDOWS.items():
        if lo <= t <= hi:
            return name
    return None


def _print_log(label: str, log: list) -> None:
    print(f"\n===== {label} =====")
    for row in log:
        win = _in_window(row["t"])
        tag = f" [{win}]" if win else ""
        print(f"t={row['t']:.2f}{tag} score1={row['score1']} score2={row['score2']} "
              f"ev1_cc={row['ev1_cc']} ev2_cc={row['ev2_cc']} "
              f"hold_adv={row['hold_adv']:+.2f} hold_p1={row['hold_p1']*100:.2f}% "
              f"defender={row['defender_side']} incoming={row['incoming']:.1f} "
              f"defender_prob={row['defender_prob']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["legacy", "new"], required=True)
    parser.add_argument("--end-sec", type=float, default=END_SEC,
                         help="処理コスト削減用。比較対象窓を含む最小限まで短縮可能")
    args = parser.parse_args()

    _tag_decisive_counter_tracker()  # ResolvedExchangeTracker 構築より前に (計測対象のMCと区別するため)
    _patch_t_tracker()
    _patch_hcache_stub()
    _patch_live_counter_tracker_stub()
    label = "旧実装 (修正前を再現)" if args.variant == "legacy" else "新実装 (現状コード、修正1+2+3適用済み)"
    log = _run(args.variant, args.end_sec)
    _print_log(label, log)

    print("\n===== 対象窓のみの抜粋 =====")
    for name in WINDOWS:
        print(f"\n-- {name} --")
        for row in log:
            if _in_window(row["t"]) == name:
                print(f"    t={row['t']:.2f} hold_adv={row['hold_adv']:+.2f} "
                      f"hold_p1(1P)={row['hold_p1']*100:.2f}% "
                      f"hold_p1(2P)={100-row['hold_p1']*100:.2f}% "
                      f"defender={row['defender_side']} defender_prob={row['defender_prob']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
