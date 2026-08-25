"""指摘12 診断 (2/2): 2試合目で2Pの本線発火時に判定が97%超となる瞬間について、
4成分 (予告おじゃま/学習モデル/脅威/応手) + 決着ホールド/決定度増幅の内訳を
フレーム単位で計装する。

本体コード (scripts/visualize_advantage_overlay.py) は変更せず、対象クラス・
関数をモンキーパッチしてログを取る (計装ラッパー方式)。

demo_fixed_3match.mp4 と完全同一の生成条件 (scripts/_gen_demo_fixed_2026-08-13.sh)
で generate(render=False) を実行する。

TARGET_LO/TARGET_HI: scripts/_diag_issue12_disp_2026-08-14.py の結果で特定した
2P本線発火の対象窓 (source絶対秒)。generate() 自体は状態連続性のため
start_sec=162 (デモ先頭=試合1開始) から実行し、ログ出力のみ対象窓で絞る。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
iv = ov.iv  # ov 内部と同一モジュールインスタンスを使う (src.indicators_v2)

START_SEC = 162.0
# 対象イベント (実画面フレーム t72.9.png で確定: source=234.9s で
# disp_adv=-72.75 (2P 97%) が発火・ホールド開始) の前後のみをカバーする
# (計算コスト削減。_resolve()/_amplify_decisive() は発火の瞬間に1回だけ
# 走るので、トリガー直後までで十分)。
END_SEC = 237.0
TARGET_LO = 220.0
TARGET_HI = 237.0

# ---- ログバッファ ----
LOG_HCACHE: list[dict] = []
LOG_FORECAST: list[dict] = []
LOG_DEFENDER_THREAT: list[dict] = []
LOG_COUNTER_UPDATE: list[dict] = []
LOG_RESOLVED_UPDATE: list[dict] = []
LOG_RESOLVED_RESOLVE: list[dict] = []
LOG_AMPLIFY: list[dict] = []

_CUR_T = {"t": None}  # generate() メインループの絶対時刻 t を共有する小道具


def _patch_hcache() -> None:
    """HeavyAdvCache.update をスタブ化 (計算コスト削減、2026-08-14)。

    ライブ per-frame ブレンド (W_MODEL*model_adv + W_THREAT*threat) 用の値
    だが、決着ホールド中はこの経路自体が丸ごとスキップされる
    (resolved_hold_freezes_settled) ため、今回の対象イベント (ホールド内部の
    _resolve/_amplify_decisive) の値には一切影響しない。saturated_chain_count/
    ukeyasusa/reach_fire_power (threat) はいずれも連鎖シミュレーション込みで
    重いため、ここをゼロ固定にして高速化する。"""

    def patched(self, b1, b2, snap, sp1, sp2, elapsed):
        result = (0.0, 0.0, [], 0.0, 0.0, 0.0, 0.0)
        LOG_HCACHE.append({"t": _CUR_T["t"], "elapsed": elapsed, "stubbed": True})
        return result

    ov.HeavyAdvCache.update = patched


def _patch_forecast() -> None:
    orig = ov.RealtimeForecastTracker.update

    def patched(self, score1, score2, tsumo1, tsumo2, rate=70.0):
        result = orig(self, score1, score2, tsumo1, tsumo2, rate)
        LOG_FORECAST.append({
            "t": _CUR_T["t"], "score1": score1, "score2": score2,
            "tsumo1": tsumo1, "tsumo2": tsumo2,
            "inc1": self.inc1, "inc2": self.inc2, "fc": result,
        })
        return result

    ov.RealtimeForecastTracker.update = patched


def _patch_defender_threat() -> None:
    orig = ov._resolve_defender_threat

    def patched(obs, snap, elapsed_sec):
        result = orig(obs, snap, elapsed_sec)
        defender_side, incoming = result
        LOG_DEFENDER_THREAT.append({
            "t": _CUR_T["t"], "chain_count": obs.chain_count,
            "trigger_sec": obs.trigger_sec, "attacker_side": obs.attacker_side,
            "attacker_total_score": (
                float(obs.attacker_event.total_score) if obs.attacker_event else None
            ),
            "pending_p1": snap.pending_p1, "pending_p2": snap.pending_p2,
            "elapsed_sec": elapsed_sec,
            "defender_side": defender_side, "incoming_ojama": incoming,
        })
        return result

    ov._resolve_defender_threat = patched


def _patch_counter_tracker() -> None:
    """ライブ per-frame 用 (main loop の counter_tracker インスタンス) の MC
    計算だけをスタブ化する (計算コスト削減、2026-08-14)。

    ResolvedExchangeTracker._counter_tracker (決着ホールド内部・#10増幅専用の
    別インスタンス) は `_patch_resolved_tracker._tag_decisive_counter_tracker`
    で `_is_decisive_amplify_tracker=True` を付与済み。本パッチはそのタグが
    無いインスタンス (=ライブ per-frame 表示専用) のみスタブ化するため、
    決着ホールドの#10増幅の値には一切影響しない (本物のMC計算のまま)。
    """
    orig = ov.CounterReachTracker._update_defender_only

    def patched(self, b1, b2, budget_sec, next1, next2, t_sec, defender_side, threshold_ojama):
        if not getattr(self, "_is_decisive_amplify_tracker", False):
            # ライブ per-frame 表示専用: パネルの応手%表示にのみ使われる
            # (決着ホールド中はこの表示自体が凍結される)。値は使わないので
            # 固定スタブで良い。
            result = (0.0, float("nan"), float("nan"))
            self.last_hands = 0.0
            return result
        result = orig(self, b1, b2, budget_sec, next1, next2, t_sec, defender_side, threshold_ojama)
        LOG_COUNTER_UPDATE.append({
            "t": _CUR_T["t"], "t_sec_arg": t_sec, "budget_sec": budget_sec,
            "defender_side": defender_side, "threshold_ojama": threshold_ojama,
            "result": result, "last_hands": self.last_hands,
        })
        return result

    ov.CounterReachTracker._update_defender_only = patched


def _tag_decisive_counter_tracker() -> None:
    """ResolvedExchangeTracker.__init__ 直後に内部 counter_tracker へ目印を付ける。"""
    orig_init = ov.ResolvedExchangeTracker.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._counter_tracker._is_decisive_amplify_tracker = True

    ov.ResolvedExchangeTracker.__init__ = patched_init


def _patch_counter_defender_adv() -> None:
    orig = ov._counter_defender_adv

    def patched(defender_side, defender_prob, incoming_ojama, b1, b2):
        amp = orig(defender_side, defender_prob, incoming_ojama, b1, b2)
        defender_board = b1 if defender_side == "1P" else b2
        try:
            damage = iv.ojama_damage(defender_board, incoming_ojama).score
        except Exception:
            damage = None
        LOG_AMPLIFY.append({
            "t": _CUR_T["t"], "kind": "counter_defender_adv",
            "defender_side": defender_side, "defender_prob": defender_prob,
            "incoming_ojama": incoming_ojama, "damage": damage, "amp": amp,
        })
        return amp

    ov._counter_defender_adv = patched


def _patch_resolved_tracker() -> None:
    orig_update = ov.ResolvedExchangeTracker.update
    orig_resolve = ov.ResolvedExchangeTracker._resolve
    orig_amplify = ov.ResolvedExchangeTracker._amplify_decisive

    def patched_update(self, r_p1, r_p2, snap, elapsed_sec):
        active, deact = orig_update(self, r_p1, r_p2, snap, elapsed_sec)
        LOG_RESOLVED_UPDATE.append({
            "t": _CUR_T["t"], "elapsed_sec": elapsed_sec,
            "ev1_present": r_p1.chain_event is not None,
            "ev2_present": r_p2.chain_event is not None,
            "ev1_score": (
                float(r_p1.chain_event.total_score) if r_p1.chain_event else None
            ),
            "ev2_score": (
                float(r_p2.chain_event.total_score) if r_p2.chain_event else None
            ),
            "active": active, "just_deactivated": deact,
            "hold_adv": self.hold_adv, "hold_p1": self.hold_p1,
        })
        return active, deact

    def patched_resolve(self, snap, elapsed_sec, score1, score2):
        ev1_cc = self._ev1.chain_count if self._ev1 else None
        ev2_cc = self._ev2.chain_count if self._ev2 else None
        orig_resolve(self, snap, elapsed_sec, score1, score2)
        LOG_RESOLVED_RESOLVE.append({
            "t": _CUR_T["t"], "elapsed_sec": elapsed_sec,
            "score1": score1, "score2": score2,
            "ev1_chain_count": ev1_cc, "ev2_chain_count": ev2_cc,
            "hold_adv": self.hold_adv, "hold_p1": self.hold_p1,
            "hold_drivers": list(self.hold_drivers),
        })

    def patched_amplify(self, adv, result):
        defender_side, incoming = self._decisive_defender(result)
        adv_before = adv
        new_adv, new_p1 = orig_amplify(self, adv, result)
        LOG_AMPLIFY.append({
            "t": _CUR_T["t"], "kind": "decisive_amplify",
            "adv_before": adv_before, "adv_after": new_adv, "p1_after": new_p1,
            "defender_side": defender_side, "incoming": incoming,
        })
        return new_adv, new_p1

    ov.ResolvedExchangeTracker.update = patched_update
    ov.ResolvedExchangeTracker._resolve = patched_resolve
    ov.ResolvedExchangeTracker._amplify_decisive = patched_amplify


def _patch_t_tracker() -> None:
    """generate() 内の絶対時刻 t を共有するため、pipe.update 呼び出し
    (毎フレーム t を伴って呼ばれる) をフックする。"""
    from src.recognition_pipeline import RecognitionPipeline
    orig = RecognitionPipeline.update

    def patched(self, fi, t, frame):
        _CUR_T["t"] = t
        return orig(self, fi, t, frame)

    RecognitionPipeline.update = patched


def apply_all_patches() -> None:
    _tag_decisive_counter_tracker()  # _patch_counter_tracker より先に (継承順は無関係だが明示)
    _patch_t_tracker()
    _patch_hcache()
    _patch_forecast()
    _patch_defender_threat()
    _patch_counter_tracker()
    _patch_counter_defender_adv()
    _patch_resolved_tracker()


def _in_window(t: "float | None") -> bool:
    return t is not None and TARGET_LO <= t <= TARGET_HI


def dump_all() -> None:
    print("\n===== HeavyAdvCache (計算コスト削減のためスタブ化・値は無視してよい) =====")
    print(f"  (呼び出し回数: {sum(1 for e in LOG_HCACHE if _in_window(e['t']))} 件、"
          f"決着ホールドの値には影響なし)")
    print("\n===== RealtimeForecastTracker (予告おじゃま) =====")
    for e in LOG_FORECAST:
        if not _in_window(e["t"]):
            continue
        print(f"t={e['t']:.2f} score1={e['score1']} score2={e['score2']} "
              f"inc1={e['inc1']:.1f} inc2={e['inc2']:.1f} fc={e['fc']:+.2f}")
    print("\n===== _resolve_defender_threat (脅威側/飛来量) =====")
    for e in LOG_DEFENDER_THREAT:
        if not _in_window(e["t"]):
            continue
        print(f"t={e['t']:.2f} chain_count={e['chain_count']} attacker={e['attacker_side']} "
              f"attacker_score={e['attacker_total_score']} pending_p1={e['pending_p1']} "
              f"pending_p2={e['pending_p2']} elapsed={e['elapsed_sec']:.2f} "
              f"-> defender={e['defender_side']} incoming={e['incoming_ojama']:.1f}")
    print("\n===== CounterReachTracker._update_defender_only (応手確率) =====")
    for e in LOG_COUNTER_UPDATE:
        if not _in_window(e["t"]):
            continue
        print(f"t={e['t']:.2f} defender={e['defender_side']} budget_sec={e['budget_sec']:.2f} "
              f"threshold_ojama={e['threshold_ojama']} result(adv,p1,p2)={e['result']} "
              f"mean_hands={e['last_hands']:.2f}")
    print("\n===== ResolvedExchangeTracker.update (決着ホールド有効/無効) =====")
    for e in LOG_RESOLVED_UPDATE:
        if not _in_window(e["t"]):
            continue
        print(f"t={e['t']:.2f} ev1_present={e['ev1_present']}(score={e['ev1_score']}) "
              f"ev2_present={e['ev2_present']}(score={e['ev2_score']}) "
              f"active={e['active']} just_deactivated={e['just_deactivated']} "
              f"hold_adv={e['hold_adv']:+.2f} hold_p1={e['hold_p1']:.3f}")
    print("\n===== ResolvedExchangeTracker._resolve (決着計算発火) =====")
    for e in LOG_RESOLVED_RESOLVE:
        if not _in_window(e["t"]):
            continue
        print(f"t={e['t']:.2f} score1={e['score1']} score2={e['score2']} "
              f"ev1_chain={e['ev1_chain_count']} ev2_chain={e['ev2_chain_count']} "
              f"hold_adv={e['hold_adv']:+.2f} hold_p1={e['hold_p1']:.3f} "
              f"drivers={e['hold_drivers']}")
    print("\n===== 増幅 (decisive_amplify / counter_defender_adv) =====")
    for e in LOG_AMPLIFY:
        if not _in_window(e["t"]):
            continue
        print(f"t={e['t']:.2f} {e}")


def main() -> int:
    apply_all_patches()
    video = Path("data/frames/review_demo_2026-08-12.mp4")
    out = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue12_breakdown.mp4")
    history: list[tuple[float, float]] = []
    ov.generate(
        video, out, max_sec=0.0, sample_interval=0.0,
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
    )
    dump_all()
    print("\n===== disp_adv (表示値、フル解像度・間引きなし) =====")
    import numpy as np
    ts, advs = [], []
    for t_sec, disp_adv in history:
        if not _in_window(t_sec):
            continue
        p1 = 0.5 + disp_adv / 200.0
        print(f"t={t_sec:.3f} disp_adv={disp_adv:+.2f} p1={p1*100:.2f}%")
        ts.append(t_sec)
        advs.append(disp_adv)
    npz_path = Path("data/verify/demo_fixed_2026-08-13/_diag_issue12_history.npz")
    np.savez_compressed(str(npz_path), t=np.array(ts), disp_adv=np.array(advs))
    print(f"\n[saved] {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
