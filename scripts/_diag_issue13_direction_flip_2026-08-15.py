"""指摘13 (enable_resolved_live_defender) の「方向反転」解剖用計装スクリプト。

t=234.90 (指摘12窓) を中心に、ResolvedExchangeTracker の
_resolve() (決着=凍結値、旧経路) と _reevaluate_live_defender() (ライブ再評価、
新経路=指摘13) それぞれが _score_advantage_full_row に渡す 47列特徴量を
ダンプし、adv 差分の寄与を特徴ごとに分解する。

本体コード (scripts/visualize_advantage_overlay.py, src/*) は一切変更せず、
モジュールレベル関数・クラスメソッドを実行時にラップして計装するのみ
(デバッガ役の規律: 修正は行わない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue13_flip.mp4")

# [注意] start_sec は必ず公式再現スクリプト (_diag_issue13_fix_verify_2026-08-15.py)
# と同一の 162.0s にする。認識パイプラインは状態を持つ (online HSV較正・
# baseline reset・お邪魔累積会計・ChainEvent履歴等) ため、t=234.90 に近い
# 場所から seek すると累積状態が変わり別のシーン (incoming=37等) を
# 誤って再現してしまう (実際に230.0s開始で試して確認済み、本ファイル末尾の
# 教訓コメント参照)。stride/fps同様、start_sec もフラグ構成の一部として
# 本番条件と厳密一致させる。
START_SEC = 162.0
END_SEC = 245.0
WINDOW_LO, WINDOW_HI = 234.5, 240.0

captured: list[dict] = []

# ---- 計装1: _score_advantage_full_row の呼び出しごとに f1/f2/cols/結果を捕捉 ----
_orig_full_row = ov._score_advantage_full_row


def _instrumented_full_row(model, b1, b2, snap, attribution_exclude):
    cols = model._puyo_full_cols
    base1, base2 = ov._side_feats_full_base(b1), ov._side_feats_full_base(b2)
    f1 = ov._side_feats_full(base1, base2, snap.net_balance_capped, snap.forecast_p1)
    f2 = ov._side_feats_full(base2, base1, -snap.net_balance_capped, snap.forecast_p2)
    adv, p1, drivers = _orig_full_row(model, b1, b2, snap, attribution_exclude)
    captured.append({
        "kind": "model_call",
        "f1": dict(f1), "f2": dict(f2), "cols": list(cols),
        "adv": adv, "p1": p1,
        "forecast_p1_raw": snap.forecast_p1, "forecast_p2_raw": snap.forecast_p2,
        "net_balance_capped": snap.net_balance_capped,
    })
    return adv, p1, drivers


ov._score_advantage_full_row = _instrumented_full_row

# ---- 計装2: _resolve() (凍結決着) と _reevaluate_live_defender() (ライブ) の
#      前後で t_sec・amp・defender_prob 等を記録 ----
_orig_resolve = ov.ResolvedExchangeTracker._resolve
_orig_reeval = ov.ResolvedExchangeTracker._reevaluate_live_defender
_orig_amplify = ov.ResolvedExchangeTracker._amplify_decisive


def _wrapped_resolve(self, snap, elapsed_sec, score1, score2):
    idx_before = len(captured)
    _orig_resolve(self, snap, elapsed_sec, score1, score2)
    for c in captured[idx_before:]:
        c["event"] = "_resolve"
        c["t_sec"] = self._t_sec
        c["defender_side"] = self.hold_defender_side
        c["incoming_ojama"] = self.hold_incoming_ojama
        c["defender_prob"] = self.hold_defender_prob
        c["hold_adv_final"] = self.hold_adv
        c["hold_p1_final"] = self.hold_p1


def _wrapped_reeval(self, b1, b2, snap=None):
    # [2026-08-15 追記] 方向反転修正で _reevaluate_live_defender に snap
    # 引数が追加されたため、本計装スクリプトのラッパーも追従する
    # (本体コードではなく計装用ラッパーの更新、デバッガ役の規律は維持)。
    idx_before = len(captured)
    _orig_reeval(self, b1, b2, snap)
    for c in captured[idx_before:]:
        c["event"] = "_reevaluate_live_defender"
        c["t_sec"] = self._t_sec
        c["defender_side"] = self.hold_defender_side
        c["incoming_ojama"] = self.hold_incoming_ojama
        c["defender_prob"] = self.hold_defender_prob
        c["hold_adv_final"] = self.hold_adv
        c["hold_p1_final"] = self.hold_p1


ov.ResolvedExchangeTracker._resolve = _wrapped_resolve
ov.ResolvedExchangeTracker._reevaluate_live_defender = _wrapped_reeval


def main() -> int:
    history: list = []
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

    print(f"\n捕捉件数: {len(captured)}")
    in_window = [c for c in captured if WINDOW_LO <= c.get("t_sec", -1) <= WINDOW_HI]
    print(f"窓内 ({WINDOW_LO}-{WINDOW_HI}s) 件数: {len(in_window)}")

    # _resolve (凍結決着、旧経路の基準点) を探す — 直近の _resolve イベント
    resolve_events = [c for c in captured if c.get("event") == "_resolve"]
    reeval_events = [c for c in in_window if c.get("event") == "_reevaluate_live_defender"]

    if not resolve_events:
        print("_resolve イベントが窓内に見つからず")
        return 1
    frozen = resolve_events[-1] if resolve_events[-1]["t_sec"] <= WINDOW_HI else (
        resolve_events[0])
    # 直近の _resolve (窓開始前の決着セッション) を使う
    prior_resolves = [c for c in resolve_events if c["t_sec"] <= WINDOW_LO]
    frozen = prior_resolves[-1] if prior_resolves else resolve_events[0]

    print("\n===== 凍結 (_resolve時点) =====")
    print(f"t_sec={frozen['t_sec']:.2f} defender_side={frozen['defender_side']} "
          f"incoming={frozen['incoming_ojama']:.1f} defender_prob={frozen['defender_prob']}")
    print(f"model adv(pre-amplify推定不可、amplify後の値のみ記録)="
          f"hold_adv_final={frozen['hold_adv_final']:.2f} hold_p1_final={frozen['hold_p1_final']*100:.1f}%")
    print(f"model単体 adv (このcallでの_score_advantage_full_row戻り値)="
          f"{frozen['adv']:.2f} p1={frozen['p1']*100:.1f}%")
    print(f"forecast_p1_raw={frozen['forecast_p1_raw']:.1f} "
          f"forecast_p2_raw={frozen['forecast_p2_raw']:.1f} "
          f"net_balance_capped={frozen['net_balance_capped']:.1f}")

    if not reeval_events:
        print("\n窓内に _reevaluate_live_defender イベントなし")
        return 1
    live = reeval_events[0]
    print("\n===== ライブ再評価 (最初の _reevaluate_live_defender) =====")
    print(f"t_sec={live['t_sec']:.2f} defender_side={live['defender_side']} "
          f"incoming={live['incoming_ojama']:.1f} defender_prob={live['defender_prob']}")
    print(f"model単体 adv={live['adv']:.2f} p1={live['p1']*100:.1f}%")
    print(f"hold_adv_final(amplify後)={live['hold_adv_final']:.2f} "
          f"hold_p1_final={live['hold_p1_final']*100:.1f}%")
    print(f"amp寄与(推定) = hold_adv_final - model単体adv = "
          f"{live['hold_adv_final'] - live['adv']:.2f}")
    print(f"forecast_p1_raw={live['forecast_p1_raw']:.1f} "
          f"forecast_p2_raw={live['forecast_p2_raw']:.1f} "
          f"net_balance_capped={live['net_balance_capped']:.1f}")

    # ---- 47列 特徴量の差分 (frozen vs live)、1P視点 (f1-f2) の値で比較 ----
    cols = live["cols"]
    print("\n===== 47列 特徴量 (1P視点 f1-f2) 凍結 vs ライブ、差分降順 =====")
    rows = []
    for c in cols:
        fv_frozen = frozen["f1"].get(c, 0.0) - frozen["f2"].get(c, 0.0)
        fv_live = live["f1"].get(c, 0.0) - live["f2"].get(c, 0.0)
        rows.append((c, fv_frozen, fv_live, fv_live - fv_frozen))
    rows.sort(key=lambda r: -abs(r[3]))
    print(f"{'feature':32s} {'frozen(f1-f2)':>14s} {'live(f1-f2)':>14s} {'delta':>10s}")
    for name, fz, lv, d in rows[:20]:
        print(f"{name:32s} {fz:14.4f} {lv:14.4f} {d:10.4f}")

    # ---- 個別 raw forecast / board_ojama_count 直読み (自前計算) ----
    print("\n===== raw値直読み (forecast / board_ojama_count) =====")
    print(f"frozen: forecast_p1_raw={frozen['forecast_p1_raw']:.1f} "
          f"(score cap後={ov.iv.ojama_forecast(frozen['forecast_p1_raw']).score:.3f}) / "
          f"forecast_p2_raw={frozen['forecast_p2_raw']:.1f} "
          f"(score cap後={ov.iv.ojama_forecast(frozen['forecast_p2_raw']).score:.3f})")
    print(f"live:   forecast_p1_raw={live['forecast_p1_raw']:.1f} "
          f"(score cap後={ov.iv.ojama_forecast(live['forecast_p1_raw']).score:.3f}) / "
          f"forecast_p2_raw={live['forecast_p2_raw']:.1f} "
          f"(score cap後={ov.iv.ojama_forecast(live['forecast_p2_raw']).score:.3f})")
    print(f"ON_FIELD_CAP={ov.iv.ON_FIELD_CAP}")

    print("\n===== board_ojama_count (score, own値) =====")
    print(f"frozen: f1={frozen['f1'].get('board_ojama_count'):.4f} "
          f"f2={frozen['f2'].get('board_ojama_count'):.4f}")
    print(f"live:   f1={live['f1'].get('board_ojama_count'):.4f} "
          f"f2={live['f2'].get('board_ojama_count'):.4f}")

    print("\n===== diff_current_max_chain / diff_max_column_height (own値) =====")
    for key in ("diff_board_ojama_count", "diff_current_max_chain", "diff_max_column_height"):
        print(f"{key}: frozen f1={frozen['f1'].get(key):.4f} live f1={live['f1'].get(key):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
