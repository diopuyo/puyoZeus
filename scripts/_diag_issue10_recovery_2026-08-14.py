"""指摘#10 (docs/DEMO_REVIEW_2026-08-13.md #10、2026-08-14デモ1 v3レビュー) の
量的裏取り: 「量的に返せないことはほぼ確定している」を数値化する補助スクリプト。

_diag_issue10_mutual_exchange_2026-08-14.py が出力した JSON
(logs/_diag_issue10_mutual_exchange_2026-08-14.json) から実際の場面の
盤面グリッドを読み込み、以下を計算する (本体コードは一切変更しない、
既存の src.indicators_v2 関数をそのまま呼ぶだけ):

  (a) 着弾側 (受け側) の吸収余力: death_margin_raw (窒息列の残り段数) と
      ojama_damage (発火点埋没モデル、reference_ojama_damage_function/
      reference_ojama_damage_nonlinear の実装) を実際の着弾個数で評価。
  (b) セカンド (返し) を組むのに必要な時間 vs 実際に得られる時間:
      counter_reach_probability (K=1..4 到達確率、既存 XVII 打ち合い応手
      確率と同一関数) を着弾後盤面に対して評価し、時間予算
      estimate_chain_anim_duration_sec を併記する。
  (c) 上記から「71%(1P有利)/29%」という決着値のうち、実際に何が
      効いていて何が効いていないか (_score_advantage の diff 内訳と
      突き合わせ) を要約する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.indicators_v2 as iv  # noqa: E402
from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402

LOG_JSON = Path("logs/_diag_issue10_mutual_exchange_2026-08-14.json")


def _board_from_grid(grid: list) -> Board:
    b = Board()
    b._grid = __import__("numpy").array(grid, dtype=int)
    return b


def main() -> None:
    data = json.loads(LOG_JSON.read_text(encoding="utf-8"))
    exch = data["exchange_scene"]
    adv = data["score_adv_scene"]
    print(f"[loaded] exchange records={len(exch)}, score_adv records={len(adv)}")
    if not exch:
        print("[warn] シーン範囲(190-205s)内に resolve_mutual_exchange 呼び出しが"
              "無い。exchange_all/score_adv_all の範囲を確認する。")
        exch_all = data["exchange_all"]
        adv_all = data["score_adv_all"]
        print(f"[fallback] all records: exchange={len(exch_all)}, score_adv={len(adv_all)}")
        exch = exch_all
        adv = adv_all

    sim = ChainSimulator()
    for i, rec in enumerate(exch):
        print(f"\n===== resolve_mutual_exchange call #{i} (t={rec['t']}) =====")
        print(f"  gen_p1_ojama={rec['gen_p1_ojama']} gen_p2_ojama={rec['gen_p2_ojama']}"
              f"  pending_p1={rec['pending_p1']} pending_p2={rec['pending_p2']}")
        print(f"  chain_count_p1={rec['chain_count_p1']} chain_count_p2={rec['chain_count_p2']}")
        print(f"  dropped_to_p1={rec['dropped_to_p1']} dropped_to_p2={rec['dropped_to_p2']}"
              f"  leftover_p1={rec['leftover_p1']} leftover_p2={rec['leftover_p2']}")
        print(f"  p1_dead={rec['p1_dead']} p2_dead={rec['p2_dead']}")

        before_p1 = _board_from_grid(rec["before_p1"]["grid"])
        before_p2 = _board_from_grid(rec["before_p2"]["grid"])
        after_p1 = _board_from_grid(rec["board_p1_after"]["grid"])
        after_p2 = _board_from_grid(rec["board_p2_after"]["grid"])
        print(f"  before_p1 heights={rec['before_p1']['heights']}"
              f" death_margin_raw={rec['before_p1']['death_margin_raw']}")
        print(f"  before_p2 heights={rec['before_p2']['heights']}"
              f" death_margin_raw={rec['before_p2']['death_margin_raw']}")
        print(f"  after_p1  heights={rec['board_p1_after']['heights']}"
              f" death_margin_raw={rec['board_p1_after']['death_margin_raw']}")
        print(f"  after_p2  heights={rec['board_p2_after']['heights']}"
              f" death_margin_raw={rec['board_p2_after']['death_margin_raw']}")

        # (a) 受け側の吸収余力: 実際に着弾させた量 (dropped_to_*) を
        # 「着弾前盤面」に対して ojama_damage で評価する (二重計上回避のため
        # after ではなく before に対して評価する、ojama_damage の docstring 通り)。
        if rec["dropped_to_p2"] > 0:
            dmg2 = iv.ojama_damage(before_p2, rec["dropped_to_p2"], simulator=sim)
            print(f"  [a] 2P受け: 着弾{rec['dropped_to_p2']}個 に対する ojama_damage="
                  f"{dmg2.score:.3f} (raw余裕段数={dmg2.raw:.2f}, "
                  f"0.05=無害/0.5=かなり不利/1.0=ほぼ死)")
        if rec["dropped_to_p1"] > 0:
            dmg1 = iv.ojama_damage(before_p1, rec["dropped_to_p1"], simulator=sim)
            print(f"  [a] 1P受け: 着弾{rec['dropped_to_p1']}個 に対する ojama_damage="
                  f"{dmg1.score:.3f} (raw余裕段数={dmg1.raw:.2f})")

        # (b) 着弾後盤面から「返しに必要な連鎖を組めるか」を counter_reach_probability
        # で評価する (受け側 = より多く着弾を受けた側)。閾値は「相手に送り返す
        # のに必要なおじゃま換算量」= 直近の net 差 (概算として dropped_to_攻撃側 の
        # 逆数、無ければ攻撃側が受けた量を到達目標とみなす)。
        defender_board, defender_label, threshold = (
            (after_p2, "2P", float(max(1, rec["dropped_to_p2"])))
            if rec["dropped_to_p2"] >= rec["dropped_to_p1"]
            else (after_p1, "1P", float(max(1, rec["dropped_to_p1"])))
        )
        if not defender_board.is_dead():
            cr = iv.counter_reach_probability(defender_board, threshold, elapsed_sec=35.0)
            print(f"  [b] {defender_label}応手 (着弾後盤面から閾値{threshold:.0f}個到達確率):"
                  f" {cr.probabilities}")
        else:
            print(f"  [b] {defender_label} は着弾後 is_dead=True (応手評価不可、"
                  f"窒息済み)")
        anim1 = iv.estimate_chain_anim_duration_sec(float(rec["chain_count_p1"]))
        anim2 = iv.estimate_chain_anim_duration_sec(float(rec["chain_count_p2"]))
        print(f"  [b] 連鎖アニメ推定所要: 1P={anim1:.2f}s (chain_count="
              f"{rec['chain_count_p1']}), 2P={anim2:.2f}s (chain_count={rec['chain_count_p2']})")

    print("\n===== _score_advantage (71%の内訳分解) =====")
    for i, rec in enumerate(adv):
        print(f"\n--- call #{i} (t={rec['t']}) adv={rec['adv']:.2f} p1={rec['p1']:.4f} ---")
        print(f"  net_balance_capped={rec['net_balance_capped']}"
              f" forecast_p1={rec['forecast_p1']} forecast_p2={rec['forecast_p2']}")
        print("  drivers(|diff|上位3、attribution_exclude適用後):")
        for name, val in rec["drivers"]:
            print(f"    {name}: diff(1P-2P)={val:+.4f}")
        print("  全diff内訳:")
        for name in rec["cols"]:
            if name in rec["diff"]:
                print(f"    {name}: {rec['diff'][name]:+.4f}")


if __name__ == "__main__":
    main()
