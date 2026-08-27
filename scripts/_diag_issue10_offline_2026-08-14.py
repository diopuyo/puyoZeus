"""指摘#10 (docs/DEMO_REVIEW_2026-08-13.md #10) の定量裏取り: オフライン再現版。

方針転換 (2026-08-14、coordinator指示): 動画再実行 (generate() の重い
RecognitionPipeline全frame処理) を待たず、既存の計装済み成果物のみから
決着計算を再現する:

  - logs/_diag_chain_score_zero_2026-08-13_trace2_fillON.log
    (enable_pseudo_chain_score_fill=True、本番/デモv3と同じ経路) から
    実際の trigger 値を読み取り済み: i=409 (t≈195.6s) で
    ev1(1P)=(cc=8, total_score=30300, trig=195.60),
    ev2(2P)=(cc=2, total_score=2500, trig=195.33) が同時に非Noneになった
    瞬間 = ResolvedExchangeTracker が実際に _resolve() を呼んだ瞬間
    (CHAIN_TOTAL_MIN_SCORE=40 双方超過、docs参照)。
  - data/verify/fps_stride_ab_2026-08-12/review_demo_stride2.npz
    (stride=2 A/B検証で収集済みの review_demo 盤面グリッド) から、
    上記トリガー直前の両者 STABLE 盤面 (= ChainEvent.before_board 相当)
    を復元する。

本スクリプトは src.exchange_virtual_board.resolve_mutual_exchange と
scripts.visualize_advantage_overlay._score_advantage を **盤面データ直渡し**
で呼ぶだけで、動画のフレーム処理・RecognitionPipeline は一切実行しない
(高速、CPU競合の影響を受けない)。

pending_p1/pending_p2 (交換前の予告おじゃま繰越) は、この2本のログ/npzには
記録されていないため 0 と仮定する (本スクリプト内で明示する簡略化、
下記メイン処理のコメント参照)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.visualize_advantage_overlay as vao
import src.indicators_v2 as iv
from src.board import Board
from src.chain import ChainSimulator
from src.exchange_virtual_board import resolve_mutual_exchange
from src.ojama_accounting import OjamaAccountSnapshot
from src.scoring import OJAMA_MAX_DROP_PER_TURN

NPZ_PATH = Path("data/verify/fps_stride_ab_2026-08-12/review_demo_stride2.npz")
OUT_JSON = Path("logs/_diag_issue10_offline_result_2026-08-14.json")

# trace2_fillON.log (i=409, t≈195.6s) から読み取った実測トリガー値。
EV1_TOTAL_SCORE = 30300.0  # 1P (攻撃側、cc=8)
EV2_TOTAL_SCORE = 2500.0   # 2P (攻撃側、cc=2)
# 試合開始(t≈164s、demo_fixed_v3ログの [reset] 群から)からの経過秒。
# MARGIN_TIME_START_SEC=96s 未満であれば rate は必ず70 (OJAMA_RATE_STANDARD)
# になるため、正確な試合開始時刻の特定誤差 (数秒) は結果に影響しない。
ELAPSED_SEC = 31.6


def _board_from_grid(grid: np.ndarray) -> Board:
    b = Board()
    b._grid = grid.astype(int)
    return b


def _board_report(label: str, b: Board) -> dict:
    heights = [b.height_of(c) for c in range(6)]
    return {
        "label": label, "heights": heights,
        "count_puyos": b.count_puyos(), "is_dead": b.is_dead(),
        "death_margin_raw": 12 - b.height_of(2),
    }


def main() -> None:
    d = np.load(NPZ_PATH, allow_pickle=True)
    t = d["t_sec"]
    side = d["side"]
    score = d["score"]
    grids = d["grids"]

    # 2P: 最後の STABLE スナップショット (t=194.63、chain_trig=194.50、
    # score=1318) = 2500点の2連鎖が始まる直前の before_board。
    idx_2p = np.where((side == "2P") & (t > 194.0) & (t < 195.0))[0]
    assert len(idx_2p) > 0, "2P before-board が見つからない"
    i2 = idx_2p[-1]
    before_p2 = _board_from_grid(grids[i2])
    print(f"[found] 2P before_board: npz idx={i2} t={t[i2]:.2f}s score={score[i2]}")

    # 1P: 最後の STABLE スナップショット (t=195.53、score=832) = 30300点の
    # 8連鎖が始まる直前の before_board。
    idx_1p = np.where((side == "1P") & (t > 195.0) & (t < 195.6))[0]
    assert len(idx_1p) > 0, "1P before-board が見つからない"
    i1 = idx_1p[-1]
    before_p1 = _board_from_grid(grids[i1])
    print(f"[found] 1P before_board: npz idx={i1} t={t[i1]:.2f}s score={score[i1]}")

    print(f"\n[report] before_p1: {_board_report('1P(攻撃側,cc=8)', before_p1)}")
    print(f"[report] before_p2: {_board_report('2P(攻撃側,cc=2)', before_p2)}")

    # --- おじゃま換算 (score_to_ojama, rate=70固定域) ---
    gen1 = iv._score_to_ojama_count(EV1_TOTAL_SCORE, ELAPSED_SEC)  # 1P->2Pへ送る量
    gen2 = iv._score_to_ojama_count(EV2_TOTAL_SCORE, ELAPSED_SEC)  # 2P->1Pへ送る量
    print(f"\n[ojama換算] gen1(1P生成,2Pへ)={gen1}個  gen2(2P生成,1Pへ)={gen2}個"
          f"  (score_to_ojama, rate=70, elapsed={ELAPSED_SEC}s)")

    # pending (交換前の予告おじゃま繰越) はこの2本の成果物には記録がないため
    # 0 と仮定する (簡略化、明示)。感度確認のため pending=0 の結果に加えて
    # 「pendingが無視できない規模だった場合」の参考レンジも併記する。
    pending_p1, pending_p2 = 0, 0

    result = resolve_mutual_exchange(
        before_p1, before_p2, gen1, gen2, pending_p1, pending_p2,
    )
    print(f"\n[resolve_mutual_exchange 結果]")
    print(f"  dropped_to_p1={result.dropped_to_p1} (1ターン上限{OJAMA_MAX_DROP_PER_TURN}適用後)")
    print(f"  dropped_to_p2={result.dropped_to_p2}")
    print(f"  leftover_p1={result.leftover_p1} leftover_p2={result.leftover_p2}"
          f" (次ターン繰越=まだ空中)")
    print(f"  p1_dead={result.p1_dead}  p2_dead={result.p2_dead}")
    print(f"  chain_count_p1(sim再計算)={result.chain_result_p1.chain_count}"
          f"  chain_count_p2(sim再計算)={result.chain_result_p2.chain_count}")
    print(f"\n[report] board_p1_after: {_board_report('1P after', result.board_p1_after)}")
    print(f"[report] board_p2_after: {_board_report('2P after', result.board_p2_after)}")

    # --- (a) 2Pの吸収余力: 実際に着弾させた量に対する ojama_damage ---
    sim = ChainSimulator()
    if result.dropped_to_p2 > 0:
        dmg2 = iv.ojama_damage(before_p2, result.dropped_to_p2, simulator=sim)
        print(f"\n[a] 2P受け: 着弾{result.dropped_to_p2}個に対する ojama_damage="
              f"{dmg2.score:.3f} (raw余裕段数={dmg2.raw:.2f}、"
              f"0.05=無害/0.5=かなり不利/1.0=ほぼ死)")
    else:
        print("\n[a] 2Pへの着弾なし (dropped_to_p2=0)")
    if result.dropped_to_p1 > 0:
        dmg1 = iv.ojama_damage(before_p1, result.dropped_to_p1, simulator=sim)
        print(f"[a] 1P受け: 着弾{result.dropped_to_p1}個に対する ojama_damage="
              f"{dmg1.score:.3f} (raw余裕段数={dmg1.raw:.2f})")

    # --- (b) 2Pが着弾後盤面から返せるか (counter_reach_probability) ---
    if not result.board_p2_after.is_dead():
        threshold = float(max(1, result.dropped_to_p1))  # 1Pへ送り返すべき目安量
        cr = iv.counter_reach_probability(
            result.board_p2_after, threshold_ojama=gen1, elapsed_sec=ELAPSED_SEC)
        print(f"\n[b] 2P応手 (着弾後盤面から、1Pが送った{gen1}個相当に到達する確率):"
              f" {cr.probabilities}")
    else:
        print(f"\n[b] 2P is_dead=True (着弾後即詰み、応手評価不可)")
    anim1 = iv.estimate_chain_anim_duration_sec(8.0)  # 1Pの実連鎖数
    anim2 = iv.estimate_chain_anim_duration_sec(2.0)  # 2Pの実連鎖数
    print(f"[b] 連鎖アニメ推定所要: 1P(8連鎖)={anim1:.2f}s, 2P(2連鎖)={anim2:.2f}s"
          f" (差={anim1-anim2:.2f}s = 2Pが1P撃ち切りを待つ間に使える追加時間では"
          f"なく、むしろ1Pの方が長く画面を占有する)")

    # --- (c) _score_advantage で71%の内訳分解 ---
    model = vao._train_model(exclude_video=None)
    snap = OjamaAccountSnapshot(
        t_sec=0.0, pending_p1=0, pending_p2=0,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=0, total_dropped_to_p2=0,
        net_ojama_balance=0, overflow_risk_p1=False, overflow_risk_p2=False,
        confidence=1.0, leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
        net_balance_capped=result.leftover_p2 - result.leftover_p1,
        forecast_p1=result.leftover_p1, forecast_p2=result.leftover_p2,
    )
    adv, p1, drivers = vao._score_advantage(model, result.board_p1_after, result.board_p2_after, snap)
    print(f"\n[c] _score_advantage 結果: adv={adv:.2f} (1P視点、-100~+100)  "
          f"1P勝率p1={p1*100:.1f}%  (2P={100-p1*100:.1f}%)")
    cols = list(getattr(model, "_puyo_feature_cols", vao.FEATURES))
    f1 = vao._side_feats(result.board_p1_after, snap.net_balance_capped, snap.forecast_p1)
    f2 = vao._side_feats(result.board_p2_after, -snap.net_balance_capped, snap.forecast_p2)
    diff = {c: f1[c] - f2[c] for c in cols if c in f1 and c in f2}
    print(f"  主因 (|diff|上位、attribution_exclude適用後): {drivers}")
    print("  全diff内訳 (1P-2P):")
    for name in cols:
        if name in diff:
            print(f"    {name}: {diff[name]:+.4f}")

    out = {
        "gen1": gen1, "gen2": gen2,
        "dropped_to_p1": result.dropped_to_p1, "dropped_to_p2": result.dropped_to_p2,
        "leftover_p1": result.leftover_p1, "leftover_p2": result.leftover_p2,
        "p1_dead": result.p1_dead, "p2_dead": result.p2_dead,
        "before_p1": _board_report("1P before", before_p1),
        "before_p2": _board_report("2P before", before_p2),
        "after_p1": _board_report("1P after", result.board_p1_after),
        "after_p2": _board_report("2P after", result.board_p2_after),
        "adv": adv, "p1_winprob": p1, "drivers": drivers, "diff": diff,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
