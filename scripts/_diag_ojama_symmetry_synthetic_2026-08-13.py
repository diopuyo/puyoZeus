"""A-3 メカニズム特定: おじゃま収支 STABLE-dedup サンプリングの方向性バイアスを
OjamaAccountingTracker 単体 (動画・CNN 不使用) で高速に検証する
(2026-08-13、docs/CROSS_CUTTING_AUDIT_2026-08-13.md P4 の補完診断)。

## 仮説 (collect_boards_lean.py / src/ojama_accounting.py の code reading で導出)
1. _drive_ojama_accounting_lean() は毎フレーム
   on_state_transition → on_tsumo_settled(1P) → on_tsumo_settled(2P) →
   get_snapshot() の順で呼ばれる (scripts/collect_boards_lean.py:706-710)。
2. on_tsumo_settled(side) は「side 自身の pending forecast (forecast_incoming)」
   を drain する (src/ojama_accounting.py:498-517)。drain 後は
   p{side}_capped = min(forecast_incoming, CAP) が下がり、
   net_balance_capped = p2_capped - p1_capped が **side 自身に有利な方向へ動く**
   (自分の未消化お邪魔が減った瞬間は net が自分有利に見える、という会計上の
   定義そのものの性質。src/ojama_accounting.py:1043-1074)。
3. npz へ実際に emit される行は「その side 自身の盤面 grid が前回と変わった
   ときだけ」(scripts/collect_boards_lean.py:653-656 _should_emit)。
   自分の盤面が変わる最も確実な契機は「自分の tsumo が着地した瞬間」であり、
   これは on_tsumo_settled(自分) が呼ばれた **その直後** の1フレームと一致する
   (同一フレーム内で drain → get_snapshot の順)。
4. したがって「1P の emit 行」は構造的に「1P 自身が直前に drain した直後」に
   偏り、「2P の emit 行」も同様に「2P 自身が直前に drain した直後」に偏る。
   drain は必ず自分有利方向 (自分の own-perspective net を押し上げる) にしか
   動かないため、**両 side の emit 平均は自分に有利な方向へ同時にバイアスされ、
   相殺せずに加算される** (勝敗に依存しない方向性の説明)。

## 本スクリプトの検証方法
OjamaAccountingTracker を直接駆動する (完全に対称な乱数プロセス、
どちらの side にも実力差を作り込まない)。
  - 各 tick で乱数により「どちらかの side が相手へおじゃまを送る」
    (forecast_incoming に直接加算、chain 検出ステートマシンは経由しない
    単体テスト、対象は会計コアのみ)
  - 各 tick で一定確率により「1P が着地」「2P が着地」を独立に発生させ、
    on_tsumo_settled を呼ぶ (= emit 代理イベント)
  - 毎 tick get_snapshot() を呼び dense_log に記録 (瞬時対称性の確認用)
  - 各 side の「着地 tick での net 値」を emit 代理配列として集計し、
    2 配列の平均の和が 0 からどれだけ・どちら向きにずれるかを測る
多数の乱数シードで繰り返し、符号が一貫して同じ向きになるかを確認する
(一貫していれば「勝敗に依存しないタイミング構造バイアス」の証拠となる)。

本体 (src/ojama_accounting.py) は一切変更しない。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402

N_TICKS: int = 3000
N_SEEDS: int = 30
P_SEND: float = 0.03        # 1 tick あたり、どちらかが相手に送る確率
SEND_AMOUNT_RANGE: tuple[int, int] = (6, 30)
P_PLACE_1P: float = 0.20    # 1 tick あたり 1P が着地する確率 (対称、実力差なし)
P_PLACE_2P: float = 0.20    # 同上 2P


def _run_one_seed(seed: int) -> dict:
    """1 シード分のシミュレーションを実行し、瞬時対称性 + emit代理バイアスを測る。"""
    rng = random.Random(seed)
    tracker = OjamaAccountingTracker()
    tracker.reset()

    dense_residuals: list[float] = []
    emitted_1p: list[float] = []
    emitted_2p: list[float] = []

    for tick in range(N_TICKS):
        t_sec = tick * 0.1
        # --- 対称乱数: どちらかがランダムに相手へおじゃまを送る (chain 相当) ---
        if rng.random() < P_SEND:
            amount = rng.randint(*SEND_AMOUNT_RANGE)
            if rng.random() < 0.5:
                tracker._p2.forecast_incoming += amount  # 1P が 2P へ送る
            else:
                tracker._p1.forecast_incoming += amount  # 2P が 1P へ送る

        # --- 対称乱数: 独立に 1P/2P が着地 (tsumo settle = 自盤面変化の代理) ---
        place_1p = rng.random() < P_PLACE_1P
        place_2p = rng.random() < P_PLACE_2P
        if place_1p:
            tracker.on_tsumo_settled("p1", t_sec)
        if place_2p:
            tracker.on_tsumo_settled("p2", t_sec)

        snap = tracker.get_snapshot(t_sec)
        net_1p = float(snap.net_balance_capped)
        net_2p = -net_1p
        dense_residuals.append(net_1p + net_2p)

        # emit 代理: 「自分が着地した tick の net 値」をその side の npz 相当行とする
        if place_1p:
            emitted_1p.append(net_1p)
        if place_2p:
            emitted_2p.append(net_2p)

    mean_1p = float(np.mean(emitted_1p)) if emitted_1p else None
    mean_2p = float(np.mean(emitted_2p)) if emitted_2p else None
    residual = (mean_1p + mean_2p) if mean_1p is not None and mean_2p is not None else None
    return {
        "seed": seed,
        "max_abs_dense_residual": float(np.max(np.abs(dense_residuals))),
        "n_emitted_1p": len(emitted_1p),
        "n_emitted_2p": len(emitted_2p),
        "mean_net_1p_emitted": mean_1p,
        "mean_net_2p_emitted": mean_2p,
        "residual_mean_should_be_0_if_synced": residual,
    }


def main() -> int:
    results = [_run_one_seed(seed) for seed in range(N_SEEDS)]
    residuals = np.array([
        r["residual_mean_should_be_0_if_synced"] for r in results
        if r["residual_mean_should_be_0_if_synced"] is not None
    ])
    max_dense = max(r["max_abs_dense_residual"] for r in results)
    summary = {
        "n_seeds": N_SEEDS,
        "n_ticks_per_seed": N_TICKS,
        "params": {
            "P_SEND": P_SEND, "SEND_AMOUNT_RANGE": SEND_AMOUNT_RANGE,
            "P_PLACE_1P": P_PLACE_1P, "P_PLACE_2P": P_PLACE_2P,
        },
        "max_abs_dense_residual_over_all_seeds": max_dense,
        "residual_mean_over_seeds": float(np.mean(residuals)),
        "residual_std_over_seeds": float(np.std(residuals)),
        "n_seeds_residual_positive": int((residuals > 0).sum()),
        "n_seeds_residual_negative": int((residuals < 0).sum()),
        "per_seed_results": results,
    }
    out_path = Path("logs/diag_ojama_symmetry_synthetic_2026-08-13.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed_results"},
                      ensure_ascii=False, indent=2))
    print(f"[diag] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
