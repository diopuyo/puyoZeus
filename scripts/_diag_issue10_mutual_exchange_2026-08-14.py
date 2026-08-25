"""デモレビュー指摘#10 (docs/DEMO_REVIEW_2026-08-13.md) の定量裏取り計装スクリプト。

デモ1 v3 (source=data/frames/review_demo_2026-08-12.mp4、デモt=33-38 = source t=195-200)
の両者発火場面で resolve_mutual_exchange / _score_advantage が実際に何を計算したかを、
本番と全く同じコマンドライン (logs/demo_fixed_v3_2026-08-13.log の [cmd] 行) で
scripts.visualize_advantage_overlay.generate() を再実行し、内部関数呼び出しを
非破壊的にフックして記録する (本体コードは一切変更しない)。

再現条件の一致 (feedback_accuracy_claims_distribution / 教訓「密サンプリングでは
再現せず stride=2 で初めて再現」) を徹底するため、CLIフラグ・start_sec (=162、
真の試合開始直前から通しで実行し状態機械/会計トラッカーの連続性を保つ)・
sample_interval=0 は本番ログと完全一致させる。end_sec のみ 210 に短縮する
(未来のフレームは過去の計算結果に影響しないため安全な高速化)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board import Board  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_issue10")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_JSON = Path("logs/_diag_issue10_mutual_exchange_2026-08-14.json")

# ============================
# フック: 現在フレームの real t (source動画の絶対秒) を捕捉する
# ============================
_STATE: dict[str, float | None] = {"t": None}
_orig_pipe_update = RecognitionPipeline.update


_PROGRESS_COUNTER = {"n": 0}


def _patched_pipe_update(self, frame_idx, time_sec, frame):  # noqa: ANN001
    _STATE["t"] = float(time_sec)
    _PROGRESS_COUNTER["n"] += 1
    # 進捗可視化 (単一プロセス制約下でのCPU競合による低速化の実態把握用、
    # 30フレームごとに print、負荷は無視できる)。
    if _PROGRESS_COUNTER["n"] % 30 == 0:
        print(f"[progress] frame#{_PROGRESS_COUNTER['n']} t={time_sec:.2f}s", flush=True)
    return _orig_pipe_update(self, frame_idx, time_sec, frame)


RecognitionPipeline.update = _patched_pipe_update


def _board_heights(b: Board) -> list[int]:
    return [b.height_of(c) for c in range(6)]


def _board_summary(b: Board) -> dict:
    return {
        "heights": _board_heights(b),
        "count_puyos": b.count_puyos(),
        "is_dead": b.is_dead(),
        "death_margin_raw": 12 - b.height_of(2),
        # 再現用の完全グリッド (後続の counter_reach/ojama_damage 分析を
        # 別プロセスの重い再実行なしで行うため、本パスで一度だけ保存する)。
        "grid": b.to_dict()["grid"],
    }


# ============================
# フック1: resolve_mutual_exchange (決着計算の中身)
# ============================
LOG_EXCHANGE: list[dict] = []
_orig_resolve_mutual = vao.resolve_mutual_exchange


def _patched_resolve_mutual(
    before_p1, before_p2, gen_p1_ojama, gen_p2_ojama, pending_p1, pending_p2,
    simulator=None,
):
    result = _orig_resolve_mutual(
        before_p1, before_p2, gen_p1_ojama, gen_p2_ojama, pending_p1, pending_p2,
        simulator=simulator,
    )
    LOG_EXCHANGE.append({
        "t": _STATE["t"],
        "before_p1": _board_summary(before_p1),
        "before_p2": _board_summary(before_p2),
        "gen_p1_ojama": gen_p1_ojama, "gen_p2_ojama": gen_p2_ojama,
        "pending_p1": pending_p1, "pending_p2": pending_p2,
        "chain_count_p1": result.chain_result_p1.chain_count,
        "chain_count_p2": result.chain_result_p2.chain_count,
        "chain_total_erased_p1": result.chain_result_p1.total_erased,
        "chain_total_erased_p2": result.chain_result_p2.total_erased,
        "chain_total_ojama_p1": result.chain_result_p1.total_ojama,
        "chain_total_ojama_p2": result.chain_result_p2.total_ojama,
        "dropped_to_p1": result.dropped_to_p1, "dropped_to_p2": result.dropped_to_p2,
        "leftover_p1": result.leftover_p1, "leftover_p2": result.leftover_p2,
        "p1_dead": result.p1_dead, "p2_dead": result.p2_dead,
        "board_p1_after": _board_summary(result.board_p1_after),
        "board_p2_after": _board_summary(result.board_p2_after),
    })
    return result


vao.resolve_mutual_exchange = _patched_resolve_mutual

# ============================
# フック2: _score_advantage (71%の内訳分解)
# ============================
LOG_SCORE_ADV: list[dict] = []
_orig_score_adv = vao._score_advantage


def _patched_score_adv(model, b1, b2, snap, feature_cols=None,
                        attribution_exclude=vao.ATTRIBUTION_EXCLUDED_INDICATORS):
    adv, p1, drivers = _orig_score_adv(
        model, b1, b2, snap, feature_cols=feature_cols,
        attribution_exclude=attribution_exclude)
    cols = list(feature_cols) if feature_cols is not None else list(
        getattr(model, "_puyo_feature_cols", vao.FEATURES))
    f1 = vao._side_feats(b1, snap.net_balance_capped, snap.forecast_p1)
    f2 = vao._side_feats(b2, -snap.net_balance_capped, snap.forecast_p2)
    diff = {c: f1[c] - f2[c] for c in cols if c in f1 and c in f2}
    LOG_SCORE_ADV.append({
        "t": _STATE["t"], "adv": adv, "p1": p1,
        "drivers": drivers, "diff": diff, "cols": cols,
        "net_balance_capped": snap.net_balance_capped,
        "forecast_p1": snap.forecast_p1, "forecast_p2": snap.forecast_p2,
    })
    return adv, p1, drivers


vao._score_advantage = _patched_score_adv


def main() -> None:
    video = Path("data/frames/review_demo_2026-08-12.mp4")
    out = OUT_DIR / "diag_issue10.mp4"
    n = vao.generate(
        video=video, out=out, max_sec=0.0, sample_interval=0.0,
        start_sec=162.0, end_sec=202.0,
        show_recognition=True,
        enable_early_fire_reaction=True,
        enable_per_side_settled=True,
        disable_score_lead_bias=True,
        disable_pressure=True,
        # counter_reach (打ち合い応手確率) は重い MC ビームサーチだが、
        # resolve_mutual_exchange/_score_advantage (本計装の対象) には
        # 一切混ざらない独立の表示専用チャンネル (hold中は disp_adv/disp_p1
        # を resolved_tracker.hold_* で完全上書きするため、W_COUNTER/W_THREAT
        # 等の他ライブ成分は不使用、scripts/visualize_advantage_overlay.py
        # ResolvedExchangeTracker docstring 参照)。単一プロセス制約下での
        # 高速化のため本計装では無効化する (資源負荷=フルCSV生成走行中との
        # 衝突回避、結果の対象値には無関係)。
        enable_counter_reach=False,
        enable_counter_remaining_time=False,
        enable_counter_defender_only=False,
        enable_resolved_exchange_eval=True,
        enable_pseudo_chain_score_fill=True,
        stable_majority_window=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
        layout="panel",
        render=False,
    )
    print(f"[done] frames written={n}")

    scene_exchange = [r for r in LOG_EXCHANGE if r["t"] is not None and 190.0 <= r["t"] <= 205.0]
    scene_adv = [r for r in LOG_SCORE_ADV if r["t"] is not None and 190.0 <= r["t"] <= 205.0]
    print(f"[scene] resolve_mutual_exchange calls in 190-205s: {len(scene_exchange)}")
    print(f"[scene] _score_advantage calls in 190-205s: {len(scene_adv)}")
    for r in scene_exchange:
        print(json.dumps(r, ensure_ascii=False, default=str))
    for r in scene_adv:
        print(json.dumps(r, ensure_ascii=False, default=str))

    with LOG_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "exchange_all": LOG_EXCHANGE, "score_adv_all": LOG_SCORE_ADV,
            "exchange_scene": scene_exchange, "score_adv_scene": scene_adv,
        }, f, ensure_ascii=False, indent=1, default=str)
    print(f"[saved] {LOG_JSON}")


if __name__ == "__main__":
    main()
