"""Gate 3 未照合 561 個の内訳診断 (2026-08-25、デバッガエージェント計装)。

## 目的

`data/verify/gate3_episode_v3_2026-08-25/v51/diagnostics.json` で確定した
未照合 1,163 個 (v51) のうち、退役 (ワイプ) 602 個を除いた残り 561 個が
何なのかを、以下 3 仮説について計装で数値確定する (根因確定タスク #1)。

- 仮説α: `cancel_own_pending_then_send_surplus` の自己相殺により、
  新規発火 chain の FIRE 額 (= gen 全部) が、本来 `gen - 自己相殺分` で
  あるべきところ過大計上されている (自己相殺分は「別の (相手側の古い)
  chain」の相殺として正しく処理されるが、発火した本人の chain 自身の
  outstanding からは絶対に差し引かれない、という `src/exchange_ledger.py`
  の `_attribute` の設計上の構造)。
- 仮説β: `_drain_by_tsumo_delta` のツモ設置検知そのものが機能していない
  (着弾が観測できない構造的理由)。
- 仮説γ: 窓 (t0=459, t1=545) の末尾に次ラウンドの序盤が含まれ、
  未決着のまま残っている。

## 変更しないもの

- `src/` 配下は読むだけ。変更・パッチ・モンキーパッチ禁止。
- `scripts/_gate3_episode_probe_v3_2026-08-25.py` は変更しない
  (本スクリプトはそのロジックを独立にコピーして計装を追加しただけ)。
- 出力は新規ディレクトリ `data/verify/gate3_unrec_2026-08-25/` のみ。

## 計装の追加点 (probe v3 との差分)

1. 全フレームで `snap.pending_p1_uncapped`/`pending_p2_uncapped` の値そのもの
   (差分でなく水準) を時系列で記録する (仮説β/γ用の可視化)。
2. `pipeline.tsumo_count(side)` の増分 (= ツモ設置検知の生カウント) を
   側別に window 全体で積算する (仮説β)。
3. `snap.total_offset_by_p1`/`total_offset_by_p2` (自己相殺の累積、cap 前後
   の "capped" 側の帳簿。uncapped 側の実測とは経路が違うため参考値扱い)
   の window 内デルタを記録する。
4. `finish()` 後、`episode_tracker._ledger._chains` (chain_id 単位の最終
   ChainRecord) を直接ダンプし、各 chain の `amount`/`canceled`/`landed`/
   `outstanding`/`oversettled` を側別に集計する (仮説α検証の核心。
   ledger の非公開属性を読むだけで、書き換えない)。
5. `episode_tracker._ledger.current_episode()` の有無をダンプする
   (窓の終わりに OPEN な episode が残っていれば、D1 の
   `closed_episodes()` 集計から丸ごと漏れている可能性を可視化する)。
6. t1 を可変にし、γ検証用に短い窓 (次ラウンド開始前で打ち切り) でも
   再実行できるようにする。

使い方:
  python scripts/_diag_gate3_unrec_2026-08-25.py \\
      --video data/frames/video_51.mp4 --t0 459.0 --t1 545.0 \\
      --out data/verify/gate3_unrec_2026-08-25/v51_full.json
  python scripts/_diag_gate3_unrec_2026-08-25.py \\
      --video data/frames/video_51.mp4 --t0 459.0 --t1 533.0 \\
      --out data/verify/gate3_unrec_2026-08-25/v51_gamma.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

from scripts.collect_indicators_v2 import (  # noqa: E402
    DEFAULT_FPS,
    TARGET_H,
    TARGET_W,
    _drive_ojama,
    _SideTracker,
    _update_game_idx,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    ExchangeEpisodeTracker,
    GenerationObservation,
    PendingUncappedFrame,
    classify_pending_uncapped_delta,
)
import src.exchange_ledger as _exchange_ledger_module  # noqa: E402
from src.exchange_ledger import Side  # noqa: E402

# --- chain_id=6 消失トレース用の一時ラップ (診断専用、src は書き換えない) ---
# `ExchangeLedger.push` を薄くラップして呼び出しを記録するだけ。元の
# 挙動 (戻り値・副作用) は一切変えない (`_orig_push` をそのまま呼ぶだけ)。
_PUSH_TRACE_LOG: list[dict] = []
_orig_ledger_push = _exchange_ledger_module.ExchangeLedger.push


def _traced_ledger_push(self, ev, ctx):  # noqa: ANN001
    _PUSH_TRACE_LOG.append({
        "kind": ev.kind.name, "side": ev.side.name, "t_sec": ev.t_sec,
        "amount": ev.amount, "chain_id": ev.chain_id, "seq": ev.seq,
        "ctx_game_idx": ctx.game_idx,
        "ledger_game_idx_before": self._game_idx,  # noqa: SLF001
    })
    return _orig_ledger_push(self, ev, ctx)


_exchange_ledger_module.ExchangeLedger.push = _traced_ledger_push
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

WARMUP_SEC: float = 30.0

FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

_SIDE_LABELS: tuple[str, str] = ("1P", "2P")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t1", type=float, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def _chain_event_key(ev: ChainEvent | None) -> tuple | None:
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


def _to_chain_observation(
    side_label: str, ev: ChainEvent, t_sec: float, game_idx: int, elapsed_sec: float,
) -> ChainEventObservation:
    return ChainEventObservation(
        side=side_label, t_sec=t_sec, mechanism=ev.mechanism or "",
        chain_count=ev.chain_count, total_score=ev.total_score,
        ojama_sent=ev.ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
    )


def _diff_generation(prev_snap, snap) -> dict[str, int]:
    return {
        "1P": snap.total_generated_by_p1 - prev_snap.total_generated_by_p1,
        "2P": snap.total_generated_by_p2 - prev_snap.total_generated_by_p2,
    }


def _make_pending_frame(
    t_sec: float, game_idx: int, snap,
    p1_tsumo: bool, p2_tsumo: bool, gen_diff: dict[str, int] | None,
) -> PendingUncappedFrame:
    p1_fin = gen_diff is not None and gen_diff["1P"] != 0
    p2_fin = gen_diff is not None and gen_diff["2P"] != 0
    return PendingUncappedFrame(
        t_sec=t_sec, game_idx=game_idx,
        p1_uncapped=float(snap.pending_p1_uncapped),
        p2_uncapped=float(snap.pending_p2_uncapped),
        p1_tsumo_placed=p1_tsumo, p2_tsumo_placed=p2_tsumo,
        p1_chain_finalized=p1_fin, p2_chain_finalized=p2_fin,
    )


def _dump_ledger_chains(episode_tracker: ExchangeEpisodeTracker) -> dict:
    """仮説α検証の核心: chain_id 単位の最終会計状態を直接ダンプする。

    `episode_tracker._ledger._chains` は非公開属性だが、**読むだけ**
    (書き換えない)。診断専用のこのスクリプトでしか使わない。
    """
    ledger = episode_tracker._ledger  # noqa: SLF001
    chains = ledger._chains  # noqa: SLF001
    rows = []
    for cid, rec in sorted(chains.items()):
        rows.append({
            "chain_id": cid,
            "side": rec.side.name,
            "opened_at_sec": rec.opened_at_sec,
            "amount": rec.amount,
            "provisional_amount": rec.provisional_amount,
            "finalized_amount": rec.finalized_amount,
            "canceled": rec.canceled,
            "landed": rec.landed,
            "outstanding": rec.outstanding,
            "oversettled": rec.oversettled,
            "state": rec.state.name,
        })
    open_ep = ledger.current_episode()
    return {
        "still_in_ledger_chains": rows,
        "sum_outstanding_still_in_ledger": sum(r["outstanding"] for r in rows),
        "sum_amount_still_in_ledger": sum(r["amount"] for r in rows),
        "sum_canceled_still_in_ledger": sum(r["canceled"] for r in rows),
        "sum_landed_still_in_ledger": sum(r["landed"] for r in rows),
        "open_episode_present": open_ep is not None,
        "open_episode_id": open_ep.episode_id if open_ep else None,
        "open_episode_opened_at_sec": open_ep.opened_at_sec if open_ep else None,
        "open_episode_event_count": len(open_ep.events) if open_ep else None,
    }


def _dump_closed_episode_details(episode_tracker: ExchangeEpisodeTracker) -> list[dict]:
    ledger = episode_tracker._ledger  # noqa: SLF001
    return [
        {
            "episode_id": s.episode_id, "status": s.status.name,
            "close_reason": s.close_reason,
            "opened_at_sec": s.opened_at_sec, "closed_at_sec": s.closed_at_sec,
            "total_generated": s.total_generated, "total_canceled": s.total_canceled,
            "total_landed": s.total_landed, "unreconciled": s.unreconciled,
            "has_settlement_input": s.has_settlement_input,
            "oversettled": s.oversettled,
        }
        for s in ledger.closed_episodes()
    ]


def _run(video: Path, t0: float, t1: float, out_path: Path) -> None:  # noqa: PLR0915
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_sec = max(0.0, t0 - WARMUP_SEC)
    start_frame = int(start_sec * fps)
    end_frame = int(t1 * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        **FORMULA_FLAGS,
    )
    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    tracker_p1 = _SideTracker()
    tracker_p2 = _SideTracker()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    episode_tracker = ExchangeEpisodeTracker(enabled=True)

    last_chain_key: dict[str, tuple | None] = {"1P": None, "2P": None}
    prev_snap = None
    prev_pending_frame: PendingUncappedFrame | None = None
    prev_tsumo = {"1P": None, "2P": None}

    # --- 追加計装 (probe v3 との差分) ---
    pending_timeseries: list[dict] = []  # 仮説β/γ: 水準の時系列
    n_tsumo_placed_total = {"1P": 0, "2P": 0}  # 仮説β: 検知された設置回数
    n_tsumo_placed_with_pending_gt0 = {"1P": 0, "2P": 0}  # 設置時に pending>0だった回数
    tsumo_placement_log: list[dict] = []  # 全設置イベントの (t_sec, side, pre_pending)
    settlement_event_log: list[dict] = []  # 全 CANCEL/LAND イベントの生ログ
    offset_start = {"1P": None, "2P": None}
    offset_end = {"1P": None, "2P": None}
    dropped_start = {"1P": None, "2P": None}
    dropped_end = {"1P": None, "2P": None}

    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        if fi % 3600 == 0:
            print(f"[progress] fi={fi} t={t_sec:.2f}", flush=True)
        result = pipeline.update(fi, t_sec, frame)
        snap = _drive_ojama(
            ojama_tracker, result.p1, result.p2, prev_state_p1, prev_state_p2, t_sec,
            tracker_p1=tracker_p1, tracker_p2=tracker_p2, pipeline=pipeline,
        )
        _update_game_idx(tracker_p1, result.p1.score)
        _update_game_idx(tracker_p2, result.p2.score)
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state

        game_idx = max(tracker_p1.game_idx, tracker_p2.game_idx)
        elapsed_sec = ojama_tracker._elapsed(t_sec)  # noqa: SLF001
        gen_diff = _diff_generation(prev_snap, snap) if prev_snap is not None else None

        for side_label, side_result in (("1P", result.p1), ("2P", result.p2)):
            ev = side_result.chain_event
            key = _chain_event_key(ev)
            changed = key != last_chain_key[side_label]
            last_chain_key[side_label] = key
            if changed and t_sec >= t0 and ev is not None:
                episode_tracker.observe(
                    _to_chain_observation(side_label, ev, t_sec, game_idx, elapsed_sec),
                )

        if gen_diff is not None and t_sec >= t0:
            for side_label in _SIDE_LABELS:
                delta = gen_diff[side_label]
                if delta == 0:
                    continue
                if delta > 0:
                    episode_tracker.observe_generation(GenerationObservation(
                        side=side_label, t_sec=t_sec, game_idx=game_idx,
                        generated_delta=delta,
                    ))

        tsumo_1p = pipeline.tsumo_count("1P")
        tsumo_2p = pipeline.tsumo_count("2P")
        p1_tsumo_placed = prev_tsumo["1P"] is not None and tsumo_1p > prev_tsumo["1P"]
        p2_tsumo_placed = prev_tsumo["2P"] is not None and tsumo_2p > prev_tsumo["2P"]

        if t_sec >= t0:
            # 【重要】`snap` はこのフレームの `_drive_ojama` (drain含む) 適用後の
            # 値。着弾判定の直前 (このフレームの drain が起きる前) の水準を見る
            # には、まだ上書きされていない前フレームの `prev_snap` を使う必要が
            # ある (post-drain 値で判定すると drain 直後は必ず 0 に近づき、
            # 「もともと何もなかった」のか「ちょうど今drainされた」のか区別
            # できない誤り、初版で混入した測定器バグ)。
            pre_p1 = prev_snap.pending_p1_uncapped if prev_snap is not None else 0.0
            pre_p2 = prev_snap.pending_p2_uncapped if prev_snap is not None else 0.0
            if p1_tsumo_placed:
                n_tsumo_placed_total["1P"] += 1
                if pre_p1 > 0:
                    n_tsumo_placed_with_pending_gt0["1P"] += 1
                tsumo_placement_log.append({
                    "t_sec": round(t_sec, 3), "side": "1P",
                    "pre_pending": pre_p1, "post_pending": snap.pending_p1_uncapped,
                })
            if p2_tsumo_placed:
                n_tsumo_placed_total["2P"] += 1
                if pre_p2 > 0:
                    n_tsumo_placed_with_pending_gt0["2P"] += 1
                tsumo_placement_log.append({
                    "t_sec": round(t_sec, 3), "side": "2P",
                    "pre_pending": pre_p2, "post_pending": snap.pending_p2_uncapped,
                })
            if offset_start["1P"] is None:
                offset_start["1P"] = snap.total_offset_by_p1
                offset_start["2P"] = snap.total_offset_by_p2
                dropped_start["1P"] = snap.total_dropped_to_p1
                dropped_start["2P"] = snap.total_dropped_to_p2
            offset_end["1P"] = snap.total_offset_by_p1
            offset_end["2P"] = snap.total_offset_by_p2
            dropped_end["1P"] = snap.total_dropped_to_p1
            dropped_end["2P"] = snap.total_dropped_to_p2
            pending_timeseries.append({
                "t_sec": round(t_sec, 3),
                "p1_uncapped": snap.pending_p1_uncapped,
                "p2_uncapped": snap.pending_p2_uncapped,
            })

        curr_pending_frame = _make_pending_frame(
            t_sec, game_idx, snap, p1_tsumo_placed, p2_tsumo_placed, gen_diff,
        )
        if prev_pending_frame is not None and t_sec >= t0:
            classification = classify_pending_uncapped_delta(
                prev_pending_frame, curr_pending_frame,
            )
            if classification.settlement is not None:
                episode_tracker.observe_settlement(classification.settlement)
                s = classification.settlement
                for side_label, cancel_amt, land_amt in (
                    ("1P", s.canceled_by_1p, s.landed_on_1p),
                    ("2P", s.canceled_by_2p, s.landed_on_2p),
                ):
                    if cancel_amt > 0:
                        settlement_event_log.append({
                            "t_sec": round(t_sec, 3), "kind": "CANCEL",
                            "side": side_label, "amount": cancel_amt,
                        })
                    if land_amt > 0:
                        settlement_event_log.append({
                            "t_sec": round(t_sec, 3), "kind": "LAND",
                            "side": side_label, "amount": land_amt,
                        })
            for side in classification.wiped_sides:
                episode_tracker.observe_wipe(side, t_sec, game_idx)
        prev_pending_frame = curr_pending_frame
        prev_tsumo["1P"], prev_tsumo["2P"] = tsumo_1p, tsumo_2p
        prev_snap = snap
        fi += 1

    cap.release()
    episode_tracker.finish()
    diag = episode_tracker.diagnostics()

    ledger_chain_dump = _dump_ledger_chains(episode_tracker)
    closed_episode_details = _dump_closed_episode_details(episode_tracker)

    # --- chain_id=6 が ledger._chains から消えている謎の追跡用デバッグ ---
    resolved_dbg = episode_tracker._resolver.resolved()  # noqa: SLF001
    ojama_dbg = episode_tracker._convert_resolved_to_ojama(resolved_dbg)  # noqa: SLF001
    # 2026-08-25 実装3: _build_events は自己相殺ルックアップを追加引数で要求する
    # (署名変更、このデバッグ専用スクリプトのみ追随)。
    self_cancel_lookup_dbg = episode_tracker._build_self_cancel_lookup()  # noqa: SLF001
    events_dbg = episode_tracker._build_events(  # noqa: SLF001
        resolved_dbg, ojama_dbg, self_cancel_lookup_dbg,
    )
    chain6_debug = {
        "resolved_rc": next(
            (
                {
                    "chain_id": rc.chain_id, "side": rc.side,
                    "opened_at_sec": rc.opened_at_sec, "closed_at_sec": rc.closed_at_sec,
                    "provisional_score": rc.provisional_score,
                    "finalized_score": rc.finalized_score,
                    "was_finalized": rc.was_finalized,
                    "finalized_source": rc.finalized_source,
                    "growth_observed": rc.growth_observed,
                }
                for rc in resolved_dbg if rc.chain_id == 6
            ), None,
        ),
        "ojama_for_chain6": (
            {
                "provisional_ojama": ojama_dbg[6].provisional_ojama,
                "finalized_ojama": ojama_dbg[6].finalized_ojama,
            } if 6 in ojama_dbg else None
        ),
        "events_for_chain6": [
            {"kind": ev.kind.name, "side": ev.side.name, "t_sec": ev.t_sec, "amount": ev.amount}
            for ev, _ in events_dbg if ev.chain_id == 6
        ],
        "all_resolved_chain_ids": sorted(rc.chain_id for rc in resolved_dbg),
        "all_event_chain_ids": sorted({ev.chain_id for ev, _ in events_dbg if ev.chain_id is not None}),
        "raw_ledger_chains_keys": sorted(episode_tracker._ledger._chains.keys()),  # noqa: SLF001
    }

    # pending の水準時系列から「0 でなかったフレーム数」も出す (仮説β/γ)
    n_frames_p1_pending_gt0 = sum(
        1 for r in pending_timeseries if r["p1_uncapped"] > 0
    )
    n_frames_p2_pending_gt0 = sum(
        1 for r in pending_timeseries if r["p2_uncapped"] > 0
    )

    report = {
        "video": str(video), "window": {"t0": t0, "t1": t1},
        "d1": dataclasses.asdict(diag.d1),
        "hypothesis_alpha_ledger_chain_dump": ledger_chain_dump,
        "hypothesis_alpha_closed_episode_details": closed_episode_details,
        "hypothesis_beta_tsumo_detection": {
            "n_tsumo_placed_total": n_tsumo_placed_total,
            "n_tsumo_placed_with_pending_gt0": n_tsumo_placed_with_pending_gt0,
            "n_frames_observed": len(pending_timeseries),
            "n_frames_p1_pending_gt0": n_frames_p1_pending_gt0,
            "n_frames_p2_pending_gt0": n_frames_p2_pending_gt0,
        },
        "self_cancel_capped_bookkeeping_delta_reference_only": {
            "1P": (
                None if offset_end["1P"] is None
                else offset_end["1P"] - offset_start["1P"]
            ),
            "2P": (
                None if offset_end["2P"] is None
                else offset_end["2P"] - offset_start["2P"]
            ),
        },
        "capped_total_dropped_delta_reference_only": {
            "1P": (
                None if dropped_end["1P"] is None
                else dropped_end["1P"] - dropped_start["1P"]
            ),
            "2P": (
                None if dropped_end["2P"] is None
                else dropped_end["2P"] - dropped_start["2P"]
            ),
        },
        "pending_timeseries_sample_every_60th_frame": pending_timeseries[::60],
        "tsumo_placement_log": tsumo_placement_log,
        "settlement_event_log": settlement_event_log,
        "chain6_debug": chain6_debug,
        "push_trace_log": _PUSH_TRACE_LOG,
    }
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "pending_timeseries_sample_every_60th_frame"}, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    args = _parse_args()
    _run(Path(args.video), args.t0, args.t1, Path(args.out))
