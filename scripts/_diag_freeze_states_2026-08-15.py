"""指摘14系「中間値のまま固定された凍結」5件の状態内訳診断 (2026-08-15、デバッガ)。

## 目的
user所感 (2026-08-15): 「お互いが撃ち合っているなら動かないのは正しい。
しかし、どちらかの連鎖が終わってツモを置き始めたら、置くたびに勝率は
微妙に変動するはず」を検証基準として、
`data/verify/backtest_issue14_2026-08-15/npz/on/*.npz` で検出済みの5件の
凍結区間について、0.1秒刻みで両者の状態 (BoardState/chain_event有無/
tsumo_count/score/盤面ハッシュ) を計装し、以下に分類する。

  (A) 全時間帯で両者連鎖中 (busy) → 所感に照らして正当
  (B) 途中から片側の連鎖が終わり設置が始まっているのに disp_adv 固定 → 不備
  (C) 決着ホールド (resolved_active) が立っていないのに固定 → 別機構、
      盤面/tsumo_count が本当に不変かまで確認する

## フラグ構成 (本番相当、厳守)
`scripts/_gen_demo_final4_2026-08-15.sh` の CLI フラグ列と同一の kwargs を
`FINAL4_KWARGS()` で構築する (stable_majority_window / entry_hardening /
scoped_exit は final4 で明示指定されていないため None のまま = 既定 False
に落ちる、docstring 参照)。

## 計装方式 (read-only, production コード無変更)
`scripts._backtest_issue14_flags_2026-08-15.py` と同じ「モジュール属性の
薄いラッパー差し替え」方式を踏襲する (このプロセス内限定、実行後に元へ戻す)。
- `RecognitionPipeline.update` をラップし、戻り値 r (PipelineResult) から
  (t, state1, state2, chain1, chain2, score1, score2, board_hash1,
  board_hash2, next_pair1, next_pair2) を記録。tsumo_count は同フレームで
  self (= pipe インスタンス) から取得する。
- `ResolvedExchangeTracker.update` をラップし (t, active, just_deactivated,
  awaiting_landing) を記録 (awaiting_landing は self._awaiting_landing を
  read-only 参照、下線始まりだが計装専用の外部参照であり production コード
  自体は無変更)。
- disp_adv 系列は `debug_history_out` (既存の公開 API) から取得。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_freeze_states_2026-08-15
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================
# 定数
# ============================

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = _ROOT / "data" / "verify" / "diag_freeze_2026-08-15"
LOG_DIR = _ROOT / "logs" / "diag_freeze_2026-08-15"

# 事象窓の前後マージン [秒] (freeze開始前/終了後の文脈を見るため)
PRE_MARGIN_SEC: float = 3.0
POST_MARGIN_SEC: float = 4.0

# レポート表示の時間刻み [秒] (計測自体は生フレーム=約1/30秒粒度、
# 表示のみ0.1秒刻みへ間引く)
REPORT_STEP_SEC: float = 0.1


@dataclass(frozen=True)
class FreezeEvent:
    """user報告の凍結事象1件分 (優先度順)。"""

    rank: int
    video_id: str
    t_start_abs: float
    freeze_len_sec: float
    fixed_value_desc: str
    hold_fraction_pct: float  # 報告済みの「決着ホールド中の割合」


# userタスク記載の5件 (data/verify/backtest_issue14_2026-08-15/npz/on/*.npz で検出済み)
FREEZE_EVENTS: tuple[FreezeEvent, ...] = (
    FreezeEvent(1, "c17", 1240.9, 8.60, "adv +8.2 (1P 60.1%)", 100.0),
    FreezeEvent(2, "c14", 1317.3, 6.27, "adv +44.6 (1P 90.3%)", 99.0),
    FreezeEvent(3, "c20", 2334.3, 3.93, "adv +26.2 (1P 78.8%)", 100.0),
    FreezeEvent(4, "c22", 2999.8, 6.40, "adv +19.0 (1P 72.1%)", 0.0),
    FreezeEvent(5, "c14", 2842.7, 3.63, "adv +0.0 (ちょうど50.0%)", 0.0),
)


def final4_kwargs() -> dict[str, object]:
    """scripts/_gen_demo_final4_2026-08-15.sh と同一の kwargs (本番相当構成)。

    stable_majority_window / enable_ojama_fall_entry_hardening /
    enable_ojama_fall_scoped_exit は final4.sh で明示指定されていないため
    kwargs に含めない (generate() 側の既定 None = 実質 False に落ちる、
    2026-08-15 不採用確定分)。
    """
    return dict(
        sample_interval=0.0,
        show_recognition=False,
        render=False,
        layout="panel",
        enable_early_fire_reaction=True,
        enable_per_side_settled=True,
        disable_score_lead_bias=True,
        disable_pressure=True,
        enable_counter_remaining_time=True,
        enable_counter_defender_only=True,
        enable_ojama_fall_placement_override=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_pseudo_chain_score_fill=True,
        enable_resolved_live_defender=True,
        enable_resolved_live_defender_strict=True,
        enable_resolved_kill_override=True,
    )


def _board_hash(board) -> "int | None":
    if board is None:
        return None
    return hash(board.grid_bytes())


def run_event(ev: FreezeEvent) -> dict:
    """1件の凍結事象を計装し、フレーム単位トレースを収集・保存する。"""
    import scripts.visualize_advantage_overlay as ov
    from src.recognition_pipeline import RecognitionPipeline

    video_path = VIDEO_DIR / f"video_{ev.video_id}.mp4"
    if not video_path.exists():
        print(f"[skip] 動画が無い: {video_path}")
        return dict(event=ev.__dict__, ok=False, reason="video_missing")

    start_sec = max(0.0, ev.t_start_abs - PRE_MARGIN_SEC)
    end_sec = ev.t_start_abs + ev.freeze_len_sec + POST_MARGIN_SEC

    pipe_trace: list[dict] = []
    hold_trace: list[dict] = []

    original_pipe_update = RecognitionPipeline.update
    original_tracker_update = ov.ResolvedExchangeTracker.update
    original_reeval = ov.ResolvedExchangeTracker._reevaluate_live_defender
    reeval_trace: list[dict] = []

    def _traced_pipe_update(self, fi, t, frame):  # noqa: ANN001
        r = original_pipe_update(self, fi, t, frame)
        pipe_trace.append(dict(
            t=float(t),
            state1=r.p1.state.name, state2=r.p2.state.name,
            chain1=r.p1.chain_event is not None,
            chain2=r.p2.chain_event is not None,
            score1=int(r.p1.score) if r.p1.score is not None else None,
            score2=int(r.p2.score) if r.p2.score is not None else None,
            confirmed_hash1=_board_hash(r.p1.confirmed_board),
            confirmed_hash2=_board_hash(r.p2.confirmed_board),
            estimated_hash1=_board_hash(getattr(r.p1, "estimated_board", None)),
            estimated_hash2=_board_hash(getattr(r.p2, "estimated_board", None)),
            next_pair1=r.p1.next_pair, next_pair2=r.p2.next_pair,
            tsumo1=self.tsumo_count("1P"), tsumo2=self.tsumo_count("2P"),
        ))
        return r

    def _traced_tracker_update(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        active, just_deactivated = original_tracker_update(self, *args, **kwargs)
        t_sec = kwargs.get("t_sec")
        if t_sec is None and len(args) >= 4:
            t_sec = args[3]
        hold_trace.append(dict(
            t=float(t_sec) if t_sec is not None else float("nan"),
            active=bool(active), just_deactivated=bool(just_deactivated),
            awaiting_landing=bool(getattr(self, "_awaiting_landing", False)),
            hold_adv=float(getattr(self, "hold_adv", float("nan"))),
            hold_p1=float(getattr(self, "hold_p1", float("nan"))),
            defender_side=getattr(self, "hold_defender_side", None),
            incoming_ojama=float(getattr(self, "hold_incoming_ojama", float("nan"))),
            last_live_reeval_t=(
                float(self._last_live_reeval_t)
                if getattr(self, "_last_live_reeval_t", None) is not None else None
            ),
        ))
        return active, just_deactivated

    def _traced_reeval(self, b1, b2, snap=None, state1=None, state2=None):  # noqa: ANN001
        before_hold_adv = getattr(self, "hold_adv", None)
        result = original_reeval(self, b1, b2, snap=snap, state1=state1, state2=state2)
        defender_side, _incoming = self._decisive_defender(self._result) if self._result else (None, 0.0)
        defender_state = (
            (state1.name if defender_side == "1P" and state1 is not None else
             state2.name if defender_side == "2P" and state2 is not None else None)
        )
        reeval_trace.append(dict(
            t=float(self._t_sec), defender_side=defender_side,
            defender_state=defender_state,
            hold_adv_before=before_hold_adv, hold_adv_after=float(self.hold_adv),
            ran=(before_hold_adv != float(self.hold_adv)),
        ))
        return result

    RecognitionPipeline.update = _traced_pipe_update
    ov.ResolvedExchangeTracker.update = _traced_tracker_update
    ov.ResolvedExchangeTracker._reevaluate_live_defender = _traced_reeval
    history: list[tuple[float, float]] = []
    try:
        kwargs = final4_kwargs()
        ov.generate(
            video_path, Path("/tmp/_unused_diag_freeze.mp4"),
            max_sec=0.0, start_sec=start_sec, end_sec=end_sec,
            debug_history_out=history,
            **kwargs,
        )
    finally:
        RecognitionPipeline.update = original_pipe_update
        ov.ResolvedExchangeTracker.update = original_tracker_update
        ov.ResolvedExchangeTracker._reevaluate_live_defender = original_reeval

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"trace_rank{ev.rank}_{ev.video_id}_t{ev.t_start_abs:.0f}.json"
    payload = dict(
        event=dict(
            rank=ev.rank, video_id=ev.video_id, t_start_abs=ev.t_start_abs,
            freeze_len_sec=ev.freeze_len_sec, fixed_value_desc=ev.fixed_value_desc,
            hold_fraction_pct_reported=ev.hold_fraction_pct,
        ),
        window=dict(start_sec=start_sec, end_sec=end_sec),
        disp_adv_trace=[dict(t=t, adv=a) for t, a in history],
        pipe_trace=pipe_trace,
        hold_trace=hold_trace,
        reeval_trace=reeval_trace,
    )
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[event {ev.rank}] {video_path.name} t={ev.t_start_abs:.1f} -> {out_json} "
          f"(adv点={len(history)} pipe点={len(pipe_trace)} hold点={len(hold_trace)})")
    return dict(event=ev.__dict__, ok=True, out_json=str(out_json))


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for ev in FREEZE_EVENTS:
        try:
            results.append(run_event(ev))
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL event {ev.rank}] {ev.video_id} t={ev.t_start_abs}: {e!r}")
            results.append(dict(event=ev.__dict__, ok=False, reason=repr(e)))
    with (OUT_DIR / "run_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[done] {n_ok}/{len(results)} 件成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
