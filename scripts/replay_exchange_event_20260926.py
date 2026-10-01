"""記録された評価入力だけで撃ち合いを再生し、E4互換の出力を作る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.exchange_display_smoothing import SwitchAwareDisplayEMA
from src.exchange_event_evaluator import FileExchangeModels, StaticInput
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_record import read_records, static_key
from src.exchange_event_cli import parse_exchange_event_args


def static_builder(record: Path) -> Any:
    """保存Dを入力キーで再利用する。変更後に新しい盤面組が必要なら再計算する。"""
    from scripts.visualize_advantage_overlay import _exchange_static_input
    cache = {row["key"]: row["input"]["d_features"] for row in read_records(record)
             if row["kind"] == "static"}

    def build(boards: tuple, snapshot: Any, elapsed: float, m0: float) -> StaticInput:
        key = static_key(boards, snapshot)
        if key not in cache:
            cache[key] = _exchange_static_input(boards, snapshot, elapsed, m0).d_features
        return StaticInput(cache[key], m0, elapsed, source_side=0)
    return build


def display_row(overlay: ExchangeEventOverlay, inputs: tuple, context: dict,
                smoothing: Any = None) -> Any:
    """表示値と由来を今回の評価から生成し、旧経路の補助列だけ記録から使う。"""
    from scripts.visualize_advantage_overlay import (
        DisplayTimelineRow, TIMELINE_DUMP_SCORE_NONE_SENTINEL, _exchange_display,
    )
    result = inputs[0]
    adv, probability = _exchange_display(overlay, context["fallback_adv"], context["fallback_p1"],
                                         smoothing, context["t_sec"])
    if overlay.tracker.probability is not None:
        probability = overlay.tracker.probability
    elif isinstance(smoothing, SwitchAwareDisplayEMA):
        probability = context["fallback_p1"]  # 平滑前の確率列の定義 (M3・鮮度) を保つ。
    scores = [TIMELINE_DUMP_SCORE_NONE_SENTINEL if s.score is None else int(s.score)
              for s in (result.p1, result.p2)]
    return DisplayTimelineRow(t_sec=context["t_sec"], game_idx=context["game_idx"],
        state1=result.p1.state.name, state2=result.p2.state.name, display_adv=adv,
        display_p1=probability, adv_raw_last=context["adv_raw_last"], source=overlay.tracker.source,
        resolved_active=context["resolved_active"], settled_ran=context["settled_ran"],
        score1=scores[0], score2=scores[1])


def replay(record: Path, out: Path, model_dir: Path | None = None,
           live_count: bool = False, observer: Any = None, e16: bool = False,
           count_sync: bool = False, death_guard: bool = False,
           evaluation_layers: bool = False, completion_check: bool = False,
           landing_counter_response: bool = False, confirmed_death_hold: bool = False,
           landing_counter_prob: bool = False, landing_hands_spec: bool = False,
           death_candidate_guard: bool = False, death_formula_guard: bool = False,
           multi_landing_death: bool = False, landing_state_safety: bool = False,
           pending_ledger: bool = False, color_score_safety: bool = False,
           completion_recovery: bool = False, midchain_completion: bool = False,
           death_pending_ledger: bool = False, hidden_row_death: bool = False,
           midchain_single_observation: bool = False, prefire_candidates: bool = False,
           prefire_snapshot: bool = False, hidden_row_belief: bool = False,
           prefire_stage_timeout: bool = False, prefire_stage_timeout_only: bool = False,
           prefire_origin_guard: bool = False, post_counter_death_bound: bool = False,
           single_death_proof_guard: bool = False,
           single_death_proof_negative_only: bool = False,
           post_counter_early_exit: bool = False,
           multilanding_node_limit: int | None = None,
           hidden_scenario_cap: int | None = None,
           switch_smoothing: bool = False) -> dict:
    """認識器も動画も開かず、tracker・終了判定・全評価器を新規生成する。"""
    from scripts.visualize_advantage_overlay import (
        _ExchangeEventEndSignals, _ExchangeDisplayEMA, _exchange_display, save_display_timeline,
    )
    from src.exchange_event_m0 import FileM0Predictor
    start = time.perf_counter()
    stream = read_records(record)
    header = next(stream)
    match_gate = None
    if prefire_origin_guard:
        from src.exchange_prefire_origin import recorded_match_gate
        match_gate = recorded_match_gate(header, Path(__file__).resolve().parents[1])
    experimental = count_sync or death_guard or evaluation_layers or completion_check
    default_model = ("models/exchange_event_v4" if e16 or count_sync else
                     "models/exchange_event_v3" if experimental else header["model_dir"])
    directory = model_dir or Path(default_model)
    overlay = ExchangeEventOverlay(FileExchangeModels.load(directory, lightweight=True),
        static_builder(record), _ExchangeEventEndSignals, FileM0Predictor(directory / "M0"),
        per_side_settled=header["per_side_settled"], live_count=live_count, e16=e16,
        count_sync=count_sync, death_guard=death_guard, evaluation_layers=evaluation_layers,
        completion_check=completion_check, landing_counter_response=landing_counter_response,
        confirmed_death_hold=confirmed_death_hold, landing_counter_prob=landing_counter_prob,
        landing_hands_spec=landing_hands_spec, death_candidate_guard=death_candidate_guard,
        death_formula_guard=death_formula_guard, multi_landing_death=multi_landing_death,
        landing_state_safety=landing_state_safety, pending_ledger=pending_ledger,
        color_score_safety=color_score_safety, completion_recovery=completion_recovery,
        midchain_completion=midchain_completion,
        death_pending_ledger=death_pending_ledger, hidden_row_death=hidden_row_death,
        midchain_single_observation=midchain_single_observation, prefire_candidates=prefire_candidates,
        prefire_snapshot=prefire_snapshot, hidden_row_belief=hidden_row_belief,
        prefire_stage_timeout=prefire_stage_timeout, prefire_stage_timeout_only=prefire_stage_timeout_only,
        prefire_origin_guard=prefire_origin_guard, prefire_match_gate=match_gate,
        post_counter_death_bound=post_counter_death_bound,
        single_death_proof_guard=single_death_proof_guard,
        single_death_proof_negative_only=single_death_proof_negative_only,
        post_counter_early_exit=post_counter_early_exit,
        multilanding_node_limit=multilanding_node_limit, hidden_scenario_cap=hidden_scenario_cap)
    rows, frames, inputs = [], 0, None
    smoothing = SwitchAwareDisplayEMA() if switch_smoothing else _ExchangeDisplayEMA()
    fallback_unknown = (None, None) if switch_smoothing else (0.0, 0.5)  # 記録に旧評価器値の無い更新行
    for item in stream:
        if item["kind"] == "update":
            inputs = item["args"]
            if (e16 or death_guard or evaluation_layers or completion_check or confirmed_death_hold) and not getattr(
                    inputs[0], "terminal_evidence_available", False):
                raise ValueError("E16/E17再生には元映像の死亡確認信号を補完した記録が必要")
            overlay.update(*inputs)
            if observer is not None:
                observer(overlay, inputs)
            fallback = ((item["fallback_adv"], item["fallback_p1"])
                        if "fallback_adv" in item else fallback_unknown)
            _exchange_display(overlay, *fallback, smoothing, inputs[3])
            frames += 1
        elif item["kind"] == "display":
            if inputs is None or inputs[3:5] != (item["t_sec"], item["game_idx"]):
                raise ValueError("表示文脈と入力フレームが不一致")
            rows.append(display_row(overlay, inputs, item, smoothing))
        elif item["kind"] == "complete" and item["frames"] != frames:
            raise ValueError("記録フレーム数が不一致")
    save_display_timeline(out / "display.npz", header["video_id"], rows)
    overlay.tracker.save(out / "events.jsonl")
    if overlay._landing_projection.post_counter_bound is not None:
        from scripts.run_e3_exchange_eval_20260926 import save_json
        save_json(out / 'post_counter_bound_audit.json', dict(rows=overlay._landing_projection.post_counter_bound.audit))
    if overlay._origin_guard is not None:
        from scripts.run_e3_exchange_eval_20260926 import save_json
        save_json(out / 'origin_guard_audit.json', overlay._origin_guard.summary())
    if overlay._midchain is not None:
        from scripts.run_e3_exchange_eval_20260926 import save_json
        save_json(out / 'midchain_audit.json', overlay._midchain.summary())
    if overlay._hidden_death is not None:
        from scripts.run_e3_exchange_eval_20260926 import save_json
        save_json(out / 'hidden_death_audit.json', overlay._hidden_death.summary())
    if overlay._prefire is not None:
        from scripts.run_e3_exchange_eval_20260926 import save_json
        save_json(out / 'prefire_audit.json', overlay._prefire.summary())
    status = dict(state="completed", frames=frames, display_frames=len(rows),
                  elapsed_seconds=time.perf_counter() - start, record_bytes=record.stat().st_size,
                  live_count=live_count, model_dir=str(directory))
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def compare(expected: Path, actual: Path) -> dict:
    """NPZの全列・dtype・NaN位置とJSONLの全バイトを厳密照合する。"""
    with np.load(expected / "display.npz") as left, np.load(actual / "display.npz") as right:
        if left.files != right.files:
            raise AssertionError("display.npzの列が不一致")
        for name in left.files:
            if left[name].dtype != right[name].dtype:
                raise AssertionError(f"display.npz dtype: {name}")
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
            if left[name].tobytes() != right[name].tobytes():
                raise AssertionError(f"display.npz bits: {name}")
    for name in ("events.jsonl", "events.diagnostics.json"):
        if name.endswith("jsonl"):
            assert (expected / name).read_bytes() == (actual / name).read_bytes(), name
        else:
            assert json.loads((expected / name).read_text()) == json.loads((actual / name).read_text()), name
    if (expected / "display.npz").read_bytes() != (actual / "display.npz").read_bytes():
        raise AssertionError("display.npzのファイルバイトが不一致")
    return dict(display="byte_identical", events="byte_identical", diagnostics="identical")


def main() -> None:
    """単独再生と同じレンダ出力との等価性検査を提供する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", "--exchange-event-model-dir", type=Path)
    parser.add_argument("--exchange-event-update", action="store_true",
                        help="描画と共通の指定（再生では常に撃ち合い評価を実行）")
    parser.add_argument("--exchange-event-live-count", action="store_true", default=False)
    parser.add_argument("--exchange-event-e16", action="store_true", default=False)
    parser.add_argument("--landing-counter-response", action="store_true", default=False)
    parser.add_argument("--landing-hands-spec", action="store_true", default=False)
    parser.add_argument("--death-candidate-guard", action="store_true", default=False)
    parser.add_argument("--death-formula-guard", action="store_true", default=False)
    parser.add_argument("--multi-landing-death", action="store_true", default=False)
    parser.add_argument("--post-counter-death-bound", action="store_true", default=False)
    parser.add_argument("--post-counter-early-exit", action="store_true", default=False)
    parser.add_argument("--hidden-scenario-cap", type=int, default=None)
    parser.add_argument("--single-death-proof-guard", action="store_true", default=False)
    parser.add_argument("--single-death-proof-negative-only", action="store_true", default=False)
    parser.add_argument("--prefire-origin-guard", action="store_true", default=False)
    parser.add_argument("--landing-state-safety", action="store_true", default=False)
    for name in ("pending-ledger", "color-score-safety", "completion-recovery", "midchain-completion",
                 "death-pending-ledger", "hidden-row-death", "midchain-single-observation", "prefire-candidates", "prefire-snapshot", "hidden-row-belief", "prefire-stage-timeout", "prefire-stage-timeout-only"):
        parser.add_argument("--" + name, action="store_true", default=False)
    parser.add_argument("--confirmed-death-hold", action="store_true", default=False)
    parser.add_argument("--landing-counter-prob", action="store_true", default=False)
    parser.add_argument("--exchange-event-switch-smoothing", action="store_true", default=False)
    for name in ("count-sync", "death-guard", "evaluation-layers", "completion-check"):
        parser.add_argument("--exchange-event-" + name, action="store_true", default=False)
    parser.add_argument("--compare", type=Path)
    options = parse_exchange_event_args(parser)
    result = replay(options.record, options.out, options.model_dir, options.exchange_event_live_count,
                    e16=options.exchange_event_e16, count_sync=options.exchange_event_count_sync,
                    death_guard=options.exchange_event_death_guard,
                    evaluation_layers=options.exchange_event_evaluation_layers,
                    completion_check=options.exchange_event_completion_check,
                    landing_counter_response=options.landing_counter_response,
                    confirmed_death_hold=options.confirmed_death_hold,
                    landing_counter_prob=options.landing_counter_prob,
                    landing_hands_spec=options.landing_hands_spec,
                    death_candidate_guard=options.death_candidate_guard,
                    death_formula_guard=options.death_formula_guard,
                    multi_landing_death=options.multi_landing_death,
                    landing_state_safety=options.landing_state_safety,
                    pending_ledger=options.pending_ledger, color_score_safety=options.color_score_safety,
                    completion_recovery=options.completion_recovery,
                    midchain_completion=options.midchain_completion,
                    death_pending_ledger=options.death_pending_ledger, hidden_row_death=options.hidden_row_death,
                    midchain_single_observation=options.midchain_single_observation,
                    prefire_candidates=options.prefire_candidates, prefire_snapshot=options.prefire_snapshot,
                    hidden_row_belief=options.hidden_row_belief,
                    prefire_stage_timeout=options.prefire_stage_timeout,
                    prefire_stage_timeout_only=options.prefire_stage_timeout_only,
                    prefire_origin_guard=options.prefire_origin_guard,
                    post_counter_death_bound=options.post_counter_death_bound,
                    single_death_proof_guard=options.single_death_proof_guard,
                    single_death_proof_negative_only=options.single_death_proof_negative_only,
                    post_counter_early_exit=options.post_counter_early_exit,
                    hidden_scenario_cap=options.hidden_scenario_cap,
                    switch_smoothing=options.exchange_event_switch_smoothing)
    if options.compare:
        result["equivalence"] = compare(options.compare, options.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
