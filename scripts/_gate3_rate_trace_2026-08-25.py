"""Gate 3 実効レート計装プローブ (2026-08-25、生成量115.7%究明用)。

`scripts/_gate3_episode_probe_v4_2026-08-25.py` のコピーに、
マージンタイム逓減の計装を追加したもの (元プローブは無変更)。

## 何を測るか (docs/CLAUDE_SESSION_HANDOFF_2026-08-25.md 節3)

v51 のラウンド t=460〜531 で、台帳の `raw_generation_total`=2,097個が
独立検算値 1,813個 (=126,896点÷レート70) の 115.7% になる原因として、
「検算値の側がマージンタイム逓減 (`compute_effective_rate`: 96秒経過以降
16秒ごとに0.75倍) を無視している」仮説を実測で確定/棄却する。

計装項目 (すべて monkeypatch / 主ループ側の観測。src/ は一切変更しない):
  1. `OjamaAccountingTracker._match_start_sec` の実際の値の時系列
     (460付近か 429付近か)
  2. その区間で `compute_effective_rate(elapsed_sec, 70)` が返した値の時系列
  3. レート70未満で換算された点数の合計 (逆算値 約57,470点=全体の45% との比較)
  4. 実際のレートで検算しなおした値 Σ(各連鎖の点数÷その時点の実効レート) が
     2,097 に近づくか

## 制約

- 変更してよいのは本ファイルのみ。src/ 配下・既存プローブは無変更。
- 既存の data/verify 配下の出力を上書きしない (出力先は新規ディレクトリ)。

使い方 (2,097 を出した v51_t533 と同一条件で再現する):
  python scripts/_gate3_rate_trace_2026-08-25.py \\
      --video data/frames/video_51.mp4 --t0 459.0 --t1 533.0 \\
      --out-dir data/verify/gate3_rate_trace_2026-08-25
出力: <out-dir>/rate_trace.json + <out-dir>/summary.md
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
from src.chain_id_resolver import ChainIdResolver, ChainObservation  # noqa: E402
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    ExchangeEpisodeTracker,
    GenerationObservation,
    PendingUncappedFrame,
    classify_pending_uncapped_delta,
)
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.scoring import (  # noqa: E402
    OJAMA_RATE_STANDARD,
    compute_effective_rate,
)

# 会計側の score_to_ojama 呼び出しを計装するため、モジュール名前空間ごと import
# (ojama_accounting は `from src.scoring import score_to_ojama` 済みなので、
#  そちらの名前を差し替える)。
import src.ojama_accounting as _oa_mod  # noqa: E402

# 状態機械のウォームアップ (v4 と同じ 30 秒)。
WARMUP_SEC: float = 30.0

# 掛け算式の段を正しく取るために必須の構成 (v4 と同じ)。
FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

_SIDE_LABELS: tuple[str, str] = ("1P", "2P")

# ---- 引き継ぎ文書 (節3) の参照値。分析セクションでの突合に使う。----
REF_LEDGER_RAW_GEN: int = 2097          # 台帳の raw_generation_total (t0=459/t1=533)
REF_INDEP_CHECK_OJAMA: int = 1813       # 独立検算値 = 126,896 ÷ 70
REF_INDEP_SCORE_TOTAL: int = 126_896    # 55,269 + 71,627
REF_HYPOTHESIS_LT70_SCORE: int = 57_470  # 「45%がレート52換算なら一致」の逆算値
# 引き継ぎ文書の分析窓 (実画面で確認済みのラウンド境界)。
ANALYSIS_T0: float = 460.0
ANALYSIS_T1: float = 531.0

# ---- 計装の共有状態 (monkeypatch した score_to_ojama から参照する) ----
_RATE_TRACE: dict = {
    "current_t_sec": None,          # 主ループが毎フレーム更新する動画時刻
    "score_to_ojama_calls": [],     # 会計側の換算呼び出しの生ログ
    "match_start_transitions": [],  # _match_start_sec の変化点
    "rate_timeline": [],            # 実効レートの時系列 (1秒粒度+変化点)
}

_ORIG_SCORE_TO_OJAMA = _oa_mod.score_to_ojama


def _traced_score_to_ojama(
    score: int,
    prev_leftover: int = 0,
    elapsed_sec: float = 0.0,
    rate_base: int = OJAMA_RATE_STANDARD,
):
    """`score_to_ojama` の計装ラッパ (会計ロジックは原本をそのまま呼ぶ)。"""
    res = _ORIG_SCORE_TO_OJAMA(
        score=score, prev_leftover=prev_leftover,
        elapsed_sec=elapsed_sec, rate_base=rate_base,
    )
    _RATE_TRACE["score_to_ojama_calls"].append({
        "t_sec": _RATE_TRACE["current_t_sec"],
        "elapsed_sec": round(float(elapsed_sec), 3),
        "score": int(score),
        "prev_leftover": int(prev_leftover),
        "rate_base": int(rate_base),
        "effective_rate": int(res.effective_rate),
        "ojama_count": int(res.ojama_count),
        "leftover_after": int(res.leftover_score),
    })
    return res


# 会計モジュール側の名前だけを差し替える (他モジュールの呼び出しは対象外)。
_oa_mod.score_to_ojama = _traced_score_to_ojama


# ---- 追加計装 (2026-08-25 コーディネーター指示): ChainIdResolver.push の
#      全観測と、処理直後の in-flight 状態 (chain_id / step_count /
#      provisional_score) を時刻つきで全件記録する。候補b (過剰分割による
#      同一値の二重計上 = 新chainが前chainの確定値を累積包含して開くか) を
#      直接観測するため。src/ は無変更 (クラス属性の差し替えのみ)。----
_STEP_TRACE: list[dict] = []
_ORIG_RESOLVER_PUSH = ChainIdResolver.push


def _traced_resolver_push(self: ChainIdResolver, obs: ChainObservation) -> None:
    """`ChainIdResolver.push` の計装ラッパ (処理は原本をそのまま呼ぶ)。"""
    _ORIG_RESOLVER_PUSH(self, obs)
    st = self._in_flight.get(obs.side)  # noqa: SLF001 (読み取り専用)
    _STEP_TRACE.append({
        "t_sec": round(float(obs.t_sec), 3),
        "side": obs.side,
        "kind": obs.kind.name,
        "chain_count": int(obs.chain_count),
        "total_score": int(obs.total_score),
        "mechanism": obs.mechanism,
        "inflight_after": None if st is None else {
            "chain_id": st.chain_id,
            "state": st.state.name,
            "step_count": st.step_count,
            "provisional_score": st.provisional_score,
            # P1-2 修正後のみ存在する控除基準 (修正前のランでは None)。
            "score_base": getattr(st, "score_base", None),
        },
    })


ChainIdResolver.push = _traced_resolver_push


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=str,
        default=str(PROJECT_ROOT / "data/frames/video_51.mp4"),
    )
    # 2,097 を出した既存測定 (gate3_episode_v4/v51_t533) と同一条件を既定にする。
    parser.add_argument("--t0", type=float, default=459.0)
    parser.add_argument("--t1", type=float, default=533.0)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PROJECT_ROOT / "data/verify/gate3_rate_trace_2026-08-25"),
    )
    # 出力ファイル名の接頭辞。既存の rate_trace.json / summary.md を
    # 上書きしないため、再実行時は別接頭辞 (例: step_trace) を指定する。
    parser.add_argument("--out-prefix", type=str, default="rate_trace")
    return parser.parse_args()


def _chain_event_key(ev: ChainEvent | None) -> tuple | None:
    """ChainEvent の同一性キー (変化検知用。v4 と同じ4値)。"""
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


def _to_chain_observation(
    side_label: str, ev: ChainEvent, t_sec: float, game_idx: int, elapsed_sec: float,
) -> ChainEventObservation:
    """ChainEvent から tracker が要求する最小情報を抜き出す (v4 と同じ)。"""
    return ChainEventObservation(
        side=side_label, t_sec=t_sec, mechanism=ev.mechanism or "",
        chain_count=ev.chain_count, total_score=ev.total_score,
        ojama_sent=ev.ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
    )


def _diff_generation(prev_snap, snap) -> dict[str, int]:
    """前フレームとの `total_generated_by_p1/p2` 差分 (v4 から流用)。"""
    return {
        "1P": snap.total_generated_by_p1 - prev_snap.total_generated_by_p1,
        "2P": snap.total_generated_by_p2 - prev_snap.total_generated_by_p2,
    }


def _make_pending_frame(
    t_sec: float, game_idx: int, snap,
    p1_tsumo: bool, p2_tsumo: bool, gen_diff: dict[str, int] | None,
) -> PendingUncappedFrame:
    """1 フレーム分の `PendingUncappedFrame` を組み立てる (v4 と同じ)。"""
    p1_fin = gen_diff is not None and gen_diff["1P"] != 0
    p2_fin = gen_diff is not None and gen_diff["2P"] != 0
    return PendingUncappedFrame(
        t_sec=t_sec, game_idx=game_idx,
        p1_uncapped=float(snap.pending_p1_uncapped),
        p2_uncapped=float(snap.pending_p2_uncapped),
        p1_tsumo_placed=p1_tsumo, p2_tsumo_placed=p2_tsumo,
        p1_chain_finalized=p1_fin, p2_chain_finalized=p2_fin,
    )


def _process_frame_chain_events(
    result, t_sec: float, t0: float, game_idx: int, elapsed_sec: float,
    episode_tracker: ExchangeEpisodeTracker, last_chain_key: dict[str, tuple | None],
) -> int:
    """1 フレーム分の ChainEvent 変化を検知して観測を供給する (v4 から流用)。"""
    n = 0
    for side_label, side_result in (("1P", result.p1), ("2P", result.p2)):
        ev = side_result.chain_event
        key = _chain_event_key(ev)
        changed = key != last_chain_key[side_label]
        last_chain_key[side_label] = key
        if changed and t_sec >= t0 and ev is not None:
            episode_tracker.observe(
                _to_chain_observation(side_label, ev, t_sec, game_idx, elapsed_sec),
            )
            n += 1
    return n


def _record_rate_instrumentation(
    t_sec: float, ojama_tracker: OjamaAccountingTracker,
    elapsed_sec: float, last_seen: dict,
) -> None:
    """計装項目1・2: _match_start_sec の変化点と実効レートの時系列を記録する。"""
    ms = ojama_tracker._match_start_sec  # noqa: SLF001 (計装のための読み取り専用)
    if ms != last_seen.get("match_start", "__unset__"):
        _RATE_TRACE["match_start_transitions"].append({
            "t_sec": round(t_sec, 3),
            "match_start_sec": None if ms is None else round(float(ms), 3),
        })
        last_seen["match_start"] = ms
    rate_now = compute_effective_rate(elapsed_sec, OJAMA_RATE_STANDARD)
    sec_bucket = int(t_sec)
    if (
        rate_now != last_seen.get("rate")
        or sec_bucket != last_seen.get("sec_bucket")
    ):
        _RATE_TRACE["rate_timeline"].append({
            "t_sec": round(t_sec, 3),
            "match_start_sec": None if ms is None else round(float(ms), 3),
            "elapsed_sec": round(float(elapsed_sec), 3),
            "effective_rate": int(rate_now),
        })
        last_seen["rate"] = rate_now
        last_seen["sec_bucket"] = sec_bucket


def _analyze_calls(calls: list[dict], w0: float, w1: float) -> dict:
    """計装項目3・4: 窓 [w0, w1] 内の換算呼び出しを集計する。

    カウンタは必ず母数と並べる (`x/n` 形式で JSON にも残す)。
    """
    in_win = [
        c for c in calls
        if c["t_sec"] is not None and w0 <= c["t_sec"] <= w1
    ]
    n_all = len(calls)
    n_win = len(in_win)
    lt70 = [c for c in in_win if c["effective_rate"] < OJAMA_RATE_STANDARD]
    score_total = sum(c["score"] for c in in_win)
    score_lt70 = sum(c["score"] for c in lt70)
    gen_total = sum(c["ojama_count"] for c in in_win)
    # 検算 (実レート・floorなしの理想値): Σ(score_i ÷ その時点の実効レート)
    recheck_actual_rate = sum(
        c["score"] / c["effective_rate"] for c in in_win if c["effective_rate"] > 0
    )
    # 参考: 一定レート70での検算 (独立検算と同じ仮定)
    recheck_flat70 = score_total / OJAMA_RATE_STANDARD
    return {
        "window": {"w0": w0, "w1": w1},
        "n_calls_in_window / n_calls_all": f"{n_win}/{n_all}",
        "n_calls_rate_lt70 / n_calls_in_window": f"{len(lt70)}/{n_win}",
        "score_total_in_window": score_total,
        "score_converted_at_rate_lt70": score_lt70,
        "score_lt70_ratio": (score_lt70 / score_total) if score_total else None,
        "gen_total_actual_floor_carry": gen_total,
        "recheck_sum_score_div_actual_rate": round(recheck_actual_rate, 1),
        "recheck_sum_score_div_flat70": round(recheck_flat70, 1),
        "calls_in_window": in_win,
    }


def _dump_all_resolved_chains(episode_tracker: ExchangeEpisodeTracker) -> list[dict]:
    """全 resolved chain の生値ダンプ (v4 プローブから流用)。"""
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    return [
        {
            "chain_id": rc.chain_id, "side": rc.side,
            "opened_at_sec": rc.opened_at_sec, "closed_at_sec": rc.closed_at_sec,
            "step_count": rc.step_count, "provisional_score": rc.provisional_score,
            "finalized_score": rc.finalized_score, "was_finalized": rc.was_finalized,
            "force_cut": rc.force_cut, "close_reason": rc.close_reason.name,
            "growth_observed": rc.growth_observed,
            "finalized_source": rc.finalized_source,
        }
        for rc in resolved
    ]


def _run_probe(video: Path, t0: float, t1: float, out_dir: Path,  # noqa: PLR0915
               out_prefix: str = "rate_trace") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    n_observed_frames = 0
    n_chain_observations = 0
    n_settlement_observations = 0
    n_generation_observations = 0
    last_seen_instr: dict = {}

    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        # 計装: monkeypatch した score_to_ojama が動画時刻を知るための共有値。
        _RATE_TRACE["current_t_sec"] = round(t_sec, 3)
        if fi % 1800 == 0:
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
        _record_rate_instrumentation(t_sec, ojama_tracker, elapsed_sec, last_seen_instr)
        gen_diff = _diff_generation(prev_snap, snap) if prev_snap is not None else None

        n_chain_observations += _process_frame_chain_events(
            result, t_sec, t0, game_idx, elapsed_sec, episode_tracker, last_chain_key,
        )

        if gen_diff is not None and t_sec >= t0:
            for side_label in _SIDE_LABELS:
                delta = gen_diff[side_label]
                if delta == 0:
                    continue
                episode_tracker.observe_generation(GenerationObservation(
                    side=side_label, t_sec=t_sec, game_idx=game_idx,
                    generated_delta=delta,
                ))
                n_generation_observations += 1

        tsumo_1p = pipeline.tsumo_count("1P")
        tsumo_2p = pipeline.tsumo_count("2P")
        p1_tsumo_placed = prev_tsumo["1P"] is not None and tsumo_1p > prev_tsumo["1P"]
        p2_tsumo_placed = prev_tsumo["2P"] is not None and tsumo_2p > prev_tsumo["2P"]
        curr_pending_frame = _make_pending_frame(
            t_sec, game_idx, snap, p1_tsumo_placed, p2_tsumo_placed, gen_diff,
        )
        if prev_pending_frame is not None and t_sec >= t0:
            classification = classify_pending_uncapped_delta(
                prev_pending_frame, curr_pending_frame,
            )
            if classification.settlement is not None:
                episode_tracker.observe_settlement(classification.settlement)
                n_settlement_observations += 1
            for side in classification.wiped_sides:
                episode_tracker.observe_wipe(side, t_sec, game_idx)
        prev_pending_frame = curr_pending_frame
        prev_tsumo["1P"], prev_tsumo["2P"] = tsumo_1p, tsumo_2p
        prev_snap = snap
        if t_sec >= t0:
            n_observed_frames += 1
        fi += 1

    cap.release()
    episode_tracker.finish()
    diag = episode_tracker.diagnostics()

    calls = _RATE_TRACE["score_to_ojama_calls"]
    analysis_gate = _analyze_calls(calls, t0, t1)          # 台帳 2,097 と同じ窓
    analysis_round = _analyze_calls(calls, ANALYSIS_T0, ANALYSIS_T1)  # 実ラウンド窓

    report = {
        "video": str(video),
        "window": {"t0": t0, "t1": t1, "warmup_sec": WARMUP_SEC},
        "references": {
            "ledger_raw_generation_total_prev_run": REF_LEDGER_RAW_GEN,
            "independent_check_ojama_flat70": REF_INDEP_CHECK_OJAMA,
            "independent_check_score_total": REF_INDEP_SCORE_TOTAL,
            "hypothesis_score_at_lt70": REF_HYPOTHESIS_LT70_SCORE,
        },
        "n_observed_frames": n_observed_frames,
        "n_chain_observations": n_chain_observations,
        "n_settlement_observations": n_settlement_observations,
        "n_generation_observations": n_generation_observations,
        # 再現確認: 本ランの台帳生値 (前回 2,097 と一致するか)
        "d7_self_cancel": {
            "raw_generation_total": diag.d7.raw_generation_total,
            "self_canceled_total": diag.d7.self_canceled_total,
        },
        # 計装項目1: _match_start_sec の変化点 (460付近か 429付近か)
        "instr1_match_start_transitions": _RATE_TRACE["match_start_transitions"],
        # 計装項目2: 実効レートの時系列 (1秒粒度+変化点)
        "instr2_rate_timeline": _RATE_TRACE["rate_timeline"],
        # 計装項目3・4: 換算呼び出しの集計 (母数つきカウンタ)
        "instr34_analysis_gate_window": analysis_gate,
        "instr34_analysis_round_window": analysis_round,
        # 全呼び出しの生ログ (窓外含む。準備区間の挙動確認用)
        "score_to_ojama_calls_all": calls,
        "diagnostics_d7_raw": dataclasses.asdict(diag.d7),
        # 追加計装 (候補b 判別用): resolver への全観測 + 直後の in-flight 状態
        "resolver_push_trace": _STEP_TRACE,
        # 本ランの resolved chain 一覧 (chain_id と push trace の突合用)
        "all_chains_dump": _dump_all_resolved_chains(episode_tracker),
        # P1-2 検収用: resolver の診断カウンタ (continuation_reopen_count 等)。
        "resolver_stats": dataclasses.asdict(
            episode_tracker._resolver.stats()  # noqa: SLF001
        ),
    }
    out_path = out_dir / f"{out_prefix}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[saved] {out_path}", flush=True)
    _write_summary(out_dir, report, out_prefix)


def _write_summary(out_dir: Path, report: dict, out_prefix: str = "rate_trace") -> None:
    """短い Markdown 要約 (数値は JSON からの転記のみ、解釈は最小限)。"""
    a_gate = report["instr34_analysis_gate_window"]
    a_round = report["instr34_analysis_round_window"]
    ms_lines = "\n".join(
        f"- t={tr['t_sec']}: match_start_sec={tr['match_start_sec']}"
        for tr in report["instr1_match_start_transitions"]
    )
    rate_changes = [
        r for i, r in enumerate(report["instr2_rate_timeline"])
        if i == 0 or r["effective_rate"] != report["instr2_rate_timeline"][i - 1]["effective_rate"]
    ]
    rate_lines = "\n".join(
        f"- t={r['t_sec']} (elapsed={r['elapsed_sec']}): rate={r['effective_rate']}"
        for r in rate_changes
    )
    lines = [
        "# Gate3 実効レート計装 (v51、生成量115.7%究明)",
        "",
        f"- video: {report['video']}",
        f"- 窓: t0={report['window']['t0']} t1={report['window']['t1']}"
        f" (warmup {report['window']['warmup_sec']}s)",
        f"- 本ランの台帳 raw_generation_total: "
        f"{report['d7_self_cancel']['raw_generation_total']}"
        f" (前回参照値 {report['references']['ledger_raw_generation_total_prev_run']})",
        "",
        "## 計装1: _match_start_sec の変化点",
        ms_lines or "- (変化なし=1件も記録されず。母数0)",
        "",
        "## 計装2: 実効レートの変化点",
        rate_lines or "- (記録なし)",
        "",
        "## 計装3・4: 換算呼び出しの集計",
        "",
        "### ゲート窓 (台帳2,097と同一窓)",
        f"- 呼び出し: {a_gate['n_calls_in_window / n_calls_all']}"
        f" / うちレート70未満: {a_gate['n_calls_rate_lt70 / n_calls_in_window']}",
        f"- 点数合計: {a_gate['score_total_in_window']}"
        f" (独立検算の分子 {report['references']['independent_check_score_total']})",
        f"- レート70未満で換算された点数: {a_gate['score_converted_at_rate_lt70']}"
        f" (仮説の逆算値 {report['references']['hypothesis_score_at_lt70']})",
        f"- 実際の生成 (floor+繰越): {a_gate['gen_total_actual_floor_carry']}",
        f"- 検算(実レート): {a_gate['recheck_sum_score_div_actual_rate']}"
        f" / 検算(一定70): {a_gate['recheck_sum_score_div_flat70']}",
        "",
        "### ラウンド窓 (t=460〜531)",
        f"- 呼び出し: {a_round['n_calls_in_window / n_calls_all']}"
        f" / うちレート70未満: {a_round['n_calls_rate_lt70 / n_calls_in_window']}",
        f"- 点数合計: {a_round['score_total_in_window']}",
        f"- レート70未満で換算された点数: {a_round['score_converted_at_rate_lt70']}",
        f"- 実際の生成 (floor+繰越): {a_round['gen_total_actual_floor_carry']}",
        f"- 検算(実レート): {a_round['recheck_sum_score_div_actual_rate']}"
        f" / 検算(一定70): {a_round['recheck_sum_score_div_flat70']}",
        "",
    ]
    out_md = out_dir / (
        "summary.md" if out_prefix == "rate_trace" else f"{out_prefix}_summary.md"
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_md}", flush=True)


if __name__ == "__main__":
    args = _parse_args()
    _run_probe(
        Path(args.video), args.t0, args.t1, Path(args.out_dir),
        out_prefix=args.out_prefix,
    )
