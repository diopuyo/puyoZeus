"""Gate 3-2b 「生成量が桁違い」事象の根因診断 (2026-08-25)。

## 背景

`scripts/_gate3_episode_probe_2026-08-24.py` の実データ検証
(`data/verify/gate3_episode_2026-08-24/diagnostics.json`) で以下が判明した:

- `total_generated=45036` (300秒・33 chain_id で非現実的)
- `self_check_mismatch_count=30/30` (100%不一致)
- `self_check_max_abs_diff=108961` (点数スケールに見える)

依頼元の仮説: マージンタイムの起点が壊れていて換算レートが 1 になっている。
本スクリプトはこれを **計装で数値確定** する (本番コードは一切変更しない)。

## 使わないもの

`scripts/_gate3_episode_probe_2026-08-24.py` は import も変更もしない。
`src/` 配下は読むだけ。本ファイルのみ新規作成する。

## 構成 (3 部)

Part A (動画処理なし・純粋な算術検査):
    `src/chain_detector.py::VideoChainTracker` の既定 `match_start_sec=0.0` が
    `src/recognition_pipeline.py` から一度も上書きされていない事実を
    (1) ソース走査、(2) 実インスタンス化の両方で確認する。
    その上で Gate 3-0 で既に確定済みの実測トリガー時刻
    (`data/verify/gate3_chainid_2026-08-24/result_w1.json`) を
    「chain_detector が使う elapsed」としてそのまま `compute_effective_rate`
    に通し、実際に rate=1 になるかを見る。

Part B (短い動画スキャン・MatchStateDetector のみ、CNN 不使用):
    t=750 (プローブの warmup 開始点) の前後に「本当に試合境界 (MENU) が
    あったのか」を疎サンプリングで確認する。境界が無ければ、プローブの
    `OjamaAccountingTracker()` フレッシュ construction が
    「実際にはもっと前から続いている試合」を「たった今始まった試合」と
    誤認したことになる (= プローブ固有のアーティファクト)。

Part C (短い再処理・60秒程度、プローブと同一経路):
    プローブと全く同じ構成 (同じ warmup 開始 t=750、同じ FORMULA_FLAGS) で
    t=750〜812 のみを再処理し、
    `OjamaAccountingTracker._match_start_sec` が実際どの値に収束するか、
    window 内の実チェーン (t=808.8, t=809.767) で `_elapsed` と
    `compute_effective_rate` が実際どうなるかを記録する。
    元の 300 秒プローブを再実行しない (コスト制約の遵守)。

出力: data/verify/gate3_rate_diag_2026-08-25/report.json
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

from src.chain_detector import VideoChainTracker  # noqa: E402
from src.match_state import MatchStateDetector  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.scoring import (  # noqa: E402
    MARGIN_TIME_DECAY_FACTOR,
    MARGIN_TIME_DECAY_INTERVAL_SEC,
    MARGIN_TIME_MAX_DECAYS,
    MARGIN_TIME_START_SEC,
    OJAMA_RATE_MIN,
    OJAMA_RATE_STANDARD,
    compute_effective_rate,
)
from scripts.collect_indicators_v2 import (  # noqa: E402
    DEFAULT_FPS,
    TARGET_H,
    TARGET_W,
    _drive_ojama,
    _SideTracker,
    _update_game_idx,
)

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUT_DIR = PROJECT_ROOT / "data/verify/gate3_rate_diag_2026-08-25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gate 3-0 で既に確定済みの w1 実測トリガー時刻
# (data/verify/gate3_chainid_2026-08-24/result_w1.json の t_start 列。
#  この診断では動画を再処理せず、既存の確定値をそのまま使う)。
W1_TRIGGER_SECONDS: list[float] = [
    747.067, 777.367, 808.8, 826.933, 874.367, 893.533, 921.567,
    929.433, 985.033, 1033.3, 1066.333, 1069.767,  # ev1 (1P側)
    774.833, 809.767, 859.867, 1014.967,  # ev2 (2P側)
]

FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)


def _part_a() -> dict:
    """chain_detector.py の match_start_sec がどこにも上書きされない事実の確認。"""
    src_text = (PROJECT_ROOT / "src" / "recognition_pipeline.py").read_text(
        encoding="utf-8",
    )
    ctor_calls = [
        line.strip() for line in src_text.splitlines() if "VideoChainTracker(" in line
    ]
    passes_match_start = any("match_start_sec" in line for line in ctor_calls)

    tracker = VideoChainTracker()  # recognition_pipeline.py と同じ引数無し構築
    default_match_start_sec = tracker._match_start_sec  # noqa: SLF001 (読み取り専用)
    sig_default = inspect.signature(VideoChainTracker.__init__).parameters[
        "match_start_sec"
    ].default

    floor_elapsed_threshold = (
        MARGIN_TIME_START_SEC + MARGIN_TIME_MAX_DECAYS * MARGIN_TIME_DECAY_INTERVAL_SEC
    )

    rows = []
    for t in sorted(W1_TRIGGER_SECONDS):
        # chain_detector.py:295 の式をそのまま再現:
        #   elapsed = max(0.0, self._last_stable_t - self._match_start_sec)
        # ここでの t が self._last_stable_t、default_match_start_sec が
        # self._match_start_sec に相当する (recognition_pipeline.py が
        # 常にこの既定値のまま構築するため)。
        elapsed_used = max(0.0, t - default_match_start_sec)
        rate = compute_effective_rate(elapsed_used, OJAMA_RATE_STANDARD)
        rows.append({
            "trigger_sec": t,
            "elapsed_sec_used_by_chain_detector": elapsed_used,
            "effective_rate": rate,
        })

    return {
        "recognition_pipeline_video_chain_tracker_ctor_calls": ctor_calls,
        "any_call_passes_match_start_sec": passes_match_start,
        "video_chain_tracker_match_start_sec_default_param": sig_default,
        "video_chain_tracker_actual_attr_after_bare_construction": default_match_start_sec,
        "margin_time_floor_elapsed_threshold_sec": floor_elapsed_threshold,
        "rate_base": OJAMA_RATE_STANDARD,
        "rate_floor": OJAMA_RATE_MIN,
        "w1_trigger_rows": rows,
        "all_rows_at_floor_rate": all(r["effective_rate"] == OJAMA_RATE_MIN for r in rows),
    }


def _part_b() -> dict:
    """t=750 前後に本当に試合境界 (MENU) があったかを疎サンプリングで確認する。

    MatchStateDetector のみ使用 (CNN 色分類・チェイン検出等は一切呼ばない)。
    粗いスキャン (0〜700 秒、60 秒刻み) + 密なスキャン (700〜810 秒、5 秒刻み)。
    """
    detector = MatchStateDetector.load_default()
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        return {"error": f"cannot open {VIDEO}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS

    def _sample(t_sec: float) -> dict:
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"t_sec": t_sec, "state": "READ_FAILED", "bg_value": None}
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        r = detector.detect(frame)
        return {"t_sec": t_sec, "state": r.state.value, "bg_value": r.bg_value}

    coarse = [_sample(t) for t in range(0, 701, 60)]
    fine = [_sample(t) for t in [700 + 5 * i for i in range(23)]]  # 700..810
    cap.release()

    all_samples = coarse + fine
    not_in_match = [s for s in all_samples if s.get("state") == "not_in_match"]
    return {
        "coarse_samples_0_to_700_step60": coarse,
        "fine_samples_700_to_810_step5": fine,
        "not_in_match_sample_count": len(not_in_match),
        "not_in_match_samples": not_in_match,
    }


def _part_c() -> dict:
    """プローブと同一経路で t=750〜812 だけ再処理し、実際の match_start_sec を測る。"""
    T0 = 780.0
    SHORT_T1 = 812.0  # 窓内の最初の2連鎖 (808.8, 809.767) を含める
    WARMUP_SEC = 30.0
    start_sec = max(0.0, T0 - WARMUP_SEC)  # =750.0 (プローブと同じ)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        return {"error": f"cannot open {VIDEO}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_frame = int(start_sec * fps)
    end_frame = int(SHORT_T1 * fps)
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
    from src.board_state_machine import BoardState
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU

    match_start_sec_history: list[tuple[float, float]] = []
    last_seen_match_start: float | None = None
    chain_rows: list[dict] = []
    last_chain_key: dict[str, tuple | None] = {"1P": None, "2P": None}

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
        result = pipeline.update(fi, t_sec, frame)
        _drive_ojama(
            ojama_tracker, result.p1, result.p2, prev_state_p1, prev_state_p2, t_sec,
            tracker_p1=tracker_p1, tracker_p2=tracker_p2, pipeline=pipeline,
        )
        _update_game_idx(tracker_p1, result.p1.score)
        _update_game_idx(tracker_p2, result.p2.score)
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state

        cur_match_start = ojama_tracker._match_start_sec  # noqa: SLF001 (読み取り専用)
        if cur_match_start != last_seen_match_start:
            match_start_sec_history.append((t_sec, cur_match_start))
            last_seen_match_start = cur_match_start

        for side_label, side_result in (("1P", result.p1), ("2P", result.p2)):
            ev = side_result.chain_event
            if ev is None:
                continue
            key = (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)
            if key == last_chain_key[side_label]:
                continue
            last_chain_key[side_label] = key
            elapsed_ojama_tracker = ojama_tracker._elapsed(t_sec)  # noqa: SLF001
            rate_ojama_tracker = compute_effective_rate(
                elapsed_ojama_tracker, OJAMA_RATE_STANDARD,
            )
            # chain_detector.py 側の (壊れた) elapsed も同時に記録する。
            elapsed_chain_detector = ev.trigger_sec  # match_start_sec=0.0 前提の式と同値
            rate_chain_detector = compute_effective_rate(
                elapsed_chain_detector, OJAMA_RATE_STANDARD,
            )
            chain_rows.append({
                "side": side_label, "t_sec": t_sec, "mechanism": ev.mechanism,
                "chain_count": ev.chain_count, "total_score": ev.total_score,
                "ojama_sent_authoritative_field": ev.ojama_sent,
                "elapsed_sec_via_ojama_tracker": elapsed_ojama_tracker,
                "rate_via_ojama_tracker": rate_ojama_tracker,
                "elapsed_sec_via_chain_detector_own_clock": elapsed_chain_detector,
                "rate_via_chain_detector_own_clock": rate_chain_detector,
            })
        fi += 1
    cap.release()

    return {
        "window": {"start_sec": start_sec, "end_sec": SHORT_T1, "target_t0": T0},
        "match_start_sec_history": match_start_sec_history,
        "final_match_start_sec": last_seen_match_start,
        "chain_rows_observed": chain_rows,
    }


def main() -> None:
    report: dict = {"video": str(VIDEO)}
    print("[part A] 純粋算術チェック (動画処理なし)...", flush=True)
    report["part_a_static_clock_check"] = _part_a()
    print("[part B] MatchStateDetector 疎サンプリング (CNN不使用)...", flush=True)
    report["part_b_match_boundary_sparse_scan"] = _part_b()
    print("[part C] 短時間再処理 (t=750〜812、~62秒)...", flush=True)
    report["part_c_short_reprocess_match_start_measurement"] = _part_c()

    out_path = OUT_DIR / "report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
