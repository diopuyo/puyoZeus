"""STABLE 確定の3フレーム連続一致窓の「振り出し戻り」頻度を測定する (read-only診断, 2026-08-13)。

## 背景
user決定 (2026-08-13): 盤面確定窓を「3フレーム連続一致」から「3フレーム中2一致の
多数決」に変える改善を、認識99.5%維持を条件に採用確定。優先度・改善見込みの
定量化のため、現状の「1フレームのノイズ混入で連続一致カウンタが1に戻り、確定が
3フレームを超えて延びる」頻度を実測する。

## 対象メカニズム
`src/board_state_machine.py` の `_update_within_current_state` 内、
`pending_count`/`pending_board` による連続多数決 (stable_frame_count、本番値=3。
scripts/_diag_placement_confirm_frames_2026-07-25.py の PRODUCTION_STABLE_FRAME_COUNT
コメント参照: collect_boards_lean.py 等の本番収集は全て 3 を明示)。

このロジックは `confirmed_board is None` の間だけ実際に確定を遅らせる
(confirmed 確立後は pending_count が回っても no-op)。`confirmed_board` が
None に戻るのは (a) 試合開始直後の初回確定前、 (b) `RecognitionPipeline`
の自己修復 `sm.reset(keep_match_state=False)` (baseline_broken reset、
diff>8 が 60 連続フレームで発火) 経由の 2 パターン。
review_demo 動画では後者が t≈303-310s に複数回連続発火することが既知
(logs/tsumo_verify_full60_2026-08-12.log で MENU 遷移が同区間に集中)。

## 計測方法
`BoardStateMachine.update` を計装ラッパーに差し替え (`src/` 側は変更しない、
stateless 検証原則)。各フレームで以下を記録する:
    - 生 CNN 観測 (signals.cnn_board の grid_bytes、ハッシュ用)
    - 遷移前後の pending_count / state / confirmed_board is None

事後処理で:
    1. confirmed_board が None から not-None になった瞬間を confirm イベントとして
       検出し、直前の None 開始フレームから何フレームで確定したかを数える。
    2. その window 内で pending_count が (2以上) から 1 に戻った回数を
       振り出し戻り件数として数える。
    3. 同じ window 内の生 CNN board 系列から、3フレーム中2一致の
       反実仮想アルゴリズムを再実行し、その場合の確定フレーム数を計算する
       (sliding window maxlen=3、直近 3 フレームの中で同一 board が
       2 回以上出現したら多数決確定)。

## 実行条件
- WSL venv、シングルスレッド。read-only 診断。src/ は一切変更しない。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_stable_window_restart_2026-08-13 --video data/frames/review_demo_2026-08-12.mp4 --max-sec 330 --out logs/stable_window_restart_measure_2026-08-13.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0
SIDES = ("1P", "2P")

# 本番の収集・レンダが明示している値 (PRODUCTION_STABLE_FRAME_COUNT、
# scripts/_diag_placement_confirm_frames_2026-07-25.py 参照)。
PRODUCTION_STABLE_FRAME_COUNT: int = 3

# 振り出し戻り判定: pending_count が 2 以上から 1 に戻った場合のみ数える
# (1 フレームだけの新規開始は振り出し戻りに数えない)。
RESET_MIN_PRIOR_COUNT: int = 2

# 反実仮想「3中2」窓のサイズ。
COUNTERFACTUAL_WINDOW: int = 3
COUNTERFACTUAL_MIN_VOTES: int = 2


class _UpdateTap:
    """BoardStateMachine.update を計装するラッパー (検証専用、src/ 不変更)。"""

    def __init__(self, sm: Any, label: str) -> None:
        self._sm = sm
        self._label = label
        self._orig_update = sm.update
        self.records: list[dict[str, Any]] = []

    def wrapped(self, frame_idx: int, signals: Any) -> Any:
        pending_before = self._sm.context.pending_count
        confirmed_before_none = self._sm.context.confirmed_board is None
        cnn_key = signals.cnn_board.grid_bytes()
        ctx = self._orig_update(frame_idx, signals)
        self.records.append({
            "frame_idx": frame_idx,
            "t_sec": signals.time_sec,
            "cnn_key": cnn_key,
            "pending_before": pending_before,
            "pending_after": ctx.pending_count,
            "state_after": ctx.state.value,
            "confirmed_before_none": confirmed_before_none,
            "confirmed_after_none": ctx.confirmed_board is None,
        })
        return ctx


def _counterfactual_majority_confirm_offset(
    cnn_keys: list[bytes], window: int, min_votes: int,
) -> "int | None":
    """window フレーム中 min_votes 一致の反実仮想で確定に要するオフセットを返す.

    cnn_keys[0] が window 開始フレーム (= 実際の確定窓の起点と同じ frame)。
    戻り値は開始から何フレーム目 (1-origin) で確定するか、確定しなければ None。
    """
    buf: list[bytes] = []
    for i, key in enumerate(cnn_keys):
        buf.append(key)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < 2:
            continue
        counts: dict[bytes, int] = {}
        for k in buf:
            counts[k] = counts.get(k, 0) + 1
        if max(counts.values()) >= min_votes:
            return i + 1
    return None


def _analyze_side(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """1 side の record 列から reset イベント・confirm イベントを再構成する."""
    reset_events: list[dict[str, Any]] = []
    confirm_events: list[dict[str, Any]] = []

    is_none = True  # records[0] より前は未確定とみなす
    window_start_idx = 0

    for i, rec in enumerate(records):
        if rec["pending_before"] >= RESET_MIN_PRIOR_COUNT and rec["pending_after"] == 1:
            reset_events.append({
                "frame_idx": rec["frame_idx"],
                "t_sec": rec["t_sec"],
                "pending_before": rec["pending_before"],
                "during_none_window": is_none,
            })

        if is_none and not rec["confirmed_after_none"]:
            # confirm イベント発生
            confirm_idx = i
            actual_frames = confirm_idx - window_start_idx + 1
            window_keys = [
                records[j]["cnn_key"] for j in range(window_start_idx, confirm_idx + 1)
            ]
            cf_offset = _counterfactual_majority_confirm_offset(
                window_keys, COUNTERFACTUAL_WINDOW, COUNTERFACTUAL_MIN_VOTES,
            )
            confirm_events.append({
                "side": label,
                "window_start_frame_idx": records[window_start_idx]["frame_idx"],
                "window_start_t_sec": records[window_start_idx]["t_sec"],
                "confirm_frame_idx": rec["frame_idx"],
                "confirm_t_sec": rec["t_sec"],
                "actual_frames_to_confirm": actual_frames,
                "counterfactual_frames_to_confirm": cf_offset,
                "extended_beyond_baseline": actual_frames > PRODUCTION_STABLE_FRAME_COUNT,
            })
            is_none = False
        elif (not is_none) and rec["confirmed_after_none"]:
            # confirmed が再び None に戻った (baseline_broken reset 等)
            window_start_idx = i + 1
            is_none = True

    # window ごとの reset 件数を突合する (時刻レンジで判定)。
    for ev in confirm_events:
        cnt = 0
        for r in reset_events:
            if (
                ev["window_start_t_sec"] <= r["t_sec"] <= ev["confirm_t_sec"]
                and r["during_none_window"]
            ):
                cnt += 1
        ev["n_resets_in_window"] = cnt

    return {
        "side": label,
        "n_total_reset_events": len(reset_events),
        "n_reset_events_during_none_window": sum(
            1 for r in reset_events if r["during_none_window"]
        ),
        "confirm_events": confirm_events,
    }


def _summarize(all_confirm_events: list[dict[str, Any]]) -> dict[str, Any]:
    """side 横断で分布統計をまとめる."""
    if not all_confirm_events:
        return {"n_events": 0}
    actual = [e["actual_frames_to_confirm"] for e in all_confirm_events]
    cf = [
        e["counterfactual_frames_to_confirm"] for e in all_confirm_events
        if e["counterfactual_frames_to_confirm"] is not None
    ]
    n_extended = sum(1 for e in all_confirm_events if e["extended_beyond_baseline"])

    def _pctl(vals: list[int], p: float) -> float:
        if not vals:
            return float("nan")
        s = sorted(vals)
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    return {
        "n_events": len(all_confirm_events),
        "n_extended_beyond_3frames": n_extended,
        "pct_extended": n_extended / len(all_confirm_events) * 100.0,
        "actual_median": statistics.median(actual),
        "actual_p90": _pctl(actual, 0.9),
        "actual_max": max(actual),
        "actual_mean": statistics.mean(actual),
        "counterfactual_median": statistics.median(cf) if cf else None,
        "counterfactual_p90": _pctl(cf, 0.9) if cf else None,
        "counterfactual_max": max(cf) if cf else None,
        "counterfactual_mean": statistics.mean(cf) if cf else None,
        "n_counterfactual_resolved": len(cf),
    }


def run(video_path: Path, max_sec: float, out_path: Path) -> None:
    """動画を frame-by-frame 処理し、計装データを収集・分析して JSON に書き出す."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[measure] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total_frames, int(max_sec * fps)) if max_sec > 0 else total_frames

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=PRODUCTION_STABLE_FRAME_COUNT,
        load_score_ocr=True,
        enable_chain_tracker=False,
        temporal_smoothing=1,
        load_next_detector=False,
        force_in_match=True,
    )

    tap_1p = _UpdateTap(pipeline._sm_1p, "1P")
    tap_2p = _UpdateTap(pipeline._sm_2p, "2P")
    pipeline._sm_1p.update = tap_1p.wrapped
    pipeline._sm_2p.update = tap_2p.wrapped

    n_processed = 0
    for frame_idx in range(end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        pipeline.update(frame_idx, t_sec, frame)
        n_processed += 1
        if n_processed % 3000 == 0:
            print(f"[measure] processed {n_processed} frames (t={t_sec:.1f}s)", file=sys.stderr)
    cap.release()

    analysis_1p = _analyze_side(tap_1p.records, "1P")
    analysis_2p = _analyze_side(tap_2p.records, "2P")
    all_events = analysis_1p["confirm_events"] + analysis_2p["confirm_events"]
    summary = _summarize(all_events)

    result = {
        "video": str(video_path),
        "fps": fps,
        "n_frames_processed": n_processed,
        "max_sec": max_sec,
        "production_stable_frame_count": PRODUCTION_STABLE_FRAME_COUNT,
        "counterfactual_window": COUNTERFACTUAL_WINDOW,
        "counterfactual_min_votes": COUNTERFACTUAL_MIN_VOTES,
        "per_side": [
            {k: v for k, v in analysis_1p.items() if k != "confirm_events"},
            {k: v for k, v in analysis_2p.items() if k != "confirm_events"},
        ],
        "confirm_events": all_events,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[measure] wrote -> {out_path}")
    print(f"[measure] summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")


def main() -> int:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="STABLE 確定窓の振り出し戻り頻度測定")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-sec", type=float, default=330.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.video, args.max_sec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
