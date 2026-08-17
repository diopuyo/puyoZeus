"""W23 測定方式アーティファクト検証 (2026-08-17)。

持続誤認26件系統1 (c23/c10) の真因は _validate_next_history の ever_seen
飢餓状態だが (docs/KNOWN_WEAKNESSES.md W23)、既知の実測はいずれも
「物差しv2の30秒チャンク再収集 (毎回 MENU から再起動 = ever_seen を強制リセット)」
という測定方式の下で得られたもの。本スクリプトは同じ動画・同じ対象時刻付近を、

    (a) chunk  : 既知の diag スクリプトと同じチャンク境界から pipeline を
                 新規構築 (force_in_match=True) して再現する (= 測定方式そのまま)
    (b) continuous_off: LOOKBACK_SEC 秒前から通し処理 (force_in_match=False、
                 実際の試合境界検知に任せる) し、enable_next_history_starvation_fix
                 は OFF のまま。これが「本番相当の連続視聴」に一番近い構成。
    (c) continuous_on : (b) と同じ通し処理だが修正フラグ ON。

の3構成で実行し、対象時刻付近で _validate_next_history が「飢餓状態
(ever_seen∪next_queue の puyo 色数 < 4)」により観測色を書き換えたイベントを
全セル・両サイドについて記録する。あわせて `_match_active_started_time` を
ログし、対象時刻がその試合の何秒目かを報告する。

本体コード変更なし (モンキーパッチで計装するのみ)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w23_artifact_measure_2026-08-17 --video c10 --mode chunk
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w23_artifact_measure_2026-08-17 --video c10 --mode continuous --starvation-fix off
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.board import COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# 動画別の既知パラメータ (docs/KNOWN_WEAKNESSES.md W23 実測値)。
VIDEO_INFO: dict[str, dict] = {
    "c10": {
        "path": Path.home() / "frames" / "video_c10.mp4",
        "chunk_start_sec": 1337.4,
        "target_sec": 1405.12,
    },
    "c23": {
        "path": Path.home() / "frames" / "video_c23.mp4",
        "chunk_start_sec": 1404.89,
        "target_sec": 1405.05,
    },
}

WINDOW_BEFORE_TARGET_SEC: float = 2.0
WINDOW_AFTER_TARGET_SEC: float = 8.0
CHUNK_END_MARGIN_SEC: float = 30.0  # チャンク再現時の処理終端 (target + margin)

STARVATION_MIN_COLORS: int = 4

# 計装: _validate_next_history をフックし、飢餓状態 (puyo色数 < 4) で
# board が書き換わった全セルを記録する。
_orig_validate = RecognitionPipeline._validate_next_history
_EVENTS: list[dict] = []
_CUR_SIDE: dict = {"t": None}


def _patched_validate(
    board, next_queue, ever_seen=None, frame_bgr=None, region=None,
    enable_starvation_fix: bool = False, min_colors_for_validation: int = 4,
):
    out = _orig_validate(
        board, next_queue, ever_seen=ever_seen, frame_bgr=frame_bgr,
        region=region, enable_starvation_fix=enable_starvation_fix,
        min_colors_for_validation=min_colors_for_validation,
    )
    seen_puyo = set()
    for pair in (next_queue or []):
        for c in pair:
            if c not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                seen_puyo.add(int(c))
    if ever_seen:
        seen_puyo |= {
            c for c in ever_seen
            if c not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN)
        }
    is_starving = len(seen_puyo) < STARVATION_MIN_COLORS
    if not is_starving:
        return out
    # W23 の真の機構 (ステップ1: 履歴外色→強制置換) と、無関係な機構
    # (ステップ2: 浮きぷよ除去による color→EMPTY 消去) を区別する。
    # ステップ1が対象にするのは「board 上の値が seen (=base集合∪next_queue∪
    # ever_seen) に無い」セルのみ (_validate_next_history 本体の
    # `if v in seen: continue` と同じ条件)。この条件を満たさない diff は
    # 浮きぷよ除去等の別機構による副作用であり W23 の対象外として除外する。
    seen_full = {COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN} | seen_puyo
    side = (
        "1P" if region is DEFAULT_P1_REGION or region == DEFAULT_P1_REGION
        else "2P" if region is DEFAULT_P2_REGION or region == DEFAULT_P2_REGION
        else "?"
    )
    for r in range(13):
        for c in range(6):
            b, a = int(board.get(r, c)), int(out.get(r, c))
            if b != a and b not in seen_full:
                _EVENTS.append({
                    "t": _CUR_SIDE["t"], "side": side,
                    "cell": [r, c], "before": b, "after": a,
                    "seen_puyo_colors": sorted(seen_puyo),
                    "starvation_fix": enable_starvation_fix,
                })


RecognitionPipeline._validate_next_history = staticmethod(_patched_validate)


def _build_pipeline(
    *, force_in_match: bool, enable_starvation_fix: bool,
) -> RecognitionPipeline:
    """構成E (本番採用 + R2 + W10ガード + override_color_guard +
    ojama_column_stack_fix) + 本タスクの修正フラグ で pipeline を構築する。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=force_in_match,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_patch_fp_hsv_guard=True,
        enable_chain_tracker=True,
        enable_floating_gap_restore=True,
        enable_landing_color_guard=True,
        enable_override_color_guard=True,
        enable_ojama_column_stack_fix=True,
        enable_next_history_starvation_fix=enable_starvation_fix,
    )


def _run(
    video_path: Path, start_sec: float, end_sec: float,
    *, force_in_match: bool, enable_starvation_fix: bool,
) -> dict:
    _EVENTS.clear()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = _build_pipeline(
        force_in_match=force_in_match, enable_starvation_fix=enable_starvation_fix,
    )
    frame_idx = start_frame
    t_sec = start_sec
    match_started_1p_at_target: float | None = None
    match_started_2p_at_target: float | None = None
    n_frames = 0
    while t_sec < end_sec:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        info = VIDEO_INFO_CUR
        target_sec = info["target_sec"]
        _CUR_SIDE["t"] = t_sec
        pipeline.update(frame_idx, t_sec, frame)
        n_frames += 1
        if abs(t_sec - target_sec) < (0.5 / fps):
            match_started_1p_at_target = pipeline._match_active_started_time
        frame_idx += 1
        t_sec = frame_idx / fps
    cap.release()
    relevant = [
        e for e in _EVENTS
        if abs(e["t"] - VIDEO_INFO_CUR["target_sec"]) <= 3.0
    ]
    # (side, cell) 別サマリ: 最初/最後の発生時刻・件数・代表 before/after。
    summary: dict[tuple, dict] = {}
    for e in _EVENTS:
        key = (e["side"], tuple(e["cell"]))
        s = summary.setdefault(key, {
            "side": e["side"], "cell": e["cell"],
            "first_t": e["t"], "last_t": e["t"], "count": 0,
            "before": e["before"], "after": e["after"],
        })
        s["first_t"] = min(s["first_t"], e["t"])
        s["last_t"] = max(s["last_t"], e["t"])
        s["count"] += 1
    cell_summary = sorted(
        summary.values(), key=lambda s: (s["first_t"], s["side"], s["cell"]),
    )
    return {
        "n_frames_processed": n_frames,
        "n_starvation_events_total": len(_EVENTS),
        "n_starvation_events_near_target": len(relevant),
        "events_near_target": relevant[:50],
        "cell_summary": cell_summary,
        "match_active_started_time_at_target": match_started_1p_at_target,
        "target_sec": VIDEO_INFO_CUR["target_sec"],
    }


VIDEO_INFO_CUR: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", choices=["c10", "c23"], required=True)
    ap.add_argument("--mode", choices=["chunk", "continuous"], required=True)
    ap.add_argument("--starvation-fix", choices=["off", "on"], default="off")
    ap.add_argument("--lookback-sec", type=float, default=300.0)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--target-sec", type=float, default=None,
        help="計測窓の中心秒 (省略時は VIDEO_INFO の既定値)。",
    )
    ap.add_argument(
        "--end-sec", type=float, default=None,
        help="処理終端秒 (省略時は target_sec + WINDOW_AFTER_TARGET_SEC)。",
    )
    args = ap.parse_args()

    global VIDEO_INFO_CUR
    VIDEO_INFO_CUR = dict(VIDEO_INFO[args.video])
    if args.target_sec is not None:
        VIDEO_INFO_CUR["target_sec"] = args.target_sec
    video_path = VIDEO_INFO_CUR["path"]
    target_sec = VIDEO_INFO_CUR["target_sec"]
    enable_fix = args.starvation_fix == "on"

    if args.mode == "chunk":
        start_sec = VIDEO_INFO_CUR["chunk_start_sec"]
        force_in_match = True
    else:
        start_sec = max(0.0, target_sec - args.lookback_sec)
        force_in_match = False
    end_sec = (
        args.end_sec if args.end_sec is not None
        else target_sec + WINDOW_AFTER_TARGET_SEC
    )

    print(
        f"[{args.video}/{args.mode}/fix={args.starvation_fix}] "
        f"start={start_sec:.2f} end={end_sec:.2f} target={target_sec:.2f}"
    )
    result = _run(
        video_path, start_sec, end_sec,
        force_in_match=force_in_match, enable_starvation_fix=enable_fix,
    )
    result["config"] = {
        "video": args.video, "mode": args.mode,
        "starvation_fix": args.starvation_fix,
        "start_sec": start_sec, "end_sec": end_sec,
        "force_in_match": force_in_match,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
