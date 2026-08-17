"""境界マルチシグナル (b-1)/(b-2) 実装検証 (2026-08-18)。

docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §3 で実装した
enable_match_end_persist_override / enable_post_match_lockdown_latch の
実写検証。video_c21 の f57548 (t≈959.133、ばたんきゅーパネル) 窓を対象に:

1. (b-1) 検証: enable_match_end_persist_override=True で is_match_active が
   t≈959.7 (persist 1.0秒後) 付近で False になることを計装確認する。
2. (b-2) 実測: ばたんきゅーパネル〜次試合の raw_active 持続確認までの
   実秒数を計測し、POST_MATCH_LOCKDOWN_MAX_SEC の実測根拠を作る。
   長い窓 (アンカーから最大 EXTENDED_WINDOW_SEC 秒) を処理し、
   raw_active が CHAIN_BAN_SEC_AFTER_MATCH_START 秒以上連続する最初の
   時刻を「次試合開始」の代理指標として記録する。

本体コード (src/) は変更しない。RecognitionPipeline を外部から直接
呼び計装するだけ。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._verify_boundary_multisignal_b1b2_2026-08-18
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_DIR: Path = Path.home() / "frames"
VIDEO: str = "c21"
ANCHOR_T_SEC: float = 959.133
GAME_START_SEC: float = 858.0
WARMUP_MARGIN_SEC: float = 2.0
# (b-1) 検証窓: 従来の Step0 診断と同じ範囲。
B1_WINDOW_END_SEC: float = 972.0
# (b-2) 実測窓: 次試合開始が確認できるまで長めに処理する。
EXTENDED_WINDOW_SEC: float = 120.0

# メインツリー絶対パスへ出力 (worktree は使い捨てのため)。
OUT_DIR: Path = Path(
    "/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
    "/data/verify/boundary_impl_verify_2026-08-18"
)


def build_pipeline(
    *, match_end_persist_override: bool, post_match_lockdown_latch: bool,
) -> RecognitionPipeline:
    """Step0 診断と同一の構成F相当 + 新フラグ。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=True,  # collect_boards_lean.py 本番実態と一致
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_patch_fp_hsv_guard=True,
        enable_floating_gap_restore=True,
        enable_landing_color_guard=True,
        enable_override_color_guard=True,
        enable_ojama_column_stack_fix=True,
        enable_next_history_starvation_fix=True,
        enable_chain_tracker=True,
        enable_match_end_persist_override=match_end_persist_override,
        enable_post_match_lockdown_latch=post_match_lockdown_latch,
    )


@dataclass
class Row:
    frame_idx: int
    t_sec: float
    dt_from_anchor: float
    is_match_active: bool
    match_end_locked: bool
    post_match_lockdown_active: "bool | None"


def run_window(
    cap: cv2.VideoCapture, fps: float, end_sec: float,
    *, match_end_persist_override: bool, post_match_lockdown_latch: bool,
) -> "list[Row]":
    pipeline = build_pipeline(
        match_end_persist_override=match_end_persist_override,
        post_match_lockdown_latch=post_match_lockdown_latch,
    )
    start_sec = max(0.0, GAME_START_SEC - WARMUP_MARGIN_SEC)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: "list[Row]" = []
    frame_idx = start_frame
    end_frame = int(end_sec * fps)
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = pipeline.update(frame_idx, t_sec, frame)
        if t_sec >= ANCHOR_T_SEC - 5.0:
            rows.append(Row(
                frame_idx=frame_idx, t_sec=round(t_sec, 4),
                dt_from_anchor=round(t_sec - ANCHOR_T_SEC, 4),
                is_match_active=bool(res.is_match_active),
                match_end_locked=bool(res.match_end_locked),
                post_match_lockdown_active=getattr(
                    pipeline, "_post_match_lockdown_active", None,
                ),
            ))
        frame_idx += 1
    return rows


def first_false_after_anchor(rows: "list[Row]") -> "Row | None":
    for r in rows:
        if r.dt_from_anchor >= 0.0 and not r.is_match_active:
            return r
    return None


def first_lockdown_release_row(rows: "list[Row]") -> "Row | None":
    """post_match_lockdown_active が False に落ちた最初の行 (= ラッチ解除)。"""
    consec = 0
    prev_active: "bool | None" = None
    for r in rows:
        if r.post_match_lockdown_active is False and prev_active is True:
            return r
        prev_active = r.post_match_lockdown_active
    return None


def main() -> None:
    video_path = VIDEO_DIR / f"video_{VIDEO}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{VIDEO}] fps={fps:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- (b-1) 検証: flag ON で is_match_active が persist 後に False へ ---
    print("[1/2] (b-1) match_end persist override 検証...")
    rows_b1 = run_window(
        cap, fps, B1_WINDOW_END_SEC,
        match_end_persist_override=True, post_match_lockdown_latch=False,
    )
    ff = first_false_after_anchor(rows_b1)
    (OUT_DIR / "b1_persist_override_timeline.json").write_text(
        json.dumps([asdict(r) for r in rows_b1], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    b1_summary = {
        "video": VIDEO, "anchor_t_sec": ANCHOR_T_SEC,
        "first_is_active_false_after_anchor": asdict(ff) if ff else None,
        "expected_dt_around": RecognitionPipeline.MATCH_END_PERSIST_OVERRIDE_SEC,
    }
    print(json.dumps(b1_summary, ensure_ascii=False, indent=2))

    # --- (b-2) 実測: ラッチ解除までの実秒数 ---
    print("[2/2] (b-2) post-match lockdown 実測 (最大 120秒 窓)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows_b2 = run_window(
        cap, fps, ANCHOR_T_SEC + EXTENDED_WINDOW_SEC,
        match_end_persist_override=False, post_match_lockdown_latch=True,
    )
    (OUT_DIR / "b2_lockdown_timeline.json").write_text(
        json.dumps([asdict(r) for r in rows_b2], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    release_row = first_lockdown_release_row(rows_b2)
    b2_summary = {
        "video": VIDEO, "anchor_t_sec": ANCHOR_T_SEC,
        "lockdown_released_row": asdict(release_row) if release_row else None,
        "measured_duration_sec": (
            round(release_row.t_sec - ANCHOR_T_SEC, 3) if release_row else None
        ),
        "note": (
            "None の場合は EXTENDED_WINDOW_SEC 内にラッチが解除されなかった"
            " (安全弁 POST_MATCH_LOCKDOWN_MAX_SEC が先に効いた可能性、"
            "またはこの動画区間内に次試合が来ない)。"
        ),
    }
    print(json.dumps(b2_summary, ensure_ascii=False, indent=2))
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"b1": b1_summary, "b2": b2_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cap.release()


if __name__ == "__main__":
    main()
