"""is_match_active 再点火経路の犯人特定 (2026-08-18、アーキ発注、診断のみ)。

c18 (022/023アンカー) / c20 (024/025アンカー) の再点火瞬間 (アンカー-0.3秒
付近) で、effective_hard_off を構成する全分岐をダンプし、どれが引き金かを
1つに確定する。

本体コード (src/) は変更しない。RecognitionPipeline を外部から直接
呼び計装するだけ (private attribute の読み出しのみ、ロジック再実装ではない)。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._diag_reignition_culprit_2026-08-18
"""
from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.match_state import MatchState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_DIR: Path = Path.home() / "frames"
OUT_DIR: Path = Path(
    "/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
    "/data/verify/boundary_impl_verify_2026-08-18"
)
WARMUP_MARGIN_SEC: float = 2.0
# 再点火 (アンカー-0.3秒付近) を捉えるための密ログ窓。
DENSE_WINDOW_BEFORE_SEC: float = 1.5
DENSE_WINDOW_AFTER_SEC: float = 0.3


@dataclass(frozen=True)
class AnchorCase:
    sheet: str
    video: str
    anchor_t_sec: float
    game_start_sec: float


CASES: "list[AnchorCase]" = [
    AnchorCase("022_023_c18", "c18", 1891.733, 1850.0),
    AnchorCase("024_025_c20", "c20", 830.6, 782.0),
]


def build_pipeline() -> RecognitionPipeline:
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=True,
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
        enable_match_end_persist_override=True,
        enable_post_match_lockdown_latch=True,
    )


@dataclass
class BranchRow:
    video: str
    frame_idx: int
    t_sec: float
    dt_from_anchor: float
    is_match_active: bool
    score_zero_both: bool
    match_end_locked: bool
    post_match_lockdown_active: bool
    match_end_persisted: bool
    chain_in_progress: bool
    chain_in_progress_suppresses: bool
    score_actively_moving: bool
    hard_match_off: bool
    effective_hard_off: bool
    raw_active: bool
    recent_active: bool
    sm_active: bool
    sm1_state: str
    sm2_state: str
    board_shows_real_gameplay: bool


def run_case(case: AnchorCase) -> "list[BranchRow]":
    video_path = VIDEO_DIR / f"video_{case.video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = max(0.0, case.game_start_sec - WARMUP_MARGIN_SEC)
    end_sec = case.anchor_t_sec + DENSE_WINDOW_AFTER_SEC
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = build_pipeline()
    frame_idx = start_frame
    dense_start_sec = case.anchor_t_sec - DENSE_WINDOW_BEFORE_SEC
    rows: "list[BranchRow]" = []
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = pipeline.update(frame_idx, t_sec, frame)
        if t_sec >= dense_start_sec:
            # --- private attribute の読み出しのみ (ロジック再実装ではない) ---
            score_zero_both = False
            sz = getattr(pipeline, "_score_zero_detector", None)
            if sz is not None:
                try:
                    score_zero_both = bool(sz.detect(frame).both_zero)
                except Exception:
                    pass
            match_end_locked = bool(res.match_end_locked)
            post_match_lockdown_active = bool(
                getattr(pipeline, "_post_match_lockdown_active", False),
            )
            match_end_locked_since = getattr(
                pipeline, "_match_end_locked_since", -1.0,
            )
            enable_persist = bool(
                getattr(pipeline, "_enable_match_end_persist_override", False),
            )
            match_end_persisted = bool(
                enable_persist and match_end_locked
                and match_end_locked_since >= 0.0
                and (t_sec - match_end_locked_since)
                >= RecognitionPipeline.MATCH_END_PERSIST_OVERRIDE_SEC
            )
            sm1_state = pipeline._sm_1p.context.state
            sm2_state = pipeline._sm_2p.context.state
            chain_in_progress = (
                sm1_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
                or sm2_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
            )
            chain_in_progress_suppresses = (
                chain_in_progress and not match_end_persisted
            )
            score_actively_moving = (
                pipeline._is_score_actively_moving(pipeline._recent_scores_1p)
                or pipeline._is_score_actively_moving(pipeline._recent_scores_2p)
            )
            hard_match_off = (
                score_zero_both or match_end_locked or post_match_lockdown_active
            )
            effective_hard_off = (
                hard_match_off and not score_actively_moving
                and not chain_in_progress_suppresses
            )
            sm_active = (
                sm1_state in (
                    BoardState.STABLE, BoardState.TSUMO_FALL, BoardState.CHAIN,
                    BoardState.OJAMA_FALL, BoardState.EFFECT,
                    BoardState.GRAVITY_SETTLE,
                )
                or sm2_state in (
                    BoardState.STABLE, BoardState.TSUMO_FALL, BoardState.CHAIN,
                    BoardState.OJAMA_FALL, BoardState.EFFECT,
                    BoardState.GRAVITY_SETTLE,
                )
            )
            recent_active = (
                pipeline._last_active_frame_time >= 0
                and (t_sec - pipeline._last_active_frame_time)
                <= RecognitionPipeline.MATCH_ACTIVE_HOLD_SEC
            )
            board_shows_real_gameplay = pipeline._board_shows_real_gameplay(frame)
            # raw_active: MatchStateDetector の生判定 (force_in_match との OR)。
            # 独立インスタンスへの再呼出し (stateless、副作用なし)。
            try:
                match_res_state = pipeline._match_detector.detect(frame).state
                raw_active = (
                    match_res_state == MatchState.IN_MATCH
                    or bool(getattr(pipeline, "_force_in_match", False))
                )
            except Exception:
                raw_active = False
            rows.append(BranchRow(
                video=case.video, frame_idx=frame_idx, t_sec=round(t_sec, 4),
                dt_from_anchor=round(t_sec - case.anchor_t_sec, 4),
                is_match_active=bool(res.is_match_active),
                score_zero_both=score_zero_both,
                match_end_locked=match_end_locked,
                post_match_lockdown_active=post_match_lockdown_active,
                match_end_persisted=match_end_persisted,
                chain_in_progress=chain_in_progress,
                chain_in_progress_suppresses=chain_in_progress_suppresses,
                score_actively_moving=score_actively_moving,
                hard_match_off=hard_match_off,
                effective_hard_off=effective_hard_off,
                raw_active=raw_active,
                recent_active=recent_active,
                sm_active=sm_active,
                sm1_state=getattr(sm1_state, "name", str(sm1_state)),
                sm2_state=getattr(sm2_state, "name", str(sm2_state)),
                board_shows_real_gameplay=board_shows_real_gameplay,
            ))
        frame_idx += 1
    cap.release()
    return rows


def find_reignition_row(rows: "list[BranchRow]") -> "BranchRow | None":
    """is_match_active が False→True に変わる最後の遷移行 (アンカー直前)。"""
    prev = None
    last_rise: "BranchRow | None" = None
    for r in rows:
        if prev is not None and (not prev.is_match_active) and r.is_match_active:
            last_rise = r
        prev = r
    return last_rise


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines: "list[str]" = [
        "# is_match_active 再点火経路の犯人特定 (2026-08-18)\n",
    ]
    for case in CASES:
        print(f"[{case.sheet}] 処理中...")
        rows = run_case(case)
        out_csv = OUT_DIR / f"reignition_diag_{case.sheet}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
        print(f"  [out] {out_csv} ({len(rows)} 行)")

        reignite = find_reignition_row(rows)
        report_lines.append(f"## {case.sheet}\n")
        if reignite is None:
            report_lines.append("再点火行が見つからなかった (窓内では常に True/False 固定)。\n")
        else:
            report_lines.append(
                f"再点火フレーム: frame_idx={reignite.frame_idx} "
                f"t_sec={reignite.t_sec} dt_from_anchor={reignite.dt_from_anchor}\n"
            )
            report_lines.append("```\n" + str(asdict(reignite)) + "\n```\n")
        report_lines.append("")

    report_path = OUT_DIR / "reignition_diag.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()
