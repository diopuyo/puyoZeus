"""真因計測: OJAMA_FALL 終了条件 (ROI top2行=0) が構造的に早すぎるか (2026-07-24)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない。
新規ファイルのみ、labeled_win/boards への書込みもしない。

## 背景 (アーキ設計、確定済み)
OjamaVisualDetector._detect_ojama_fall_exit (src/ojama_visual_detector.py:139-164)
は可視最上段 2 行 (OJAMA_ROI_HEIGHT=2) のおじゃま count==0 で即 STABLE 復帰する。
おじゃまは上段を素通りして下段まで落下する物理なので、この判定は「まだ落下中」
の段階で退出しうる。退出が早いと non_stable_cnn_history (最大 5 frame) に
サンプルが溜まる前に STABLE 復帰し、empty_to_color 3 票ゲート
(board_state_machine.py:_vote_majority_board, DEFAULT_EMPTY_TO_COLOR_MIN_VOTES=3)
で実在おじゃまが構造的に却下される。

案B: OJAMA_FALL の終了条件を GravitySettleDetector 相当の「盤面全体が安定」
判定に置換する。本スクリプトは置換前に、以下を実測で確定する:
    1. OJAMA_FALL 滞在時間の分布 (動画別・おじゃま量別)
    2. 退出後も盤面変化が継続しているか (退出が早すぎた直接証拠)
       GRAVITY_SETTLE_MAX_SEC(1.5s) が大量おじゃまで足りるか
    3. non_stable_cnn_history 実サンプル数 → 3 票ゲートでの却下セルを
       「時間不足(層2)」「色ノイズ(層3)」に分類
    4. おじゃまセル単体のフレーム間フリッカー率 (層3 重み判断用)

## 手法 (すべて import 済み src 関数の再利用、独自ロジック最小化)
実際の RecognitionPipeline.load_default() (本番デフォルト) を走らせ、各 side の
状態列 + cnn_board を frame 毎に記録する (confirmed_board も記録、baseline 用)。
OJAMA_FALL の各滞在区間について:
    - 「自然 settle」時刻: GravitySettleDetector と全く同じ計算式
      (GRAVITY_SETTLE_MIN_FRAMES/PHYSICS_CLEAR_MIN/PUYO_DIFF_THRESHOLD、
       すべて src.board_state_machine から import) を OJAMA_FALL 開始点から
      キャップなしで適用し、盤面 (raw cnn puyo 数) が真に安定する時刻を求める。
      → 実際の退出時刻との差分 (gap) が「早すぎた秒数」の直接測定値。
      → GRAVITY_SETTLE_MAX_SEC(1.5s) を超えるか否かが案Bの閾値材料。
    - 「footprint (落ち着いた想定おじゃまセル)」: 自然 settle 時点近傍の
      cnn_board 最頻値 (mode) で真値近似。
    - 実際の non_stable_cnn_history を独自再構築 (最大5 frame キャップ、
      entry 直後frame は積まれない、等 board_state_machine.py の実装を厳密再現)
      し、 _vote_majority_board / _merge_diff_only (src からそのまま import) で
      「実際に何が confirmed されたか」 を再現。footprint との差分セルを
      却下セルとし、 history_len<3 なら層2、 history はあるが該当セルの
      投票数<3 なら層3、 に分類する。

Usage (WSL 経由、CLAUDE.md プロセス管理ルール準拠):
    wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \\
      PYTHONPATH=. ./venv/bin/python scripts/_diag_ojama_fall_exit_timing_2026-07-24.py"
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_OJAMA,
)
from src.board_state_machine import (  # noqa: E402
    BoardState,
    DEFAULT_EMPTY_TO_COLOR_MIN_VOTES,
    DEFAULT_NON_STABLE_HISTORY_SIZE,
    GRAVITY_SETTLE_MAX_SEC,
    GRAVITY_SETTLE_MIN_FRAMES,
    GRAVITY_SETTLE_PHYSICS_CLEAR_MIN,
    GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD,
    _merge_diff_only,
    _vote_majority_board,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.visualize_recognition import (  # noqa: E402
    P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y, ROI_H, ROI_W,
)

# ============================
# 定数
# ============================

# 診断対象ウィンドウ: 前例 (recognition_diag_placement_dropout_2026-07-24) と
# 同一 (video別の実測境界・良好AUC対照/不振対照の位置づけを再利用、比較容易化)。
TARGET_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("c62", 862.0, 955.0, "game9 (872.4-949.5) 実測境界 + warmup margin"),
    ("30", 225.0, 315.0, "idx3 (233-451) 最長ゲーム冒頭90s、良好AUC動画(対照)"),
    ("35", 3110.0, 3200.0, "idx46 (3118-3247) 冒頭90s、序盤/中盤/終盤AUC不振動画"),
    ("38", 2585.0, 2675.0, "idx37 (2593-2780) 冒頭90s、序盤/中盤/終盤AUC不振動画"),
)

# 自然 settle 探索の最大先読み秒数 (これを超えて収束しない場合は「未収束」扱い)。
LOOKAHEAD_CAP_SEC: float = 5.0

# 実色 (お邪魔/空/UNKNOWN 以外) のフリッカー判定用集合。
_REAL_COLOR_VALUES: frozenset[int] = frozenset({1, 2, 3, 4, 5})

OUTPUT_DIR: Path = (
    PROJ_ROOT / "data" / "verify" / "recognition_diag_ojama_fall_exit_timing_2026-07-24"
)

# viz を出す代表例の上限数 (動画毎、gap 降順)。
MAX_VIZ_PER_VIDEO: int = 3


def _video_path(video_stem: str) -> Path:
    return PROJ_ROOT / "data" / "frames" / f"video_{video_stem}.mp4"


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 frame・1 side 分の最小記録。"""

    frame_idx: int
    t: float
    state: str
    cnn_board: Board
    confirmed_board: "Board | None"


@dataclass
class _OjamaFallEvent:
    """1 OJAMA_FALL 滞在区間の診断結果。"""

    video: str
    side: str
    entry_t: float
    exit_t: float
    duration_sec: float  # 実測 ROI 退出までの滞在時間
    history_len_at_exit: int  # 独自再構築した non_stable_cnn_history 長
    natural_settle_elapsed_sec: "float | None"  # entry から自然settleまで (キャップ無視)
    natural_settle_converged: bool  # LOOKAHEAD_CAP_SEC 内に収束したか
    gap_sec: "float | None"  # natural_settle_elapsed_sec - duration_sec (早すぎた秒数)
    exceeds_gravity_settle_cap: bool  # natural_settle_elapsed_sec > GRAVITY_SETTLE_MAX_SEC
    footprint_ojama_cells: int  # 想定「真」おじゃまセル数 (amount 代理指標)
    accepted_cells: int  # 却下ゲート通過 (実際に confirmed された) セル数
    rejected_insufficient_time: int  # 層2: history_len < 3 由来
    rejected_color_noise: int  # 層3: history はあるがセル票<3 由来
    rejected_other: int  # 想定外 (要目視確認)
    flicker_rate_mean: "float | None"  # footprint セルの平均フレーム間フリッカー率
    flicker_color_transitions: int  # 実色との遷移件数合計 (footprint セル)
    baseline_missing: bool  # baseline confirmed_board が見つからず判定不能


# ============================
# パス1: pipeline 走査
# ============================


def _collect_records(
    video_stem: str, start_sec: float, end_sec: float,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    """video を走査し、1P/2P それぞれの frame 記録を返す (本番デフォルト設定)。"""
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = RecognitionPipeline.load_default()  # 本番デフォルト設定そのまま
    pipe.set_video_id(video_stem)

    recs_1p: list[_FrameRec] = []
    recs_2p: list[_FrameRec] = []
    fi = start_frame
    n_read = 0
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        for side_recs, side_res in ((recs_1p, r.p1), (recs_2p, r.p2)):
            side_recs.append(_FrameRec(
                frame_idx=fi, t=t, state=side_res.state.name,
                cnn_board=side_res.cnn_board.copy(),
                confirmed_board=(
                    side_res.confirmed_board.copy()
                    if side_res.confirmed_board is not None else None
                ),
            ))
        fi += 1
        n_read += 1
        if n_read % 1800 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] [{video_stem}] t={t:.1f}s まで処理済み "
                f"({n_read} frames)", flush=True,
            )
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# OJAMA_FALL 区間抽出
# ============================


def _find_ojama_fall_segments(records: list[_FrameRec]) -> list[tuple[int, int]]:
    """(entry_idx, exit_idx) のリストを返す。exit_idx = 最初の非OJAMA_FALL frame。

    ウィンドウ末尾まで OJAMA_FALL のまま終わる区間 (退出未観測) は含めない。
    """
    segments: list[tuple[int, int]] = []
    entry: "int | None" = None
    of_name = BoardState.OJAMA_FALL.name
    for i, rec in enumerate(records):
        prev_is_of = i > 0 and records[i - 1].state == of_name
        if rec.state == of_name and not prev_is_of:
            entry = i
        if prev_is_of and rec.state != of_name and entry is not None:
            segments.append((entry, i))
            entry = None
    return segments


def _lookahead_bound(
    records: list[_FrameRec], entry_idx: int, exit_idx: int, fps: float,
) -> int:
    """自然settle探索の終端 index (exclusive)。次の CHAIN/OJAMA_FALL 開始で打ち切る
    (後続イベントによる汚染防止)。"""
    cap_idx = min(len(records), entry_idx + int(LOOKAHEAD_CAP_SEC * fps) + 1)
    chain_name = BoardState.CHAIN.name
    of_name = BoardState.OJAMA_FALL.name
    for i in range(exit_idx, cap_idx):
        st = records[i].state
        if st == chain_name or (st == of_name and i > exit_idx):
            return i
    return cap_idx


# ============================
# 自然 settle 計算 (GravitySettleDetector と同一式、キャップ無視)
# ============================


def _natural_settle_rel_idx(counts: list[int]) -> "int | None":
    """GRAVITY_SETTLE_* 定数と同一ロジックで自然収束 index (entry からの相対) を返す。

    board_state_machine.GravitySettleDetector.detect() の判定式を忠実に再現
    (キャップ GRAVITY_SETTLE_MAX_SEC のみ適用しない = 「本来何秒かかるか」を見る)。
    """
    if len(counts) < 2:
        return None
    prev_count = counts[0]
    stable_consec = 0
    for rel in range(1, len(counts)):
        if rel < GRAVITY_SETTLE_PHYSICS_CLEAR_MIN:
            prev_count = counts[rel]
            stable_consec = 0
            continue
        diff = abs(counts[rel] - prev_count)
        prev_count = counts[rel]
        stable_consec = stable_consec + 1 if diff < GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD else 0
        if stable_consec >= GRAVITY_SETTLE_MIN_FRAMES:
            return rel
    return None


def _mode_board(boards: list[Board]) -> Board:
    """cell 毎の最頻値盤面 (footprint = 真値近似)。"""
    result = Board()
    if not boards:
        return result
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            counter: dict[int, int] = {}
            for b in boards:
                v = b.get(r, c)
                counter[v] = counter.get(v, 0) + 1
            max_v, _n = max(counter.items(), key=lambda x: x[1])
            if max_v != COLOR_EMPTY:
                result.set(r, c, max_v)
    return result


# ============================
# non_stable_cnn_history 独自再構築 + ゲート再現
# ============================


def _reconstruct_history_and_confirmed(
    records: list[_FrameRec], entry_idx: int, exit_idx: int, baseline: "Board | None",
) -> tuple[list[Board], "Board | None"]:
    """board_state_machine.py の非STABLE history 蓄積ルールを再現する。

    entry frame 自体 (_apply_transition が処理) は積まれず、entry+1..exit-1 の
    _update_within_current_state 経路のみ積まれる (最大 DEFAULT_NON_STABLE_HISTORY_SIZE)。
    """
    raw_history = [records[i].cnn_board for i in range(entry_idx + 1, exit_idx)]
    capped_history = raw_history[-DEFAULT_NON_STABLE_HISTORY_SIZE:]
    if baseline is None:
        return capped_history, None
    empty_guard = (
        _vote_majority_board(capped_history, min_votes=DEFAULT_EMPTY_TO_COLOR_MIN_VOTES)
        if capped_history else None
    )
    reconstructed = _merge_diff_only(
        baseline, records[exit_idx].cnn_board, empty_to_color_guard=empty_guard,
    )
    return capped_history, reconstructed


def _find_baseline(records: list[_FrameRec], entry_idx: int) -> "Board | None":
    for i in range(entry_idx - 1, -1, -1):
        if records[i].confirmed_board is not None:
            return records[i].confirmed_board
    return None


# ============================
# 却下セル分類 + フリッカー計測
# ============================


def _classify_cells(
    baseline: Board, footprint: Board, reconstructed: "Board | None",
    capped_history: list[Board], history_len: int,
) -> tuple[int, int, int, int]:
    """(accepted, rejected_insufficient_time, rejected_color_noise, rejected_other)。"""
    accepted = 0
    rej_time = 0
    rej_noise = 0
    rej_other = 0
    if reconstructed is None:
        return 0, 0, 0, 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if baseline.get(r, c) != COLOR_EMPTY or footprint.get(r, c) != COLOR_OJAMA:
                continue
            recon_v = reconstructed.get(r, c)
            if recon_v == COLOR_OJAMA:
                accepted += 1
                continue
            if history_len < DEFAULT_EMPTY_TO_COLOR_MIN_VOTES:
                rej_time += 1
                continue
            cell_votes = sum(1 for b in capped_history if b.get(r, c) == COLOR_OJAMA)
            if cell_votes < DEFAULT_EMPTY_TO_COLOR_MIN_VOTES:
                rej_noise += 1
            else:
                rej_other += 1
    return accepted, rej_time, rej_noise, rej_other


def _compute_flicker(
    records: list[_FrameRec], entry_idx: int, window_end_idx: int,
    baseline: Board, footprint: Board,
) -> tuple["float | None", int]:
    """footprint おじゃまセルのフレーム間フリッカー率 (平均) + 実色遷移件数合計。"""
    target_cells = [
        (r, c) for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        if baseline.get(r, c) == COLOR_EMPTY and footprint.get(r, c) == COLOR_OJAMA
    ]
    if not target_cells or window_end_idx - entry_idx < 2:
        return None, 0
    rates: list[float] = []
    color_transitions_total = 0
    for (r, c) in target_cells:
        vals = [records[i].cnn_board.get(r, c) for i in range(entry_idx, window_end_idx)]
        n_pairs = len(vals) - 1
        transitions = sum(1 for k in range(1, len(vals)) if vals[k] != vals[k - 1])
        color_transitions = sum(
            1 for k in range(1, len(vals))
            if vals[k] != vals[k - 1]
            and (vals[k] in _REAL_COLOR_VALUES or vals[k - 1] in _REAL_COLOR_VALUES)
        )
        rates.append(transitions / n_pairs if n_pairs > 0 else 0.0)
        color_transitions_total += color_transitions
    return float(np.mean(rates)), color_transitions_total


# ============================
# イベント構築
# ============================


def _build_event(
    video: str, side: str, records: list[_FrameRec],
    entry_idx: int, exit_idx: int, fps: float,
) -> "_OjamaFallEvent | None":
    entry_t = records[entry_idx].t
    exit_t = records[exit_idx].t
    duration_sec = exit_t - entry_t

    baseline = _find_baseline(records, entry_idx)
    if baseline is None:
        return _OjamaFallEvent(
            video=video, side=side, entry_t=entry_t, exit_t=exit_t,
            duration_sec=duration_sec, history_len_at_exit=0,
            natural_settle_elapsed_sec=None, natural_settle_converged=False,
            gap_sec=None, exceeds_gravity_settle_cap=False,
            footprint_ojama_cells=0, accepted_cells=0,
            rejected_insufficient_time=0, rejected_color_noise=0, rejected_other=0,
            flicker_rate_mean=None, flicker_color_transitions=0, baseline_missing=True,
        )

    bound_idx = _lookahead_bound(records, entry_idx, exit_idx, fps)
    counts = [records[i].cnn_board.count_puyos() for i in range(entry_idx, bound_idx)]
    settle_rel = _natural_settle_rel_idx(counts)
    settle_abs = entry_idx + settle_rel if settle_rel is not None else None

    if settle_abs is not None:
        natural_elapsed = records[settle_abs].t - entry_t
        footprint_src = [
            records[i].cnn_board
            for i in range(max(entry_idx, settle_abs - 4), settle_abs + 1)
        ]
        window_end_idx = settle_abs + 1
    else:
        natural_elapsed = None
        footprint_src = [
            records[i].cnn_board
            for i in range(max(entry_idx, bound_idx - 5), bound_idx)
        ]
        window_end_idx = bound_idx
    footprint = _mode_board(footprint_src)

    capped_history, reconstructed = _reconstruct_history_and_confirmed(
        records, entry_idx, exit_idx, baseline,
    )
    history_len = len(capped_history)
    accepted, rej_time, rej_noise, rej_other = _classify_cells(
        baseline, footprint, reconstructed, capped_history, history_len,
    )
    flicker_mean, flicker_color = _compute_flicker(
        records, entry_idx, window_end_idx, baseline, footprint,
    )
    footprint_count = sum(
        1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        if baseline.get(r, c) == COLOR_EMPTY and footprint.get(r, c) == COLOR_OJAMA
    )

    gap_sec = (natural_elapsed - duration_sec) if natural_elapsed is not None else None
    exceeds_cap = natural_elapsed is not None and natural_elapsed > GRAVITY_SETTLE_MAX_SEC

    return _OjamaFallEvent(
        video=video, side=side, entry_t=entry_t, exit_t=exit_t,
        duration_sec=duration_sec, history_len_at_exit=history_len,
        natural_settle_elapsed_sec=natural_elapsed,
        natural_settle_converged=settle_abs is not None,
        gap_sec=gap_sec, exceeds_gravity_settle_cap=exceeds_cap,
        footprint_ojama_cells=footprint_count, accepted_cells=accepted,
        rejected_insufficient_time=rej_time, rejected_color_noise=rej_noise,
        rejected_other=rej_other, flicker_rate_mean=flicker_mean,
        flicker_color_transitions=flicker_color, baseline_missing=False,
    )


# ============================
# CSV / summary 出力
# ============================


def _write_events_csv(events: list[_OjamaFallEvent], out_path: Path) -> None:
    cols = [
        "video", "side", "entry_t", "exit_t", "duration_sec", "history_len_at_exit",
        "natural_settle_elapsed_sec", "natural_settle_converged", "gap_sec",
        "exceeds_gravity_settle_cap", "footprint_ojama_cells", "accepted_cells",
        "rejected_insufficient_time", "rejected_color_noise", "rejected_other",
        "flicker_rate_mean", "flicker_color_transitions", "baseline_missing",
    ]
    lines = [",".join(cols)]
    for e in events:
        vals = [
            e.video, e.side, f"{e.entry_t:.3f}", f"{e.exit_t:.3f}", f"{e.duration_sec:.3f}",
            e.history_len_at_exit,
            "" if e.natural_settle_elapsed_sec is None else f"{e.natural_settle_elapsed_sec:.3f}",
            e.natural_settle_converged,
            "" if e.gap_sec is None else f"{e.gap_sec:.3f}",
            e.exceeds_gravity_settle_cap, e.footprint_ojama_cells, e.accepted_cells,
            e.rejected_insufficient_time, e.rejected_color_noise, e.rejected_other,
            "" if e.flicker_rate_mean is None else f"{e.flicker_rate_mean:.4f}",
            e.flicker_color_transitions, e.baseline_missing,
        ]
        lines.append(",".join(str(v) for v in vals))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_summary(events: list[_OjamaFallEvent]) -> dict:
    valid = [e for e in events if not e.baseline_missing]
    with_footprint = [e for e in valid if e.footprint_ojama_cells > 0]
    converged = [e for e in with_footprint if e.natural_settle_converged]
    gaps = [e.gap_sec for e in converged if e.gap_sec is not None]
    durations = [e.duration_sec for e in with_footprint]
    amounts = [e.footprint_ojama_cells for e in with_footprint]
    corr = None
    if len(durations) >= 3 and np.std(durations) > 0 and np.std(amounts) > 0:
        corr = float(np.corrcoef(durations, amounts)[0, 1])
    total_rej_time = sum(e.rejected_insufficient_time for e in valid)
    total_rej_noise = sum(e.rejected_color_noise for e in valid)
    total_rej_other = sum(e.rejected_other for e in valid)
    total_accepted = sum(e.accepted_cells for e in valid)
    flicker_vals = [e.flicker_rate_mean for e in with_footprint if e.flicker_rate_mean is not None]
    return {
        "n_events_total": len(events),
        "n_events_baseline_missing": len(events) - len(valid),
        "n_events_with_footprint_ojama": len(with_footprint),
        "n_events_natural_settle_converged": len(converged),
        "n_events_not_converged_within_5s": len(with_footprint) - len(converged),
        "mean_gap_sec_early_exit": (float(np.mean(gaps)) if gaps else None),
        "median_gap_sec_early_exit": (float(np.median(gaps)) if gaps else None),
        "rate_gap_positive_early_exit": (
            sum(1 for g in gaps if g > 0) / len(gaps) if gaps else None
        ),
        "n_exceeds_gravity_settle_cap_1_5s": sum(
            1 for e in converged if e.exceeds_gravity_settle_cap
        ),
        "rate_exceeds_gravity_settle_cap_1_5s": (
            sum(1 for e in converged if e.exceeds_gravity_settle_cap) / len(converged)
            if converged else None
        ),
        "duration_vs_footprint_amount_corr": corr,
        "duration_sec_mean": (float(np.mean(durations)) if durations else None),
        "duration_sec_median": (float(np.median(durations)) if durations else None),
        "footprint_ojama_cells_mean": (float(np.mean(amounts)) if amounts else None),
        "rejected_cells_total": total_rej_time + total_rej_noise + total_rej_other,
        "rejected_insufficient_time_total": total_rej_time,
        "rejected_color_noise_total": total_rej_noise,
        "rejected_other_total": total_rej_other,
        "accepted_cells_total": total_accepted,
        "rate_rejected_insufficient_time_of_rejected": (
            total_rej_time / (total_rej_time + total_rej_noise + total_rej_other)
            if (total_rej_time + total_rej_noise + total_rej_other) else None
        ),
        "rate_rejected_color_noise_of_rejected": (
            total_rej_noise / (total_rej_time + total_rej_noise + total_rej_other)
            if (total_rej_time + total_rej_noise + total_rej_other) else None
        ),
        "flicker_rate_mean_overall": (float(np.mean(flicker_vals)) if flicker_vals else None),
    }


def _format_summary_text(overall: dict, per_video: dict[str, dict]) -> str:
    lines = [
        "==== OJAMA_FALL 終了条件 早すぎ判定 実測サマリ (2026-07-24) ====",
        f"検出 OJAMA_FALL 区間総数: {overall['n_events_total']} "
        f"(baseline不明で判定不能: {overall['n_events_baseline_missing']})",
        f"footprintおじゃま有り区間: {overall['n_events_with_footprint_ojama']}",
        f"自然settle収束 (5秒以内): {overall['n_events_natural_settle_converged']} "
        f"/ 未収束: {overall['n_events_not_converged_within_5s']}",
        f"早すぎ秒数 (自然settle - 実退出) 平均: {overall['mean_gap_sec_early_exit']}"
        f" 中央値: {overall['median_gap_sec_early_exit']}"
        f" (正=早すぎ、比率: {overall['rate_gap_positive_early_exit']})",
        f"GRAVITY_SETTLE_MAX_SEC(1.5s)超過 (収束事例中): "
        f"{overall['n_exceeds_gravity_settle_cap_1_5s']} "
        f"({overall['rate_exceeds_gravity_settle_cap_1_5s']})",
        f"滞在秒数 vs footprintおじゃま量 相関係数: "
        f"{overall['duration_vs_footprint_amount_corr']}",
        f"滞在秒数 平均/中央値: {overall['duration_sec_mean']}/{overall['duration_sec_median']}",
        f"却下セル総数: {overall['rejected_cells_total']} (採用 {overall['accepted_cells_total']}) 内訳:"
        f" 層2(時間不足)={overall['rejected_insufficient_time_total']}"
        f"({overall['rate_rejected_insufficient_time_of_rejected']})"
        f" 層3(色ノイズ)={overall['rejected_color_noise_total']}"
        f"({overall['rate_rejected_color_noise_of_rejected']})"
        f" その他={overall['rejected_other_total']}",
        f"おじゃまセル フリッカー率 (平均): {overall['flicker_rate_mean_overall']}",
        "--- 動画別 ---",
    ]
    for video, s in per_video.items():
        lines.append(
            f"  {video}: events={s['n_events_total']} "
            f"footprint有={s['n_events_with_footprint_ojama']} "
            f"収束={s['n_events_natural_settle_converged']} "
            f"gap平均={s['mean_gap_sec_early_exit']} "
            f"cap超過率={s['rate_exceeds_gravity_settle_cap_1_5s']} "
            f"却下(層2/層3)={s['rejected_insufficient_time_total']}/"
            f"{s['rejected_color_noise_total']}",
        )
    return "\n".join(lines)


# ============================
# viz
# ============================


def _roi_for_side(side: str) -> tuple[int, int]:
    return (P1_ROI_X, P1_ROI_Y) if side == "1P" else (P2_ROI_X, P2_ROI_Y)


def _seek_frame(cap: cv2.VideoCapture, frame_idx: int) -> "np.ndarray | None":
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _write_timeline_plot(
    video_stem: str, side: str, records: list[_FrameRec],
    entry_idx: int, exit_idx: int, bound_idx: int,
    settle_abs_idx: "int | None", event: _OjamaFallEvent, out_path: Path,
) -> None:
    """状態遷移 + ぷよ数 + 各種マーカーのタイムライン。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idxs = list(range(entry_idx, bound_idx))
    ts = np.array([records[i].t - event.entry_t for i in idxs])
    counts = np.array([records[i].cnn_board.count_puyos() for i in idxs])
    state_map = {
        "STABLE": 0, "TSUMO_FALL": 1, "CHAIN": 2, "OJAMA_FALL": 3,
        "EFFECT": 4, "GRAVITY_SETTLE": 5, "MENU": 6,
    }
    states_num = np.array([state_map.get(records[i].state, -1) for i in idxs])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.step(ts, states_num, where="post", color="#1f77b4")
    ax1.set_yticks(list(state_map.values()))
    ax1.set_yticklabels(list(state_map.keys()))
    ax1.set_title(f"{video_stem} {side} OJAMA_FALL entry_t={event.entry_t:.2f}s state 遷移")
    ax2.plot(ts, counts, marker=".", markersize=3, color="#2ca02c", label="cnn puyo数")
    ax2.axvline(0.0, color="green", linestyle="--", label="entry")
    ax2.axvline(event.duration_sec, color="orange", linestyle="--", label="実退出(ROI)")
    if settle_abs_idx is not None:
        ax2.axvline(
            records[settle_abs_idx].t - event.entry_t, color="red", linestyle="--",
            label="自然settle",
        )
    ax2.axvline(GRAVITY_SETTLE_MAX_SEC, color="black", linestyle=":", label="GRAVITY_SETTLE_MAX_SEC")
    ax2.set_xlabel("entry からの経過秒")
    ax2.set_ylabel("puyo count (cnn)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_pixel_strip(
    video_stem: str, side: str, fps: float,
    entry_t: float, exit_t: float, settle_t: "float | None", out_path: Path,
) -> None:
    """entry / 実退出 / 自然settle 時点の実画素 (ROI crop) を横並びにする。"""
    roi_x, roi_y = _roi_for_side(side)
    cap = cv2.VideoCapture(str(_video_path(video_stem)))
    times = [("entry", entry_t), ("実退出(ROI)", exit_t)]
    if settle_t is not None:
        times.append(("自然settle", settle_t))
    crops = []
    for label, t in times:
        fi = int(round(t * fps))
        frame = _seek_frame(cap, fi)
        if frame is None:
            continue
        crop = frame[roi_y:roi_y + ROI_H, roi_x:roi_x + ROI_W].copy()
        cv2.putText(
            crop, f"{label} t={t:.2f}s", (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55,
            (0, 255, 255), 1, cv2.LINE_AA,
        )
        crops.append(crop)
    cap.release()
    if not crops:
        return
    h = max(c.shape[0] for c in crops)
    sep = np.full((h, 6, 3), (255, 255, 255), dtype=np.uint8)
    parts = []
    for c in crops:
        if c.shape[0] != h:
            pad = np.zeros((h, c.shape[1], 3), dtype=np.uint8)
            pad[: c.shape[0]] = c
            c = pad
        parts.append(c)
        parts.append(sep)
    out = np.hstack(parts[:-1])
    cv2.imwrite(str(out_path), out)


# ============================
# メイン
# ============================


def _print_progress(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events: list[_OjamaFallEvent] = []
    per_video_summary: dict[str, dict] = {}
    # viz 用に (video, side, records, entry_idx, exit_idx, bound_idx, settle_abs_idx, event) を保持
    viz_candidates: list[tuple] = []

    for video_stem, start_sec, end_sec, note in TARGET_WINDOWS:
        _print_progress(f"[{video_stem}] 開始 window={start_sec:.1f}-{end_sec:.1f}s ({note})")
        t0 = time.time()
        recs_1p, recs_2p, fps = _collect_records(video_stem, start_sec, end_sec)
        elapsed = time.time() - t0
        _print_progress(
            f"[{video_stem}] pass1 完了 ({len(recs_1p)} frame, {elapsed:.1f}s, "
            f"{len(recs_1p) / max(elapsed, 1e-6):.2f} fps相当)",
        )

        video_events: list[_OjamaFallEvent] = []
        for side, records in (("1P", recs_1p), ("2P", recs_2p)):
            segments = _find_ojama_fall_segments(records)
            for entry_idx, exit_idx in segments:
                ev = _build_event(video_stem, side, records, entry_idx, exit_idx, fps)
                if ev is None:
                    continue
                video_events.append(ev)
                bound_idx = _lookahead_bound(records, entry_idx, exit_idx, fps)
                settle_rel = _natural_settle_rel_idx(
                    [records[i].cnn_board.count_puyos() for i in range(entry_idx, bound_idx)],
                )
                settle_abs = entry_idx + settle_rel if settle_rel is not None else None
                viz_candidates.append((
                    video_stem, side, records, entry_idx, exit_idx, bound_idx, settle_abs, ev,
                ))

        all_events.extend(video_events)
        _write_events_csv(video_events, OUTPUT_DIR / f"events_{video_stem}.csv")
        per_video_summary[video_stem] = _aggregate_summary(video_events)
        per_video_summary[video_stem]["note"] = note
        _print_progress(
            f"[{video_stem}] OJAMA_FALL区間={len(video_events)}件 "
            f"footprint有={per_video_summary[video_stem]['n_events_with_footprint_ojama']}件 "
            f"完了",
        )

    # viz: gap_sec (早すぎ秒数) 降順で動画毎に代表例を出す
    for video_stem, _s, _e, _n in TARGET_WINDOWS:
        cands = [
            v for v in viz_candidates
            if v[0] == video_stem and v[7].footprint_ojama_cells > 0 and v[7].gap_sec is not None
        ]
        cands.sort(key=lambda v: v[7].gap_sec, reverse=True)
        for (vs, side, records, entry_idx, exit_idx, bound_idx, settle_abs, ev) in (
            cands[:MAX_VIZ_PER_VIDEO]
        ):
            label = f"{ev.entry_t:.2f}".replace(".", "_")
            fps_for_video = len(records) / max(
                (records[-1].t - records[0].t), 1e-6,
            ) if len(records) > 1 else 60.0
            _write_timeline_plot(
                vs, side, records, entry_idx, exit_idx, bound_idx, settle_abs, ev,
                OUTPUT_DIR / f"viz_{vs}_{side}_t{label}_timeline.png",
            )
            settle_t = records[settle_abs].t if settle_abs is not None else None
            _write_pixel_strip(
                vs, side, fps_for_video, ev.entry_t, ev.exit_t, settle_t,
                OUTPUT_DIR / f"viz_{vs}_{side}_t{label}_pixels.png",
            )
        _print_progress(f"[{video_stem}] viz {min(len(cands), MAX_VIZ_PER_VIDEO)}件 出力完了")

    overall = _aggregate_summary(all_events)
    summary = {"overall": overall, "per_video": per_video_summary}
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    text = _format_summary_text(overall, per_video_summary)
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


if __name__ == "__main__":
    main()
