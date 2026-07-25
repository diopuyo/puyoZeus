"""confirmed_board 書き込み元トレース診断 (色フリッカ/消失 根因確定・Step1)。

背景: 盤面矛盾監査器 (scripts/physics_violation_audit.py) が検出する
color_flicker/conservation_loss/conservation_gain が、STABLE 中に
confirmed_board を書き換える 13 系統の経路 (P1〜P13、アーキ確定済) の
どれ由来かを切り分けるための read-only 計装スクリプト。

read-only 原則: src/ は一切変更しない。RecognitionPipeline / BoardStateMachine
の該当関数・メソッドを monkeypatch (functools.wraps) でラップし、
confirmed_board への書き込みを経路IDタグ付きで記録する。with 節を抜けると
必ず元の実装に復元する (physics_violation_audit.py の
_wrap_update_for_progress と同じ一時パッチパターン)。

フック対象 (アーキ仕様):
    P1: board_state_machine._merge_diff_only
        (NON-STABLE→STABLE 復帰時の差分マージ)
    P5: board_state_machine._apply_stable_recovery_gate
        (STABLE 中の事後復旧ゲート)
    P2: recognition_pipeline.infer_placement (物理推論による着地確定)
    P3: recognition_pipeline.resolve_after_placement (連鎖即時判定)
    P4/P6/P7/P8/P9: recognition_pipeline.RecognitionPipeline._step_side 内の
        インライン補正 (連鎖確定上書き/NEXT制約/着地投票/長期override/
        T2差し戻し等)。個別の関数境界がないため、P1/P2/P3/P5 で捕捉済みの
        セルを除いた `_step_side` 呼び出し前後の残差 diff を
        "P_inline_untracked_P4_P6_P7_P8_P9" としてまとめて記録する
        (ユーザー指定の フォールバック方式: メソッド呼び出し前後の board diff)。

出力:
    data/verify/write_trace/<stem>_<side>.jsonl
        {route_id, frame_idx, t_sec, side, cells:[[r,c,before,after]...], meta}
        を適用順で記録。
    data/verify/write_trace/<stem>_crosstab_summary.json / .txt
        監査器 (physics_violation_audit) が検出した color_flicker/
        conservation_loss/conservation_gain の各違反について、
        (prev_frame_idx, frame_idx] 区間の write_trace から直近の書き込み経路を
        タグ付けし、速度バケット (fast<0.3s / slow>=0.3s) 別に集計した表。

Usage:
    PYTHONPATH=. python -m scripts._diag_confirmed_write_trace_2026-07-25 --smoke
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import src.board_state_machine as bsm  # noqa: E402
import src.recognition_pipeline as rp  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from scripts.physics_violation_audit import (  # noqa: E402
    SMOKE_MAX_SEC, SMOKE_START_SEC, SMOKE_VIDEO_STEM, Violation,
    _check_color_flicker, _check_conservation,
    _mask_records_by_excluded_intervals, _scan_telop_exclusion_intervals,
)
from scripts.recognition_physics_review import _FrameRecord, _capture_frames  # noqa: E402

# ============================
# 定数
# ============================
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "write_trace"

# 経路IDタグ (アーキ仕様の P1〜P13 のうち計装対象)。
ROUTE_P1_MERGE_DIFF_ONLY: str = "P1_merge_diff_only"
ROUTE_P5_STABLE_RECOVERY_GATE: str = "P5_stable_recovery_gate"
ROUTE_P2_INFER_PLACEMENT: str = "P2_infer_placement"
ROUTE_P3_RESOLVE_AFTER_PLACEMENT: str = "P3_resolve_after_placement"
# P4(連鎖確定上書き)/P6(NEXT制約)/P7(着地投票)/P8(長期override)/P9(T2差し戻し)は
# recognition_pipeline._step_side 内のインラインコードで関数境界がないため、
# P1/P2/P3/P5 捕捉済みセルを除いた残差 diff としてまとめて記録する。
ROUTE_INLINE_CATCHALL: str = "P_inline_untracked_P4_P6_P7_P8_P9"
# Step2 (色フリッカ何手ズレ診断): TSUMO_FALL→STABLE 遷移フレームの
# next_queue スナップショット (cells を伴わない meta-only レコード)。
ROUTE_P2_DIAG_QUEUE_CONTEXT: str = "P2_diag_queue_context"

# ズレ幅判定のカテゴリラベル (何手ズレ診断・Step2)。
OFFSET_LABEL_ZERO: str = "0手(今回ツモ=prev_tail[-2])"
OFFSET_LABEL_ONE: str = "1手先(prev_tail[-1])"
OFFSET_LABEL_TWO: str = "2手先(post_tail[-1])"
OFFSET_LABEL_UNMATCHED: str = "unmatched(どれとも不一致)"
OFFSET_LABEL_NO_CONTEXT: str = "no_queue_context(突合不能)"
# queue tail 保持数 (次の NEXT 変化イベント履歴を遡る深さ)。
QUEUE_TAIL_DEPTH: int = 3

# クロス集計: 速度バケット閾値 (秒)。curr_t_sec - prev_t_sec がこれ未満なら fast。
SPEED_BUCKET_THRESHOLD_SEC: float = 0.3
SPEED_BUCKET_FAST: str = "fast(<0.3s)"
SPEED_BUCKET_SLOW: str = "slow(>=0.3s)"
ROUTE_UNATTRIBUTED: str = "unattributed(write_trace内に該当なし)"

# 試合開始直後 確定遅延診断 (Step3, 2026-07-25追記):
# baseline-broken自己リセット/is_active ちらつき/match_just_started 再発火の
# 主犯切り分け用。reset() の呼び出し種別タグ。
RESET_KIND_BASELINE_BROKEN: str = "baseline_broken_self_reset"
RESET_KIND_DRIFT_RESYNC: str = "drift_resync_keep_match_state"


@dataclass
class _TraceCtx:
    """全フックが共有する可変コンテキスト (単一スレッド同期実行前提)。

    RecognitionPipeline.update は同期呼び出しで 1P→2P の順に _step_side を
    呼ぶため、共有可変状態でも競合しない (feedback: 並列化しない前提)。
    """

    video_stem: str = ""
    side: str = ""
    frame_idx: int = -1
    t_sec: float = 0.0
    # この _step_side 呼び出し内で P1/P2/P3/P5 が書いたセル集合
    # (キャッチオール残差 diff からの除外用)。
    claimed_cells: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class WriteTraceRecord:
    """1 件の書き込みイベント (経路タグ付き)。"""

    route_id: str
    frame_idx: int
    t_sec: float
    side: str
    cells: list[list[int]]
    meta: dict


class WriteTraceRecorder:
    """write_trace レコードを蓄積する (read-only、状態は本クラスのみ保持)。"""

    def __init__(self) -> None:
        self.records: list[WriteTraceRecord] = []

    def record(
        self, ctx: _TraceCtx, route_id: str, cells: list[list[int]], meta: dict,
    ) -> None:
        """1 件記録する (cells が空なら記録しない)。"""
        if not cells:
            return
        self.records.append(WriteTraceRecord(
            route_id=route_id, frame_idx=ctx.frame_idx, t_sec=ctx.t_sec,
            side=ctx.side, cells=cells, meta=meta,
        ))

    def record_meta_only(self, ctx: _TraceCtx, route_id: str, meta: dict) -> None:
        """cells を伴わない meta 専用レコードを記録する (queue_context 診断用)。

        通常の record() は cells が空だと記録しない (盤面書き込みイベント
        専用のため)。本メソッドは「書き込みは起きていないが診断に必要な
        コンテキスト」(NEXT queue スナップショット等) を記録するための
        別経路であり、cells=[] を許容する。
        """
        self.records.append(WriteTraceRecord(
            route_id=route_id, frame_idx=ctx.frame_idx, t_sec=ctx.t_sec,
            side=ctx.side, cells=[], meta=meta,
        ))


@dataclass
class MatchStartDiagRecorder:
    """試合開始直後 確定遅延診断 (Step3) 専用の軽量計装レコーダー (read-only)。

    ①baseline-broken自己リセットループ/②is_activeちらつきによる
    match_just_started 再発火/④HSV較正タイミングの主犯切り分けに使う。
    write_trace (P1〜P9 経路タグ) とは独立した集計 (盤面セル書き込みを
    伴わないイベントも扱うため、既存 WriteTraceRecorder とは別クラスにする)。
    """

    reset_events: list[dict] = field(default_factory=list)
    match_start_events: list[dict] = field(default_factory=list)
    is_active_flips: list[dict] = field(default_factory=list)


def _make_reset_wrapper(orig, trace_ctx: _TraceCtx, diag: MatchStartDiagRecorder):
    """②③: board_state_machine.BoardStateMachine.reset をラップし、呼び出しを記録する。

    keep_match_state=False (recognition_pipeline.py:4625 baseline-broken
    自己リセット) と keep_match_state=True (同:4655 drift resync) は
    src 側で呼び出し引数が異なるため、この差だけで呼び出し元を判別できる
    (別途 monkeypatch 不要)。frame_idx/t_sec/side は `_step_side` ラッパー
    (_make_step_side_wrapper) が reset() 呼び出しに先立って trace_ctx に
    設定済 (同期実行前提、_TraceCtx docstring 参照)。
    """

    @functools.wraps(orig)
    def wrapped(self, *, keep_match_state: bool = False):
        kind = RESET_KIND_DRIFT_RESYNC if keep_match_state else RESET_KIND_BASELINE_BROKEN
        diag.reset_events.append({
            "kind": kind, "side": trace_ctx.side,
            "frame_idx": trace_ctx.frame_idx, "t_sec": trace_ctx.t_sec,
        })
        return orig(self, keep_match_state=keep_match_state)

    return wrapped


def _make_update_wrapper(orig, diag: MatchStartDiagRecorder, prev_holder: dict):
    """②: RecognitionPipeline.update をラップし、match_active_started の新規セット

    (= is_active が切れて再度立ち上がった回数、match_just_started 再発火の
    直接証拠) と is_match_active の反転回数を記録する (instance 属性の
    読み取りのみ、src は一切変更しない)。
    """

    @functools.wraps(orig)
    def wrapped(self, frame_idx, time_sec, frame):
        result = orig(self, frame_idx, time_sec, frame)
        started_frame = getattr(self, "_match_active_started_frame", -1)
        if started_frame >= 0 and prev_holder["started_frame"] < 0:
            diag.match_start_events.append({"frame_idx": frame_idx, "t_sec": time_sec})
        prev_holder["started_frame"] = started_frame
        cur_active = bool(result.is_match_active)
        if (
            prev_holder["is_active"] is not None
            and cur_active != prev_holder["is_active"]
        ):
            diag.is_active_flips.append({
                "frame_idx": frame_idx, "t_sec": time_sec,
                "from": prev_holder["is_active"], "to": cur_active,
            })
        prev_holder["is_active"] = cur_active
        return result

    return wrapped


# ============================
# P1: _merge_diff_only 分岐推定 (診断専用、src非改変)
# ============================


def _classify_merge_branch(
    base_v: int, cnn_v: int, merged_v: int, guard_v: int | None,
    allow_puyo_to_empty: bool,
) -> str:
    """_merge_diff_only の1セル分の分岐を観測値から後追い推定する (診断専用)。

    src/board_state_machine.py:_merge_diff_only の実装は変更せず、
    baseline/cnn/merged/guard の観測値のみから「どの分岐を通ったか」を
    再構成する (本関数は差分が実際に発生した cell のみ呼ばれる前提、
    baseline維持系の分岐は理論上ここに現れない)。
    """
    if cnn_v == COLOR_UNKNOWN:
        return "D_cnn_unknown_keep_baseline"
    if not allow_puyo_to_empty and base_v != COLOR_EMPTY and cnn_v == COLOR_EMPTY:
        return "B_puyo_to_empty_banned"
    if base_v == COLOR_EMPTY and cnn_v != COLOR_EMPTY:
        if guard_v is None:
            return "F_guard_absent_direct_write"
        if guard_v == COLOR_EMPTY:
            return "F_guard_rejected"
        if merged_v == guard_v and merged_v != cnn_v:
            return "F_guard_majority_value_write"
        if merged_v == cnn_v:
            return "F_guard_passed_direct_write"
        return "unclassified"
    return "direct_write"


def _diff_cells(before: object, after: object, exclude: set[tuple[int, int]] | None = None) -> list[list[int]]:
    """2つの Board から (r,c,before_v,after_v) の変化セル一覧を作る (exclude 除外)。"""
    cells: list[list[int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if exclude is not None and (r, c) in exclude:
                continue
            bv, av = int(before.get(r, c)), int(after.get(r, c))  # type: ignore[attr-defined]
            if bv != av:
                cells.append([r, c, bv, av])
    return cells


# ============================
# フック本体 (monkeypatch wrapper 生成)
# ============================


def _make_merge_diff_only_wrapper(orig, trace_ctx: _TraceCtx, recorder: WriteTraceRecorder):
    """P1: board_state_machine._merge_diff_only をラップする。"""

    @functools.wraps(orig)
    def wrapped(baseline, new_cnn, *args, **kwargs):
        merged = orig(baseline, new_cnn, *args, **kwargs)
        if baseline is None:
            return merged  # 初回確定 (書き換え元なし、対象外)
        guard = kwargs.get("empty_to_color_guard")
        allow_puyo_to_empty = kwargs.get("allow_puyo_to_empty", True)
        cells: list[list[int]] = []
        branches: dict[str, str] = {}
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                bv, mv = int(baseline.get(r, c)), int(merged.get(r, c))
                if bv == mv:
                    continue
                cnn_v = int(new_cnn.get(r, c))
                guard_v = int(guard.get(r, c)) if guard is not None else None
                branch = _classify_merge_branch(bv, cnn_v, mv, guard_v, allow_puyo_to_empty)
                branches[f"{r}_{c}"] = branch
                cells.append([r, c, bv, mv])
                trace_ctx.claimed_cells.add((r, c))
        recorder.record(
            trace_ctx, ROUTE_P1_MERGE_DIFF_ONLY, cells,
            meta={"history_was_empty": guard is None, "branches": branches},
        )
        return merged

    return wrapped


def _make_stable_recovery_gate_wrapper(orig, trace_ctx: _TraceCtx, recorder: WriteTraceRecorder):
    """P5: board_state_machine._apply_stable_recovery_gate をラップする (in-place mutation)。"""

    @functools.wraps(orig)
    def wrapped(ctx, signals, min_frames):
        confirmed = ctx.confirmed_board
        before = confirmed.copy() if confirmed is not None else None
        orig(ctx, signals, min_frames)
        after = ctx.confirmed_board
        if before is None or after is None:
            return
        directions: dict[str, str] = {}
        cells: list[list[int]] = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                bv, av = int(before.get(r, c)), int(after.get(r, c))
                if bv == av:
                    continue
                directions[f"{r}_{c}"] = "add" if bv == COLOR_EMPTY else "fix"
                cells.append([r, c, bv, av])
                trace_ctx.claimed_cells.add((r, c))
        recorder.record(
            trace_ctx, ROUTE_P5_STABLE_RECOVERY_GATE, cells,
            meta={"directions": directions},
        )

    return wrapped


def _make_infer_placement_wrapper(orig, trace_ctx: _TraceCtx, recorder: WriteTraceRecorder):
    """P2: recognition_pipeline.infer_placement をラップする。"""

    @functools.wraps(orig)
    def wrapped(prev_confirmed, cnn_board, falling_pair, *args, **kwargs):
        result = orig(prev_confirmed, cnn_board, falling_pair, *args, **kwargs)
        if result is None or prev_confirmed is None:
            return result
        cells = _diff_cells(prev_confirmed, result)
        for r, c, _, _ in cells:
            trace_ctx.claimed_cells.add((r, c))
        # 何手ズレ診断 (Step2): falling_pair は infer_placement 呼び出し時点で
        # 既に手元にある第3位置引数 (prev_confirmed, cnn_board, falling_pair の順)。
        # meta に残すことで P2_diag_queue_context の queue tail と突合できる。
        recorder.record(
            trace_ctx, ROUTE_P2_INFER_PLACEMENT, cells,
            meta={"falling_pair": list(falling_pair) if falling_pair else None},
        )
        return result

    return wrapped


def _make_resolve_after_placement_wrapper(orig, trace_ctx: _TraceCtx, recorder: WriteTraceRecorder):
    """P3: recognition_pipeline.resolve_after_placement をラップする。"""

    @functools.wraps(orig)
    def wrapped(inferred_landing, chain_sim, *args, **kwargs):
        final_board, chain_count = orig(inferred_landing, chain_sim, *args, **kwargs)
        cells = _diff_cells(inferred_landing, final_board)
        for r, c, _, _ in cells:
            trace_ctx.claimed_cells.add((r, c))
        recorder.record(
            trace_ctx, ROUTE_P3_RESOLVE_AFTER_PLACEMENT, cells,
            meta={"chain_count": int(chain_count)},
        )
        return final_board, chain_count

    return wrapped


def _make_step_side_wrapper(
    orig, trace_ctx: _TraceCtx, recorder: WriteTraceRecorder, video_stem_holder: dict,
):
    """P4/P6/P7/P8/P9 キャッチオール: RecognitionPipeline._step_side 全体をラップする。

    呼び出し前後の confirmed_board diff から、P1/P2/P3/P5 (このメソッド内で
    入れ子に発火する) が既に捕捉したセルを除いた残差を
    ROUTE_INLINE_CATCHALL として記録する。
    """

    @functools.wraps(orig)
    def wrapped(self, side, frame_idx, time_sec, is_active, cnn_board, chain_event, **kwargs):
        sm = kwargs["sm"]
        trace_ctx.video_stem = video_stem_holder["stem"]
        trace_ctx.side = side
        trace_ctx.frame_idx = frame_idx
        trace_ctx.t_sec = time_sec
        trace_ctx.claimed_cells = set()
        board_before = sm.context.confirmed_board
        before_copy = board_before.copy() if board_before is not None else None
        # 何手ズレ診断 (Step2): 遷移前の state と next_queue 末尾を
        # orig() 呼び出し前 (= sm.update() が走る前) にスナップショットする。
        # recognition_pipeline.py:3803 の prev_next_queue = list(sm.context.next_queue)
        # と同一タイミング (orig 内で sm.update() する直前) を外側から再現する。
        prev_state = sm.context.state
        prev_queue_tail = [list(p) for p in sm.context.next_queue[-QUEUE_TAIL_DEPTH:]]

        result = orig(
            self, side, frame_idx, time_sec, is_active, cnn_board, chain_event, **kwargs,
        )

        board_after = sm.context.confirmed_board
        if before_copy is not None and board_after is not None:
            residual_cells = _diff_cells(before_copy, board_after, exclude=trace_ctx.claimed_cells)
            recorder.record(
                trace_ctx, ROUTE_INLINE_CATCHALL, residual_cells,
                meta={
                    "note": "_step_side 内インライン補正の残差diff (P4連鎖確定上書き/"
                            "P6 NEXT制約/P7着地投票/P8長期override/P9 T2差し戻し等、"
                            "個別関数境界がないため合算)",
                },
            )
        # 何手ズレ診断 (Step2): TSUMO_FALL→STABLE 遷移フレームのみ、
        # 遷移前後の next_queue 末尾 QUEUE_TAIL_DEPTH 件を記録する
        # (P2_infer_placement の meta.falling_pair と (side, frame_idx) で突合する)。
        post_state = sm.context.state
        if prev_state == bsm.BoardState.TSUMO_FALL and post_state == bsm.BoardState.STABLE:
            post_queue_tail = [list(p) for p in sm.context.next_queue[-QUEUE_TAIL_DEPTH:]]
            recorder.record_meta_only(
                trace_ctx, ROUTE_P2_DIAG_QUEUE_CONTEXT,
                meta={"prev_queue_tail": prev_queue_tail, "post_queue_tail": post_queue_tail},
            )
        return result

    return wrapped


@contextmanager
def _install_write_trace_hooks(video_stem: str):
    """write_trace 計装を一時的に有効化する (with を抜けると必ず元実装へ復元)。

    Step3 (2026-07-25) 追記: MatchStartDiagRecorder (reset呼び出し/
    match_active_started/is_active反転) も同時に計装し、
    `(recorder, matchstart_diag)` のタプルで yield する。
    """
    trace_ctx = _TraceCtx()
    recorder = WriteTraceRecorder()
    matchstart_diag = MatchStartDiagRecorder()
    prev_holder: dict = {"started_frame": -1, "is_active": None}
    video_stem_holder = {"stem": video_stem}

    orig_merge = bsm._merge_diff_only
    orig_gate = bsm._apply_stable_recovery_gate
    orig_infer = rp.infer_placement
    orig_resolve = rp.resolve_after_placement
    orig_step_side = rp.RecognitionPipeline._step_side
    orig_reset = bsm.BoardStateMachine.reset
    orig_update = rp.RecognitionPipeline.update

    bsm._merge_diff_only = _make_merge_diff_only_wrapper(orig_merge, trace_ctx, recorder)
    bsm._apply_stable_recovery_gate = _make_stable_recovery_gate_wrapper(
        orig_gate, trace_ctx, recorder,
    )
    rp.infer_placement = _make_infer_placement_wrapper(orig_infer, trace_ctx, recorder)
    rp.resolve_after_placement = _make_resolve_after_placement_wrapper(
        orig_resolve, trace_ctx, recorder,
    )
    rp.RecognitionPipeline._step_side = _make_step_side_wrapper(
        orig_step_side, trace_ctx, recorder, video_stem_holder,
    )
    bsm.BoardStateMachine.reset = _make_reset_wrapper(orig_reset, trace_ctx, matchstart_diag)
    rp.RecognitionPipeline.update = _make_update_wrapper(
        orig_update, matchstart_diag, prev_holder,
    )
    try:
        yield recorder, matchstart_diag
    finally:
        bsm._merge_diff_only = orig_merge
        bsm._apply_stable_recovery_gate = orig_gate
        rp.infer_placement = orig_infer
        rp.resolve_after_placement = orig_resolve
        rp.RecognitionPipeline._step_side = orig_step_side
        bsm.BoardStateMachine.reset = orig_reset
        rp.RecognitionPipeline.update = orig_update


# ============================
# write_trace jsonl 出力
# ============================


def _write_trace_jsonl(records: list[WriteTraceRecord], out_path: Path) -> None:
    """1 (video, side) 分の write_trace を jsonl (適用順) で書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            line = {
                "route_id": rec.route_id, "frame_idx": rec.frame_idx,
                "t_sec": rec.t_sec, "side": rec.side, "cells": rec.cells,
                "meta": rec.meta,
            }
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


# ============================
# クロス集計: 監査器違反 × write_trace 直近書き込み経路 × 速度バケット
# ============================


def _cells_overlap(v_cells: list[tuple[int, int]], trace_cells: list[list[int]]) -> bool:
    """違反セル集合と write_trace セル集合が (row, col) 位置で重なるか。"""
    trace_positions = {(cell[0], cell[1]) for cell in trace_cells}
    return any((r, c) in trace_positions for r, c in v_cells)


def _attribute_route_for_violation(
    v: Violation, side_records: list[WriteTraceRecord],
) -> str:
    """1 件の違反に対し (prev_frame_idx, frame_idx] 区間の直近書き込み経路を返す。

    v.cells が非空 (color_flicker) ならセル位置が重なる write_trace を優先し、
    重なりが無ければ区間内の全 write_trace にフォールバックする。
    v.cells が空 (conservation_loss/gain) は区間内の全 write_trace を対象にする。
    複数候補がある場合は frame_idx が最大 (=直近) のものを採用する。

    Step2 (2026-07-25) 追記: P2_diag_queue_context は盤面書き込みを伴わない
    meta-only レコード (cells=[]) のため、候補プールから除外する。除外しないと
    conservation_loss/gain (v.cells が空で in_range 全体を候補にする分岐) が
    実際に盤面を書き換えていない診断レコードに誤帰属してしまう
    (同一 frame_idx の P2_infer_placement 実書き込みより後に追加されるため
    stable sort で attribution を奪ってしまう回帰)。
    """
    lo, hi = v.prev_frame_idx, v.frame_idx
    in_range = [
        r for r in side_records
        if lo < r.frame_idx <= hi and r.route_id != ROUTE_P2_DIAG_QUEUE_CONTEXT
    ]
    if not in_range:
        return ROUTE_UNATTRIBUTED
    if v.cells:
        overlapping = [r for r in in_range if _cells_overlap(v.cells, r.cells)]
        candidates = overlapping if overlapping else in_range
    else:
        candidates = in_range
    candidates.sort(key=lambda r: r.frame_idx)
    return candidates[-1].route_id


def _speed_bucket(v: Violation) -> str:
    """curr_t_sec - prev_t_sec による速度バケット分類。"""
    delta = v.t_sec - v.prev_t_sec
    return SPEED_BUCKET_FAST if delta < SPEED_BUCKET_THRESHOLD_SEC else SPEED_BUCKET_SLOW


def _build_crosstab(
    violations: list[Violation], records_by_side: dict[str, list[WriteTraceRecord]],
) -> dict:
    """経路別×速度バケット別の件数表を (type別内訳込みで) 構築する。"""
    table: dict[tuple[str, str, str], int] = {}
    detail_rows: list[dict] = []
    for v in violations:
        side_records = records_by_side.get(v.side, [])
        route = _attribute_route_for_violation(v, side_records)
        bucket = _speed_bucket(v)
        key = (v.type, route, bucket)
        table[key] = table.get(key, 0) + 1
        detail_rows.append({
            "type": v.type, "side": v.side, "frame_idx": v.frame_idx,
            "prev_frame_idx": v.prev_frame_idx, "t_sec": v.t_sec,
            "prev_t_sec": v.prev_t_sec, "attributed_route": route,
            "speed_bucket": bucket,
        })
    ranked = sorted(
        (
            {"type": t, "route_id": r, "speed_bucket": b, "count": n}
            for (t, r, b), n in table.items()
        ),
        key=lambda row: -row["count"],
    )
    return {"n_violations": len(violations), "by_type_route_speed": ranked, "detail_rows": detail_rows}


def _format_crosstab_text(crosstab: dict) -> str:
    """クロス集計を人間可読テキストに整形する。"""
    lines = [
        "==== write_trace クロス集計 (色フリッカ/消失 根因確定・Step1) ====",
        f"対象違反件数: {crosstab['n_violations']}",
        "--- 類型 x 経路 x 速度バケット (件数降順) ---",
    ]
    for row in crosstab["by_type_route_speed"]:
        lines.append(
            f"  type={row['type']:20s} route={row['route_id']:35s} "
            f"speed={row['speed_bucket']:14s} count={row['count']:4d}",
        )
    return "\n".join(lines)


# ============================
# P2 何手ズレ判定 (Step2)
# ============================


def _color_pair_matches(a: list[int] | None, b: list[int] | None) -> bool:
    """2 色ペアが (回転による順序違いを無視して) 一致するか。"""
    if a is None or b is None:
        return False
    return set(a) == set(b)


def _classify_p2_offset(
    falling_pair: list[int] | None, queue_ctx: WriteTraceRecord | None,
) -> str:
    """P2 の falling_pair が queue tail のどの位置と一致するかを判定する。

    参照位置:
      ref0 (0手/今回ツモ) = prev_queue_tail[-2]  (従来ロジック falling_pair_old と同一位置)
      ref1 (1手先)        = prev_queue_tail[-1]  (遷移直前時点で最新の NEXT 読み)
      ref2 (2手先)        = post_queue_tail[-1]  (遷移完了後にさらに進んだ NEXT 読み)
    queue_ctx が無い (= TSUMO_FALL→STABLE 遷移を経ない P2 呼び出し、例: route B
    prev_state!=TSUMO_FALL 経路) 場合は突合不能として区別する。
    """
    if queue_ctx is None:
        return OFFSET_LABEL_NO_CONTEXT
    prev_tail = queue_ctx.meta.get("prev_queue_tail") or []
    post_tail = queue_ctx.meta.get("post_queue_tail") or []
    ref0 = prev_tail[-2] if len(prev_tail) >= 2 else None
    ref1 = prev_tail[-1] if len(prev_tail) >= 1 else None
    ref2 = post_tail[-1] if len(post_tail) >= 1 else None
    if _color_pair_matches(falling_pair, ref0):
        return OFFSET_LABEL_ZERO
    if _color_pair_matches(falling_pair, ref1):
        return OFFSET_LABEL_ONE
    if _color_pair_matches(falling_pair, ref2):
        return OFFSET_LABEL_TWO
    return OFFSET_LABEL_UNMATCHED


def _build_p2_offset_table(records_by_side: dict[str, list[WriteTraceRecord]]) -> dict:
    """P2_infer_placement の falling_pair と queue tail をズレ幅別に集計する。

    (side, frame_idx) で P2_diag_queue_context と突合する
    (両者は同一 _step_side 呼び出し内で同一 frame_idx/side を持つため一意)。
    対象は全 P2 書き込み (違反有無を問わない、色フリッカ違反への絞り込みは
    crosstab 側の attributed_route で別途可能)。
    """
    counts: dict[str, int] = {}
    detail_rows: list[dict] = []
    for side, side_records in records_by_side.items():
        queue_ctx_by_frame: dict[int, WriteTraceRecord] = {
            rec.frame_idx: rec
            for rec in side_records if rec.route_id == ROUTE_P2_DIAG_QUEUE_CONTEXT
        }
        for rec in side_records:
            if rec.route_id != ROUTE_P2_INFER_PLACEMENT:
                continue
            falling_pair = rec.meta.get("falling_pair")
            if falling_pair is None:
                continue
            queue_ctx = queue_ctx_by_frame.get(rec.frame_idx)
            label = _classify_p2_offset(falling_pair, queue_ctx)
            counts[label] = counts.get(label, 0) + 1
            detail_rows.append({
                "side": side, "frame_idx": rec.frame_idx, "t_sec": rec.t_sec,
                "falling_pair": falling_pair, "offset_label": label,
                "prev_queue_tail": queue_ctx.meta.get("prev_queue_tail") if queue_ctx else None,
                "post_queue_tail": queue_ctx.meta.get("post_queue_tail") if queue_ctx else None,
            })
    return {"counts": counts, "detail_rows": detail_rows}


def _format_offset_table_text(offset_table: dict) -> str:
    """ズレ幅内訳表を人間可読テキストに整形する。"""
    total = len(offset_table["detail_rows"])
    lines = [
        "==== P2 falling_pair 何手ズレ判定 (Step2) ====",
        f"対象 P2 書き込み件数 (falling_pair 有り): {total}",
    ]
    denom = total or 1
    for label, n in sorted(offset_table["counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {label:35s} count={n:4d} ({100.0 * n / denom:5.1f}%)")
    return "\n".join(lines)


# ============================
# 試合開始直後 確定遅延診断 (Step3) 集計・出力
# ============================

# 遅延帯 (アーキ調査で確認済の一斉反映フレーム帯 1P≈frame14527/2P≈14530 を
# 含む秒区間、user指定)。
DELAY_WINDOW_START_SEC: float = 472.0
DELAY_WINDOW_END_SEC: float = 485.0


def _filter_window(events: list[dict], start_sec: float, end_sec: float) -> list[dict]:
    """t_sec が [start_sec, end_sec] に入るイベントのみ抽出する。"""
    return [e for e in events if start_sec <= e["t_sec"] <= end_sec]


def _count_by_kind(events: list[dict]) -> dict[str, int]:
    """reset_events の kind 別件数を数える。"""
    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return counts


def _build_matchstart_diag_summary(
    diag: MatchStartDiagRecorder, start_sec: float, max_sec: float,
) -> dict:
    """reset/match_start/is_active 反転の全窓+遅延帯別集計を構築する。"""
    end_sec = start_sec + max_sec
    # 走査窓と遅延帯定数の共通区間 (走査窓が遅延帯と重ならない動画/窓では
    # start > end の空区間になり、件数は自然に 0 になる)。
    delay_start = max(DELAY_WINDOW_START_SEC, start_sec)
    delay_end = min(DELAY_WINDOW_END_SEC, end_sec)
    delay_reset = _filter_window(diag.reset_events, delay_start, delay_end)
    delay_start_ev = _filter_window(diag.match_start_events, delay_start, delay_end)
    delay_flip = _filter_window(diag.is_active_flips, delay_start, delay_end)
    return {
        "scan_window": {"start_sec": start_sec, "end_sec": end_sec},
        "delay_window": {"start_sec": delay_start, "end_sec": delay_end},
        "full_window": {
            "n_reset_events": len(diag.reset_events),
            "reset_kind_counts": _count_by_kind(diag.reset_events),
            "n_match_start_events": len(diag.match_start_events),
            "n_is_active_flips": len(diag.is_active_flips),
        },
        "delay_window_only": {
            "n_reset_events": len(delay_reset),
            "reset_kind_counts": _count_by_kind(delay_reset),
            "n_match_start_events": len(delay_start_ev),
            "n_is_active_flips": len(delay_flip),
        },
        "reset_events": diag.reset_events,
        "match_start_events": diag.match_start_events,
        "is_active_flips": diag.is_active_flips,
    }


def _format_matchstart_diag_text(summary: dict) -> str:
    """人間可読テキストに整形する。"""
    fw, dw = summary["full_window"], summary["delay_window_only"]
    lines = [
        "==== 試合開始直後 確定遅延診断 (Step3, 2026-07-25) ====",
        f"走査窓: {summary['scan_window']['start_sec']:.1f}s-{summary['scan_window']['end_sec']:.1f}s",
        f"遅延帯: {summary['delay_window']['start_sec']:.1f}s-{summary['delay_window']['end_sec']:.1f}s",
        "--- 全窓 ---",
        f"  reset呼び出し 計{fw['n_reset_events']}件 内訳={fw['reset_kind_counts']}",
        f"  match_start再発火(is_active再起動) {fw['n_match_start_events']}件",
        f"  is_active反転 {fw['n_is_active_flips']}件",
        "--- 遅延帯のみ ---",
        f"  reset呼び出し 計{dw['n_reset_events']}件 内訳={dw['reset_kind_counts']}",
        f"  match_start再発火(is_active再起動) {dw['n_match_start_events']}件",
        f"  is_active反転 {dw['n_is_active_flips']}件",
    ]
    for label, key in (
        ("reset イベント詳細 (全窓)", "reset_events"),
        ("match_start再発火 詳細 (全窓)", "match_start_events"),
        ("is_active反転 詳細 (全窓)", "is_active_flips"),
    ):
        if summary[key]:
            lines.append(f"--- {label} ---")
            for e in summary[key]:
                lines.append(f"  {e}")
    return "\n".join(lines)


def _write_matchstart_diag_outputs(
    diag: MatchStartDiagRecorder, out_stem: str, start_sec: float, max_sec: float,
) -> None:
    """Step3 集計を json/txt で出力する。"""
    summary = _build_matchstart_diag_summary(diag, start_sec, max_sec)
    (OUTPUT_DIR / f"{out_stem}_matchstart_diag.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    text = _format_matchstart_diag_text(summary)
    (OUTPUT_DIR / f"{out_stem}_matchstart_diag.txt").write_text(text, encoding="utf-8")
    print(text)


# ============================
# 1 動画分の実行 (Step1 本体)
# ============================


def _run_one_video(
    video_stem: str, start_sec: float, max_sec: float,
    *,
    force_in_match: bool = True,
    enable_landing_observed_color: bool = False,
    output_stem: str | None = None,
) -> None:
    """1 動画・1 窓分を計装付きで処理し、write_trace + クロス集計を出力する。

    force_in_match / enable_landing_observed_color: 2026-07-25 試合開始直後
    確定遅延診断 (Step3) 用に追加。既定 True/False = 従来通り
    (_capture_frames の既定と一致、bit-identical)。
    force_in_match=False + enable_landing_observed_color=True で
    scripts/_diag_placement_confirm_frames_2026-07-25.py の精密計測構成
    (force_in_match は load_default 既定 False) と一致する
    (ただし stable_frame_count は本関数側の 3 のまま変更しない、後述参照)。
    output_stem: 出力ファイル名の stem 上書き (None なら video_stem)。
    従来構成の write_trace (例: c34_1P.jsonl) を上書きせず区別するために使う。
    """
    out_stem = output_stem if output_stem is not None else video_stem
    print(
        f"[{time.strftime('%H:%M:%S')}] [{video_stem}] write_trace計装 開始 "
        f"start={start_sec:.1f}s dur={max_sec:.1f}s "
        f"force_in_match={force_in_match} "
        f"enable_landing_observed_color={enable_landing_observed_color} "
        f"output_stem={out_stem}", flush=True,
    )
    t0 = time.time()
    with _install_write_trace_hooks(video_stem) as (recorder, matchstart_diag):
        by_side = _capture_frames(
            video_stem, start_sec, max_sec,
            enable_landing_observed_color=enable_landing_observed_color,
            force_in_match=force_in_match,
        )
    print(
        f"[{time.strftime('%H:%M:%S')}] [{video_stem}] 処理完了 ({time.time() - t0:.1f}s) "
        f"write_trace記録 {len(recorder.records)} 件 "
        f"reset呼び出し {len(matchstart_diag.reset_events)} 件 "
        f"match_start再発火 {len(matchstart_diag.match_start_events)} 件 "
        f"is_active反転 {len(matchstart_diag.is_active_flips)} 件", flush=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_by_side: dict[str, list[WriteTraceRecord]] = {"1P": [], "2P": []}
    for rec in recorder.records:
        records_by_side.setdefault(rec.side, []).append(rec)
    for side, side_records in records_by_side.items():
        out_path = OUTPUT_DIR / f"{out_stem}_{side}.jsonl"
        _write_trace_jsonl(side_records, out_path)
        print(f"  [{side}] write_trace {len(side_records)} 件 → {out_path}")

    _write_matchstart_diag_outputs(matchstart_diag, out_stem, start_sec, max_sec)

    # 監査器本体 (physics_violation_audit.main) と同じ試合外/演出テロップ除外を
    # 適用してから違反検出する (masking無しだと監査器出力と件数がずれるため、
    # クロス集計の分母は必ず監査器本体の実出力と一致させる)。
    telop_intervals = _scan_telop_exclusion_intervals(video_stem, start_sec, max_sec)
    violations: list[Violation] = []
    for side in ("1P", "2P"):
        recs: list[_FrameRecord] = _mask_records_by_excluded_intervals(
            by_side.get(side, []), telop_intervals,
        )
        if not recs:
            continue
        violations += _check_color_flicker(recs, video_stem, side)
        violations += _check_conservation(recs, video_stem, side)

    crosstab = _build_crosstab(violations, records_by_side)
    (OUTPUT_DIR / f"{out_stem}_crosstab_summary.json").write_text(
        json.dumps(crosstab, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    text = _format_crosstab_text(crosstab)
    (OUTPUT_DIR / f"{out_stem}_crosstab_summary.txt").write_text(text, encoding="utf-8")
    print(text)

    # Step2: P2 falling_pair 何手ズレ判定 (A2 仮説の確証用)。
    offset_table = _build_p2_offset_table(records_by_side)
    (OUTPUT_DIR / f"{out_stem}_p2_offset_summary.json").write_text(
        json.dumps(offset_table, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    offset_text = _format_offset_table_text(offset_table)
    (OUTPUT_DIR / f"{out_stem}_p2_offset_summary.txt").write_text(offset_text, encoding="utf-8")
    print(offset_text)


# ============================
# CLI / main
# ============================


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする。"""
    ap = argparse.ArgumentParser(description="confirmed_board 書き込み元トレース診断 (Step1)")
    ap.add_argument(
        "--smoke", action="store_true",
        help=f"スモークモード: video_{SMOKE_VIDEO_STEM} の短窓のみ同期実行する。",
    )
    ap.add_argument("--video", type=str, default=None, help="対象動画stem (--smoke指定時は無視)。")
    ap.add_argument("--start-sec", type=float, default=None, dest="start_sec")
    ap.add_argument("--max-sec", type=float, default=None, dest="max_sec")
    # 試合開始直後 確定遅延診断 (Step3, 2026-07-25) 用。既定は従来通り
    # (force_in_match=True, enable_landing_observed_color=False) で
    # bit-identical、明示指定時のみ挙動変更。
    ap.add_argument(
        "--force-in-match", dest="force_in_match", action=argparse.BooleanOptionalAction,
        default=True,
        help="既定True(従来通り)。--no-force-in-match で "
             "MatchStateDetector実判定を使い試合開始直後保護機構を実際に発動させる。",
    )
    ap.add_argument(
        "--enable-landing-observed-color", dest="enable_landing_observed_color",
        action="store_true", default=False,
        help="既定False(従来通り)。指定時は着地セル CNN==HSV 一致色補正を有効化する"
             "(_diag_placement_confirm_frames_2026-07-25.py と同一構成にする際に指定)。",
    )
    ap.add_argument(
        "--output-stem", dest="output_stem", type=str, default=None,
        help="出力ファイル名 stem 上書き(既定None=video_stemと同一)。"
             "従来構成の write_trace を上書きしたくない場合に指定する。",
    )
    return ap.parse_args()


def main() -> None:
    """メイン処理: 対象窓を計装付きで処理し、write_trace + クロス集計を出力する。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない
    args = _parse_args()
    if args.smoke or args.video is None:
        stem, start_sec, max_sec = SMOKE_VIDEO_STEM, SMOKE_START_SEC, SMOKE_MAX_SEC
    else:
        stem = args.video
        start_sec = args.start_sec if args.start_sec is not None else 0.0
        max_sec = args.max_sec if args.max_sec is not None else 60.0
    _run_one_video(
        stem, start_sec, max_sec,
        force_in_match=args.force_in_match,
        enable_landing_observed_color=args.enable_landing_observed_color,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
