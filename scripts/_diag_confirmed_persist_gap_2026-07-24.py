"""真因診断: 却下セルの「STABLE 中の永続」を座標レベルで実証する (2026-07-24)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない。
新規ファイルのみ、labeled_win/boards への書込みもしない。

## 依頼背景 (アーキ調査による見立て)
NON-STABLE→STABLE 復帰時に F ガード (empty_to_color、
DEFAULT_EMPTY_TO_COLOR_MIN_VOTES=3、src/board_state_machine.py:41) が設置セルを
却下すると、STABLE 中はセル単位再修正が無いまま却下が「永続」し、数手後の連鎖が
その欠損盤面で計算される、という placement dropout (#47) の真因仮説を座標レベルで
実測する。

## 重要な事前訂正 (本スクリプト作成時にコード読解で確定、依頼文の前提と異なる)
依頼文は「事後復旧ゲート (enable_stable_recovery_gate) は既定 OFF」としているが、
実際には **RecognitionPipeline.load_default() の default は True** (2026-06-02
user viz 採用承認により変更済、src/recognition_pipeline.py:606,1437)。
BoardStateMachine 自体の内部 default (src/board_state_machine.py:430) は
False のままだが、RecognitionPipeline 層が明示的に True を渡すため配線は
完全に生きている (_build_state_machine 経由、src/recognition_pipeline.py:
1030,1041)。つまり **本番デフォルト設定は既に復旧ゲート ON** である。
本スクリプトの計測4 (A/B) は「デフォルト ON (True)」と「明示 OFF (False、
= 旧仕様相当)」の比較として構成する。

## 却下セルの発生機構 (コード読解で確定)
NON-STABLE→STABLE 遷移は `_apply_transition` (board_state_machine.py:520-570)
で処理され、直前 state が NON_STABLE_STATES (TSUMO_FALL/CHAIN/OJAMA_FALL/
EFFECT/GRAVITY_SETTLE) の場合のみ `_merge_diff_only` で F ガードが適用される。
non_stable_cnn_history は state 切替毎にリセットされる (同ファイル:561) ため、
F ガードに寄与する履歴は「STABLE 復帰直前の 1 区間のみ」。GRAVITY_SETTLE 区間は
history 蓄積を行わない早期 return (同ファイル:583-585) のため empty_guard が
None になり F ガードは発火しない (=GRAVITY_SETTLE→STABLE は無条件採用)。
遷移フレーム自体では復旧ゲートは発火しない (`_apply_transition` 内では
recovery gate 呼び出しなし、STABLE_WARMUP_FRAMES も enable_warmup_guard
default False のため即 0)。よって **遷移フレームの実 confirmed_board は
F ガード merge の結果そのもの** であり、本スクリプトはこれを手動再現せず
直接読み取る (却下セル判定の真実性を最大化する設計、
_diag_ojama_fall_board_settle_ab_2026-07-24.py の設計方針を踏襲)。

## 手法 (既存 2 資産をマージ、重複ロジックの再実装を避ける)
importlib で以下をそのまま再利用する (ファイル名にハイフンを含み通常の
import 文が使えないため):
    scripts/_diag_ojama_fall_exit_timing_2026-07-24.py:
        TARGET_WINDOWS, _video_path, _seek_frame (区間検出・pixels viz手法)
    scripts/_diag_place_coldstart_dropout_2026-07-24.py:
        PRE_TRIGGER_LOOKBACK_FRAMES, _INVALID_COLORS, _new_chain_trigger_idxs,
        _roi_for_side, _crop_roi, CROP_MARGIN_PX (chain trigger 解析手法)
non_stable_cnn_history の再現、EMPTY→色 却下セル分類、STABLE 中の永続追跡、
severity 座標逆引き、復旧ゲート A/B は本スクリプト独自実装 (既存 2 資産は
「OJAMA_FALL 区間限定」「count のみ (座標なし)」であり、本タスクの座標レベル
永続実証には汎用化・拡張が必須なため)。

## 計測
1. NON-STABLE→STABLE 復帰時に F ガードが却下したセル座標集合
   (層2=history<3 / 層3=votes<3 / conflict_other=それ以外)。
2. 却下セル毎に、次の NON-STABLE 遷移までの STABLE 滞在フレーム数と、
   その間 CNN が却下色を観測し続けた継続率・「CNNは見えているのに
   confirmed は空のまま」フレーム数 (= 永続の直接証拠)。
3. 既存 placement_dropout 診断の severity_confirmed 事例と同じ判定基準
   (ChainSimulator 補正で chain_count が変化) を本スクリプト内で再計算し、
   missing_cells 座標が計測1の却下セルの持続の延長線上にあるかを逆引き。
4. enable_stable_recovery_gate=True (default) vs False の A/B。
   同一区間を 2 回走査し、却下セル数・永続フレーム数分布を比較する。

## 制約
- read-only診断 (src/は改変しない)。enable_stable_recovery_gateは既存フラグを
  そのまま True/False で渡すだけ、新規実装はしない。
- 熱対策: cv2.setNumThreads(1)、並列しない。
- 本走行禁止: --smoke で 1動画・短窓のみ処理する。

Usage (本走行、WSL 経由、CLAUDE.md プロセス管理ルール準拠。呼び出し元 = main Claude):
    wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \\
      setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python \\
      scripts/_diag_confirmed_persist_gap_2026-07-24.py \\
      > logs/confirmed_persist_gap_2026-07-24.log 2>&1 < /dev/null'"

Usage (スモーク、短窓 1 件・動作確認用):
    PYTHONPATH=. ./venv/bin/python \\
      scripts/_diag_confirmed_persist_gap_2026-07-24.py --smoke
"""
from __future__ import annotations

import argparse
import importlib.util
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
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS,
)
from src.board_state_machine import (  # noqa: E402
    BoardState,
    DEFAULT_EMPTY_TO_COLOR_MIN_VOTES,
    DEFAULT_NON_STABLE_HISTORY_SIZE,
    NON_STABLE_STATES,
)
from src.chain import ChainSimulator  # noqa: E402
from src.placement_inferrer import enumerate_landing_patterns  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
import src.recognition_pipeline as _rp_module  # noqa: E402  monkeypatch対象module参照
from scripts.visualize_recognition import (  # noqa: E402
    CELL_H, CELL_W, draw_cell_overlay,
)

# ============================
# 2026-07-24 追加: infer_placement フック (候補(d)診断)
#
# #47設置ドロップの二重機構実証: F ガード (_merge_diff_only) が却下したセルの
# うち、TSUMO_FALL起因のものについて placement_inferrer.infer_placement が
# 「呼ばれて None (commit refuse)」 だったのか「そもそも呼ばれなかった」 のか
# 「呼ばれて成功したが何らかの理由で最終 confirmed には反映されなかった」 のかを
# 実証する。src/ 本体は一切変更せず、本診断プロセス内で module 属性を
# 一時差し替え (monkeypatch) して呼出引数/返り値を観測するのみ。
# ============================

# ============================
# 既存 2 資産の動的 import (再利用、読み取り専用利用のみ)
# ============================
_EXIT_TIMING_PATH = PROJ_ROOT / "scripts" / "_diag_ojama_fall_exit_timing_2026-07-24.py"
_spec_a = importlib.util.spec_from_file_location(
    "_diag_ojama_fall_exit_timing_reuse2", _EXIT_TIMING_PATH,
)
assert _spec_a is not None and _spec_a.loader is not None
_exit_timing_mod = importlib.util.module_from_spec(_spec_a)
sys.modules[_spec_a.name] = _exit_timing_mod
_spec_a.loader.exec_module(_exit_timing_mod)  # 定義のみ実行 (main() は未実行)

_video_path = _exit_timing_mod._video_path
_seek_frame = _exit_timing_mod._seek_frame
FULL_TARGET_WINDOWS: tuple = _exit_timing_mod.TARGET_WINDOWS

_COLDSTART_PATH = PROJ_ROOT / "scripts" / "_diag_place_coldstart_dropout_2026-07-24.py"
_spec_b = importlib.util.spec_from_file_location(
    "_diag_place_coldstart_dropout_reuse2", _COLDSTART_PATH,
)
assert _spec_b is not None and _spec_b.loader is not None
_coldstart_mod = importlib.util.module_from_spec(_spec_b)
sys.modules[_spec_b.name] = _coldstart_mod
_spec_b.loader.exec_module(_coldstart_mod)

_new_chain_trigger_idxs = _coldstart_mod._new_chain_trigger_idxs
_roi_for_side = _coldstart_mod._roi_for_side
_crop_roi = _coldstart_mod._crop_roi
PRE_TRIGGER_LOOKBACK_FRAMES: int = _coldstart_mod.PRE_TRIGGER_LOOKBACK_FRAMES
_INVALID_COLORS: tuple = _coldstart_mod._INVALID_COLORS

# ============================
# 定数
# ============================

# NON_STABLE_STATES の name 集合 (F ガードが有効な直前 state の判定用)。
_NON_STABLE_STATE_NAMES: frozenset[str] = frozenset(s.name for s in NON_STABLE_STATES)

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "confirmed_persist_gap_2026-07-24"

# スモーク専用窓: video 30、既存診断で baseline 確保済の開始点 225.0 を流用。
SMOKE_TARGET_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("30", 225.0, 260.0, "スモーク専用: video30 冒頭35秒、baseline確保済の開始点225.0を流用"),
)

# viz を出す severity 確定事例の上限数 (動画毎)。
MAX_SEVERITY_VIZ_PER_VIDEO: int = 4

PROGRESS_LOG_INTERVAL_FRAMES: int = 1800

# missing cell マーカー円の半径オフセット (セル幅からの引き算、視認性確保用)。
_MARKER_RADIUS_MARGIN_PX: int = 4
_MARKER_THICKNESS_PX: int = 3

# TSUMO_FALL起因却下セルの infer_placement 呼出結果分類 (候補(d)診断)。
_INFER_STATUS_NONE: str = "infer_none"          # 呼ばれたが None (commit refuse)
_INFER_STATUS_NOT_CALLED: str = "infer_not_called"  # この frame/side で未呼出
_INFER_STATUS_OTHER: str = "infer_other"        # 呼ばれて成功したが最終confirmedは空のまま
_INFER_STATUS_NA: str = "n_a"                   # 非TSUMO_FALL起因 (対象外)


def _print_progress(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 frame・1 side 分の記録 (measurement1-3 全てに必要な最小集合)。"""

    frame_idx: int
    t: float
    state: str
    cnn_board: Board
    confirmed_board: "Board | None"
    chain_trigger_sec: "float | None"
    chain_before_grid: "np.ndarray | None"
    chain_count_event: "int | None"


@dataclass
class _RejectedCell:
    """計測1: 1 遷移・1 セル分の却下記録。"""

    video: str
    side: str
    exit_idx: int  # records 内 list-index (同一 records 内でのみ有効)
    exit_t: float
    row: int
    col: int
    rejected_color: int
    layer: str  # "layer2_insufficient_time" / "layer3_color_noise" / "conflict_other"
    history_len: int
    cell_votes: int
    prev_state: str  # 遷移直前 state (TSUMO_FALL/OJAMA_FALL/GRAVITY_SETTLE/EFFECT/CHAIN)
    # 以下2026-07-24候補(d)診断追加分 (prev_state=TSUMO_FALL 以外は n_a/-1固定):
    infer_status: str  # _INFER_STATUS_* のいずれか
    n_landing_patterns: int  # baseline盤面での物理配置パターン総数 (1なら一意)


@dataclass
class _PersistRecord:
    """計測2: 却下セル 1 つの STABLE 滞在中の永続度。"""

    video: str
    side: str
    exit_idx: int
    exit_t: float
    row: int
    col: int
    rejected_color: int
    layer: str
    stable_run_frames: int
    frames_cnn_present_confirmed_empty: int  # 永続の直接証拠フレーム数
    recovered: bool
    recovered_offset_frames: "int | None"
    cnn_present_rate: float


@dataclass
class _SeverityCase:
    """計測3: severity_confirmed 座標逆引きの確定事例。"""

    video: str
    side: str
    trigger_frame_idx: int
    trigger_t: float
    row: int
    col: int
    rejected_color: int
    layer: str
    causal_exit_frame_idx: int
    causal_exit_t: float
    chain_count_before: int
    chain_count_corrected: int
    before_grid: "np.ndarray | None" = field(repr=False, compare=False, default=None)
    confirmed_at_reject_grid: "np.ndarray | None" = field(
        repr=False, compare=False, default=None,
    )


@dataclass
class _InferCallRecord:
    """1回の infer_placement 呼出記録 (候補(d)診断フック、src非改変)。"""

    frame_idx: int
    side: str
    returned_none: bool


class _InferPlacementHook:
    """recognition_pipeline.infer_placement を実行時差し替えして記録する診断フック。

    src/ ファイルは一切変更しない。プロセス内 module 属性の一時差し替えのみで、
    install()/uninstall() で必ず元に戻す (呼出元 _collect_records が保証)。
    side判定は infer_placement 呼出時の region kwarg と
    DEFAULT_P1_REGION/DEFAULT_P2_REGION の同一性 (is 比較) で行う
    (recognition_pipeline.py:3866-3869,4121-4124 で side別singleton定数を渡す実装に依拠)。
    """

    def __init__(self) -> None:
        self.calls: list[_InferCallRecord] = []
        self._current_frame_idx: int = -1
        self._original = _rp_module.infer_placement

    def set_current_frame(self, frame_idx: int) -> None:
        """呼出元 (_collect_records) が frame 処理直前に現在フレーム番号を通知する。"""
        self._current_frame_idx = frame_idx

    def _wrapped(self, *args: object, **kwargs: object) -> object:
        result = self._original(*args, **kwargs)
        region = kwargs.get("region")
        if region is _rp_module.DEFAULT_P1_REGION:
            side = "1P"
        elif region is _rp_module.DEFAULT_P2_REGION:
            side = "2P"
        else:
            side = "unknown"
        self.calls.append(_InferCallRecord(
            frame_idx=self._current_frame_idx, side=side,
            returned_none=(result is None),
        ))
        return result

    def install(self) -> None:
        _rp_module.infer_placement = self._wrapped  # type: ignore[assignment]

    def uninstall(self) -> None:
        _rp_module.infer_placement = self._original  # type: ignore[assignment]


# ============================
# パス1: pipeline 走査 (enable_stable_recovery_gate 指定可)
# ============================


def _collect_records(
    video_stem: str, start_sec: float, end_sec: float, enable_recovery_gate: bool,
    infer_hook: "_InferPlacementHook | None" = None,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    """video を走査し、1P/2P それぞれの frame 記録を返す (指定 gate 構成)。

    infer_hook: 非None時、frame処理毎に infer_placement 呼出を記録する
    (候補(d)診断用、2026-07-24追加)。install/uninstallは本関数内で完結させ、
    呼出元にmonkeypatchの後始末を委ねない (read-only診断原則の徹底)。
    """
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = RecognitionPipeline.load_default(enable_stable_recovery_gate=enable_recovery_gate)
    pipe.set_video_id(video_stem)

    recs_1p: list[_FrameRec] = []
    recs_2p: list[_FrameRec] = []
    fi = start_frame
    n_read = 0
    tag = "ON" if enable_recovery_gate else "OFF"
    if infer_hook is not None:
        infer_hook.install()
    try:
        while fi < end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            t = fi / fps
            if infer_hook is not None:
                infer_hook.set_current_frame(fi)
            r = pipe.update(fi, t, frame)
            for side_recs, side_res in ((recs_1p, r.p1), (recs_2p, r.p2)):
                ce = side_res.chain_event
                side_recs.append(_FrameRec(
                    frame_idx=fi, t=t, state=side_res.state.name,
                    cnn_board=side_res.cnn_board.copy(),
                    confirmed_board=(
                        side_res.confirmed_board.copy()
                        if side_res.confirmed_board is not None else None
                    ),
                    chain_trigger_sec=(float(ce.trigger_sec) if ce is not None else None),
                    chain_before_grid=(ce.before_board._grid.copy() if ce is not None else None),
                    chain_count_event=(int(ce.chain_count) if ce is not None else None),
                ))
            fi += 1
            n_read += 1
            if n_read % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                _print_progress(
                    f"[{video_stem}/{tag}] t={t:.1f}s まで処理済み ({n_read} frames)",
                )
    finally:
        if infer_hook is not None:
            infer_hook.uninstall()
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# 計測1: NON-STABLE→STABLE 遷移・却下セル
# ============================


def _find_transitions_to_stable(records: list[_FrameRec]) -> list[tuple[int, int, str]]:
    """NON-STABLE→STABLE 遷移一覧: (segment_start_idx, exit_idx, preceding_state_name)。

    segment_start_idx は「直前の同一 state 連続区間の開始 index」。F ガードの
    non_stable_cnn_history はこの区間のみに由来する (state 切替毎にリセットされる、
    board_state_machine.py:561)。ウィンドウ先頭で既に STABLE の場合は除外。
    """
    out: list[tuple[int, int, str]] = []
    for i in range(1, len(records)):
        prev_state = records[i - 1].state
        if prev_state == "STABLE" or records[i].state != "STABLE":
            continue
        seg_start = i - 1
        while seg_start > 0 and records[seg_start - 1].state == prev_state:
            seg_start -= 1
        out.append((seg_start, i, prev_state))
    return out


def _find_baseline_confirmed(records: list[_FrameRec], seg_start_idx: int) -> "Board | None":
    """遷移直前の最後の confirmed_board (= F ガード merge の baseline)。"""
    for i in range(seg_start_idx - 1, -1, -1):
        if records[i].confirmed_board is not None:
            return records[i].confirmed_board
    return None


def _reconstruct_capped_history(
    records: list[_FrameRec], seg_start_idx: int, exit_idx: int, preceding_state: str,
) -> list[Board]:
    """F ガード非STABLE履歴の再現 (board_state_machine._update_within_current_state 準拠)。

    GRAVITY_SETTLE 区間は history 蓄積を行わない早期 return (同ファイル:583-585) の
    ため、常に空を返す (= empty_guard=None = F ガード不発火、と実コード同一挙動)。
    """
    if preceding_state == BoardState.GRAVITY_SETTLE.name:
        return []
    raw = [records[i].cnn_board for i in range(seg_start_idx + 1, exit_idx)]
    return raw[-DEFAULT_NON_STABLE_HISTORY_SIZE:]


def _classify_rejected_cells(
    records: list[_FrameRec], baseline: Board, exit_idx: int, capped_history: list[Board],
) -> list[tuple[int, int, int, str, int, int]]:
    """却下セルを層別分類する (実 confirmed_board との直接比較、手動merge再現はしない)。

    却下セル条件: baseline(r,c)=EMPTY かつ 遷移frameのcnn_board(r,c)=有効色
    かつ 遷移frameの実confirmed_board(r,c)=EMPTY のまま (=F ガードが実際に却下)。
    層2 (history<3) / 層3 (votes<3) / conflict_other (それ以外、多数決が別色)。

    Returns:
        [(row, col, rejected_color, layer, history_len, cell_votes), ...]
    """
    exit_rec = records[exit_idx]
    confirmed_exit = exit_rec.confirmed_board
    if confirmed_exit is None:
        return []
    history_len = len(capped_history)
    out: list[tuple[int, int, int, str, int, int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if baseline.get(r, c) != COLOR_EMPTY:
                continue
            cnn_v = int(exit_rec.cnn_board.get(r, c))
            if cnn_v in (COLOR_EMPTY, COLOR_UNKNOWN):
                continue
            if int(confirmed_exit.get(r, c)) != COLOR_EMPTY:
                continue  # 却下されていない (F ガード通過済み)
            if history_len < DEFAULT_EMPTY_TO_COLOR_MIN_VOTES:
                out.append((r, c, cnn_v, "layer2_insufficient_time", history_len, 0))
                continue
            cell_votes = sum(1 for b in capped_history if int(b.get(r, c)) == cnn_v)
            layer = (
                "layer3_color_noise" if cell_votes < DEFAULT_EMPTY_TO_COLOR_MIN_VOTES
                else "conflict_other"
            )
            out.append((r, c, cnn_v, layer, history_len, cell_votes))
    return out


# ============================
# 計測2: STABLE 中の永続追跡
# ============================


def _track_persistence(
    records: list[_FrameRec], exit_idx: int, run_end_idx: int,
    row: int, col: int, rejected_color: int,
) -> tuple[int, int, bool, "int | None", float]:
    """却下セル 1 つの STABLE 滞在区間全体を追跡する。

    Returns:
        (persist_count, run_len, recovered, recovered_offset, cnn_present_rate)
        persist_count: confirmed=EMPTY かつ cnn=却下色 のフレーム数 (永続の直接証拠)。
        cnn_present_rate: run全体でCNNが却下色を観測し続けた比率 (recovered後も含む)。
    """
    run_len = run_end_idx - exit_idx
    if run_len <= 0:
        return 0, 0, False, None, 0.0
    persist_count = 0
    cnn_present_count = 0
    recovered = False
    recovered_offset: "int | None" = None
    for offset in range(run_len):
        rec = records[exit_idx + offset]
        confirmed_v = (
            int(rec.confirmed_board.get(row, col))
            if rec.confirmed_board is not None else COLOR_EMPTY
        )
        cnn_v = int(rec.cnn_board.get(row, col))
        if cnn_v == rejected_color:
            cnn_present_count += 1
        if confirmed_v == COLOR_EMPTY:
            if cnn_v == rejected_color:
                persist_count += 1
        elif not recovered:
            recovered = True
            recovered_offset = offset
    return persist_count, run_len, recovered, recovered_offset, cnn_present_count / run_len


def _classify_infer_status(
    prev_state: str, side: str, exit_frame_idx: int, baseline: Board,
    infer_calls_by_key: "dict[tuple[int, str], _InferCallRecord] | None",
) -> tuple[str, int]:
    """TSUMO_FALL起因却下遷移について、infer_placement呼出結果と物理パターン数を判定する。

    候補(d) (物理のみ強制フォールバック) の効きうる範囲診断の核心ロジック。
    非TSUMO_FALL起因 (OJAMA_FALL/GRAVITY_SETTLE/EFFECT/CHAIN) は infer_placement
    自体が呼ばれる経路にないため常に n_a を返す (recognition_pipeline.py:3811 の
    prev_state==TSUMO_FALL 条件に一致させる)。

    Returns:
        (infer_status, n_landing_patterns)。
    """
    n_patterns = len(enumerate_landing_patterns(baseline))
    if prev_state != BoardState.TSUMO_FALL.name or infer_calls_by_key is None:
        return _INFER_STATUS_NA, n_patterns
    rec = infer_calls_by_key.get((exit_frame_idx, side))
    if rec is None:
        return _INFER_STATUS_NOT_CALLED, n_patterns
    if rec.returned_none:
        return _INFER_STATUS_NONE, n_patterns
    return _INFER_STATUS_OTHER, n_patterns


def _run_measurement_1_2(
    video: str, recs_1p: list[_FrameRec], recs_2p: list[_FrameRec],
    infer_calls_by_key: "dict[tuple[int, str], _InferCallRecord] | None" = None,
) -> tuple[list[_RejectedCell], list[_PersistRecord]]:
    """計測1+2 を 1P/2P 両サイドで実行する。

    infer_calls_by_key: 非None時、TSUMO_FALL起因却下セルに infer_status/
    n_landing_patterns を付与する (候補(d)診断、2026-07-24追加)。
    """
    rejected: list[_RejectedCell] = []
    persist: list[_PersistRecord] = []
    for side, records in (("1P", recs_1p), ("2P", recs_2p)):
        transitions = _find_transitions_to_stable(records)
        for i, (seg_start, exit_idx, prev_state) in enumerate(transitions):
            if prev_state not in _NON_STABLE_STATE_NAMES:
                continue  # MENU→STABLE 等 F ガード対象外
            baseline = _find_baseline_confirmed(records, seg_start)
            if baseline is None:
                continue
            capped_history = _reconstruct_capped_history(records, seg_start, exit_idx, prev_state)
            cell_tuples = _classify_rejected_cells(records, baseline, exit_idx, capped_history)
            if not cell_tuples:
                continue
            run_end = transitions[i + 1][0] if i + 1 < len(transitions) else len(records)
            exit_t = records[exit_idx].t
            exit_frame_idx = records[exit_idx].frame_idx
            infer_status, n_patterns = _classify_infer_status(
                prev_state, side, exit_frame_idx, baseline, infer_calls_by_key,
            )
            for (r, c, color, layer, hist_len, votes) in cell_tuples:
                rejected.append(_RejectedCell(
                    video=video, side=side, exit_idx=exit_idx, exit_t=exit_t,
                    row=r, col=c, rejected_color=color, layer=layer,
                    history_len=hist_len, cell_votes=votes,
                    prev_state=prev_state, infer_status=infer_status,
                    n_landing_patterns=n_patterns,
                ))
                p_count, run_len, recov, recov_off, cnn_rate = _track_persistence(
                    records, exit_idx, run_end, r, c, color,
                )
                persist.append(_PersistRecord(
                    video=video, side=side, exit_idx=exit_idx, exit_t=exit_t,
                    row=r, col=c, rejected_color=color, layer=layer,
                    stable_run_frames=run_len,
                    frames_cnn_present_confirmed_empty=p_count,
                    recovered=recov, recovered_offset_frames=recov_off,
                    cnn_present_rate=cnn_rate,
                ))
    return rejected, persist


# ============================
# 計測3: severity 座標逆引き
# ============================


def _missing_cell_coords(
    before_grid: np.ndarray, lookback_grid: np.ndarray,
) -> list[tuple[int, int]]:
    """before=EMPTY かつ lookback=有効色 の座標一覧。

    _diag_place_coldstart_dropout_2026-07-24._missing_cells と同一判定条件の
    座標版 (元関数は count のみ返すため、座標逆引きに必要な本関数を追加する)。
    """
    out: list[tuple[int, int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            bv = int(before_grid[r, c])
            lv = int(lookback_grid[r, c])
            if bv == COLOR_EMPTY and lv not in _INVALID_COLORS:
                out.append((r, c))
    return out


def _is_severity_trigger(
    sim: ChainSimulator, before_grid: np.ndarray, lookback_grid: np.ndarray,
    missing_coords: list[tuple[int, int]],
) -> tuple[bool, int, int]:
    """missing_cells 補正で chain_count が変化するか (severity_confirmed 判定)。"""
    corrected = before_grid.copy()
    for (r, c) in missing_coords:
        corrected[r, c] = lookback_grid[r, c]
    pred_before = sim.simulate(Board.from_list(before_grid.tolist()))
    pred_corrected = sim.simulate(Board.from_list(corrected.tolist()))
    return (
        pred_before.chain_count != pred_corrected.chain_count,
        pred_before.chain_count, pred_corrected.chain_count,
    )


def _find_causal_transition(
    persist_records: list[_PersistRecord], video: str, side: str,
    row: int, col: int, trigger_idx: int,
) -> "_PersistRecord | None":
    """missing cell が trigger_idx 時点でどの却下遷移の持続の延長線上にあるかを逆引き。

    exit_idx<=trigger_idx かつ (未回復 または 回復が trigger より後) の
    最新の _PersistRecord を返す (= 持続が trigger まで途切れなかった事例)。
    """
    candidates = [
        p for p in persist_records
        if p.video == video and p.side == side and p.row == row and p.col == col
        and p.exit_idx <= trigger_idx
        and (
            p.recovered_offset_frames is None
            or p.exit_idx + p.recovered_offset_frames > trigger_idx
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.exit_idx)


def _run_measurement_3(
    video: str, recs_1p: list[_FrameRec], recs_2p: list[_FrameRec],
    persist: list[_PersistRecord], sim: ChainSimulator,
) -> list[_SeverityCase]:
    """計測3: 連鎖トリガーの missing_cells を計測1/2の持続記録に逆引きする。"""
    out: list[_SeverityCase] = []
    for side, records in (("1P", recs_1p), ("2P", recs_2p)):
        for idx in _new_chain_trigger_idxs(records):
            rec = records[idx]
            before_grid = rec.chain_before_grid
            if before_grid is None:
                continue
            lookback_idx = max(0, idx - PRE_TRIGGER_LOOKBACK_FRAMES)
            lookback_grid = records[lookback_idx].cnn_board._grid
            missing_coords = _missing_cell_coords(before_grid, lookback_grid)
            if not missing_coords:
                continue
            is_sev, cnt_before, cnt_corrected = _is_severity_trigger(
                sim, before_grid, lookback_grid, missing_coords,
            )
            if not is_sev:
                continue
            for (r, c) in missing_coords:
                causal = _find_causal_transition(persist, video, side, r, c, idx)
                if causal is None:
                    continue
                causal_rec = records[causal.exit_idx]
                out.append(_SeverityCase(
                    video=video, side=side,
                    trigger_frame_idx=rec.frame_idx, trigger_t=rec.t,
                    row=r, col=c, rejected_color=causal.rejected_color, layer=causal.layer,
                    causal_exit_frame_idx=causal_rec.frame_idx, causal_exit_t=causal_rec.t,
                    chain_count_before=cnt_before, chain_count_corrected=cnt_corrected,
                    before_grid=before_grid.copy(),
                    confirmed_at_reject_grid=(
                        causal_rec.confirmed_board._grid.copy()
                        if causal_rec.confirmed_board is not None else None
                    ),
                ))
    return out


# ============================
# viz
# ============================


def _mark_cell(frame: np.ndarray, roi_x: int, roi_y: int, row: int, col: int) -> None:
    """missing cell を目立たせるマーカー円を重畳する (隠し段は draw_cell_overlay と同様非表示)。"""
    if row < HIDDEN_ROWS:
        return
    display_row = row - HIDDEN_ROWS
    cx = roi_x + col * CELL_W + CELL_W // 2
    cy = roi_y + display_row * CELL_H + CELL_H // 2
    radius = CELL_W // 2 - _MARKER_RADIUS_MARGIN_PX
    cv2.circle(frame, (cx, cy), radius, (255, 0, 255), _MARKER_THICKNESS_PX, cv2.LINE_AA)


def _render_severity_viz(video_stem: str, case: _SeverityCase, out_path: Path) -> None:
    """3コマ montage: 却下時実画面(生) / 却下時confirmed overlay / 連鎖起点still missing。"""
    cap = cv2.VideoCapture(str(_video_path(video_stem)))
    roi_x, roi_y = _roi_for_side(case.side)
    frame_raw = _seek_frame(cap, case.causal_exit_frame_idx)
    frame_reject = _seek_frame(cap, case.causal_exit_frame_idx)
    frame_trigger = _seek_frame(cap, case.trigger_frame_idx)
    cap.release()
    if frame_raw is None or frame_reject is None or frame_trigger is None:
        return

    cv2.putText(
        frame_raw, f"却下時実画面(生) t={case.causal_exit_t:.2f}s", (roi_x, roi_y - 8),
        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
    )
    confirmed_reject = (
        Board.from_list(case.confirmed_at_reject_grid.tolist())
        if case.confirmed_at_reject_grid is not None else Board()
    )
    draw_cell_overlay(frame_reject, confirmed_reject, roi_x, roi_y)
    _mark_cell(frame_reject, roi_x, roi_y, case.row, case.col)
    cv2.putText(
        frame_reject, f"却下時confirmed missing({case.row},{case.col})", (roi_x, roi_y - 8),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
    )
    before_board = (
        Board.from_list(case.before_grid.tolist()) if case.before_grid is not None else Board()
    )
    draw_cell_overlay(frame_trigger, before_board, roi_x, roi_y)
    _mark_cell(frame_trigger, roi_x, roi_y, case.row, case.col)
    cv2.putText(
        frame_trigger,
        f"連鎖起点still missing chain={case.chain_count_before}->{case.chain_count_corrected}",
        (roi_x, roi_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
    )

    crops = [_crop_roi(f, roi_x, roi_y) for f in (frame_raw, frame_reject, frame_trigger)]
    h = max(c.shape[0] for c in crops)
    sep = np.full((h, 6, 3), (255, 255, 255), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for c in crops:
        parts.append(c)
        parts.append(sep)
    out = np.hstack(parts[:-1])
    cv2.imwrite(str(out_path), out)


def _write_ab_histogram(
    video_stem: str, persist_on: list[_PersistRecord], persist_off: list[_PersistRecord],
    out_path: Path,
) -> None:
    """A/B (復旧ゲート ON/OFF) の永続フレーム数分布ヒストグラム。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals_on = [p.frames_cnn_present_confirmed_empty for p in persist_on]
    vals_off = [p.frames_cnn_present_confirmed_empty for p in persist_off]
    max_val = max([1] + vals_on + vals_off)
    bins = np.arange(0, max_val + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals_off, bins=bins, alpha=0.5, label=f"OFF (n={len(vals_off)})", color="#d62728")
    ax.hist(vals_on, bins=bins, alpha=0.5, label=f"ON=default (n={len(vals_on)})", color="#1f77b4")
    ax.set_xlabel("永続フレーム数 (CNNは有色観測、confirmedはEMPTYのまま)")
    ax.set_ylabel("却下セル数")
    ax.set_title(f"{video_stem}: enable_stable_recovery_gate ON/OFF 永続フレーム数分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================
# 集計 + CSV/summary 出力
# ============================


def _stats_for_persist(persist: list[_PersistRecord]) -> dict:
    vals = [p.frames_cnn_present_confirmed_empty for p in persist]
    n_recovered = sum(1 for p in persist if p.recovered)
    return {
        "n_rejected_cells": len(persist),
        "n_recovered": n_recovered,
        "recovered_rate": (n_recovered / len(persist) if persist else None),
        "persist_frames_mean": (float(np.mean(vals)) if vals else None),
        "persist_frames_median": (float(np.median(vals)) if vals else None),
        "persist_frames_max": (max(vals) if vals else None),
    }


def _build_ab_summary(
    persist_on: list[_PersistRecord], persist_off: list[_PersistRecord],
) -> dict:
    return {"gate_on_default": _stats_for_persist(persist_on), "gate_off": _stats_for_persist(persist_off)}


def _write_rejected_csv(rejected: list[_RejectedCell], out_path: Path) -> None:
    cols = [
        "video", "side", "exit_idx", "exit_t", "row", "col", "rejected_color",
        "layer", "history_len", "cell_votes", "prev_state", "infer_status",
        "n_landing_patterns",
    ]
    lines = [",".join(cols)]
    for r in rejected:
        lines.append(
            f"{r.video},{r.side},{r.exit_idx},{r.exit_t:.3f},{r.row},{r.col},"
            f"{r.rejected_color},{r.layer},{r.history_len},{r.cell_votes},"
            f"{r.prev_state},{r.infer_status},{r.n_landing_patterns}",
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_prev_state_breakdown(rejected: list[_RejectedCell]) -> dict[str, int]:
    """prev_state別の却下セル数集計 (計測1をF ガード対象state全体で分解)。"""
    out: dict[str, int] = {}
    for r in rejected:
        out[r.prev_state] = out.get(r.prev_state, 0) + 1
    return out


def _build_infer_status_breakdown(rejected: list[_RejectedCell]) -> dict[str, int]:
    """TSUMO_FALL起因却下セルの infer_placement 呼出結果内訳 (候補(d)診断)。"""
    out: dict[str, int] = {}
    for r in rejected:
        if r.prev_state != BoardState.TSUMO_FALL.name:
            continue
        out[r.infer_status] = out.get(r.infer_status, 0) + 1
    return out


def _estimate_candidate_d_scope(rejected: list[_RejectedCell]) -> dict:
    """候補(d) (物理のみ強制フォールバック) が効きうる範囲を推定する。

    条件: TSUMO_FALL起因 かつ infer_placement=None かつ baseline盤面での
    物理配置パターンが一意 (n_landing_patterns==1、= 曖昧性なしで確定可能)。
    セル数 (row,col単位) と transition数 (1着地=最大2セル、二重計上防止用) の
    両方を返す。
    """
    hits = [
        r for r in rejected
        if r.prev_state == BoardState.TSUMO_FALL.name
        and r.infer_status == _INFER_STATUS_NONE
        and r.n_landing_patterns == 1
    ]
    transitions = {(r.video, r.side, r.exit_idx) for r in hits}
    return {"n_cells": len(hits), "n_transitions": len(transitions)}


def _write_persist_csv(persist: list[_PersistRecord], out_path: Path) -> None:
    cols = [
        "video", "side", "exit_idx", "exit_t", "row", "col", "rejected_color", "layer",
        "stable_run_frames", "frames_cnn_present_confirmed_empty", "recovered",
        "recovered_offset_frames", "cnn_present_rate",
    ]
    lines = [",".join(cols)]
    for p in persist:
        lines.append(
            f"{p.video},{p.side},{p.exit_idx},{p.exit_t:.3f},{p.row},{p.col},"
            f"{p.rejected_color},{p.layer},{p.stable_run_frames},"
            f"{p.frames_cnn_present_confirmed_empty},{p.recovered},"
            f"{'' if p.recovered_offset_frames is None else p.recovered_offset_frames},"
            f"{p.cnn_present_rate:.4f}",
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_severity_csv(severity: list[_SeverityCase], out_path: Path) -> None:
    cols = [
        "video", "side", "trigger_frame_idx", "trigger_t", "row", "col",
        "rejected_color", "layer", "causal_exit_frame_idx", "causal_exit_t",
        "chain_count_before", "chain_count_corrected",
    ]
    lines = [",".join(cols)]
    for s in severity:
        lines.append(
            f"{s.video},{s.side},{s.trigger_frame_idx},{s.trigger_t:.3f},{s.row},{s.col},"
            f"{s.rejected_color},{s.layer},{s.causal_exit_frame_idx},{s.causal_exit_t:.3f},"
            f"{s.chain_count_before},{s.chain_count_corrected}",
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _format_summary_text(overall: dict, per_video: dict[str, dict]) -> str:
    lines = [
        "==== confirmed_board 却下セル永続実証 サマリ (2026-07-24) ====",
        f"却下セル総数(全動画・ON=default): {overall['n_rejected_cells_total']}",
        f"  層2(時間不足)={overall['n_layer2_total']} "
        f"層3(色ノイズ)={overall['n_layer3_total']} "
        f"conflict_other={overall['n_conflict_other_total']}",
        f"永続フレーム数(ON=default) 平均/中央値/最大: "
        f"{overall['persist_frames_mean']}/{overall['persist_frames_median']}"
        f"/{overall['persist_frames_max']}",
        f"復旧ゲートによる回復率(ON=default): {overall['recovered_rate']}",
        f"severity座標逆引き確定事例数 (持続が実害を起こした確定事例): "
        f"{overall['n_severity_cases_total']}",
        "--- prev_state別 却下セル数 (F ガード対象state全体) ---",
    ]
    for state_name, cnt in sorted(overall["prev_state_breakdown"].items()):
        lines.append(f"  {state_name}: {cnt}")
    lines.append(
        "--- TSUMO_FALL起因却下セルの infer_placement 呼出結果内訳 (候補(d)診断) ---",
    )
    for status_name, cnt in sorted(overall["infer_status_breakdown_tsumo_fall"].items()):
        lines.append(f"  {status_name}: {cnt}")
    lines.append(
        f"候補(d)が効きうる範囲 (infer_placement=None かつ 物理パターン一意): "
        f"セル数={overall['candidate_d_scope']['n_cells']} "
        f"transition数={overall['candidate_d_scope']['n_transitions']}",
    )
    lines.append("--- A/B (enable_stable_recovery_gate ON=default vs OFF) 動画別 ---")
    for video, s in per_video.items():
        ab = s["ab_summary"]
        lines.append(
            f"  {video}: 却下セル ON={ab['gate_on_default']['n_rejected_cells']} "
            f"OFF={ab['gate_off']['n_rejected_cells']} / "
            f"永続平均 ON={ab['gate_on_default']['persist_frames_mean']} "
            f"OFF={ab['gate_off']['persist_frames_mean']} / "
            f"回復率 ON={ab['gate_on_default']['recovered_rate']} "
            f"OFF={ab['gate_off']['recovered_rate']} / "
            f"severity事例={s['n_severity_cases']}",
        )
    return "\n".join(lines)


def _build_overall_summary(
    all_rejected: list[_RejectedCell], all_persist: list[_PersistRecord],
    all_severity: list[_SeverityCase],
) -> dict:
    stats = _stats_for_persist(all_persist)
    return {
        "n_rejected_cells_total": len(all_rejected),
        "n_layer2_total": sum(1 for r in all_rejected if r.layer == "layer2_insufficient_time"),
        "n_layer3_total": sum(1 for r in all_rejected if r.layer == "layer3_color_noise"),
        "n_conflict_other_total": sum(1 for r in all_rejected if r.layer == "conflict_other"),
        "persist_frames_mean": stats["persist_frames_mean"],
        "persist_frames_median": stats["persist_frames_median"],
        "persist_frames_max": stats["persist_frames_max"],
        "recovered_rate": stats["recovered_rate"],
        "n_severity_cases_total": len(all_severity),
        "prev_state_breakdown": _build_prev_state_breakdown(all_rejected),
        "infer_status_breakdown_tsumo_fall": _build_infer_status_breakdown(all_rejected),
        "candidate_d_scope": _estimate_candidate_d_scope(all_rejected),
    }


# ============================
# メイン
# ============================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="短窓1件のみ処理 (動作確認用、本走行ではない)",
    )
    return parser.parse_args()


def _process_video(
    video_stem: str, start_sec: float, end_sec: float, note: str,
) -> tuple[list[_RejectedCell], list[_PersistRecord], list[_SeverityCase], dict]:
    """1動画分の計測1〜4を実行する (ON=default パスで計測1-3、OFFも追加してA/B)。

    候補(d)診断: ON パスのみ _InferPlacementHook を仕込み、infer_placement 呼出結果
    (frame_idx, side) -> _InferCallRecord の辞書を計測1に渡す。OFF パスは
    候補(d)診断の対象外 (A/B対照はF ガードの永続分布のみ比較すれば十分) のため
    フック非装着で従来通り (計算コスト削減、read-only原則にも合致)。
    """
    _print_progress(f"[{video_stem}] 開始 (gate=ON=default) window={start_sec:.1f}-{end_sec:.1f}s ({note})")
    t0 = time.time()
    infer_hook = _InferPlacementHook()
    recs_1p_on, recs_2p_on, fps = _collect_records(
        video_stem, start_sec, end_sec, True, infer_hook=infer_hook,
    )
    _print_progress(
        f"[{video_stem}] ON pass 完了 ({len(recs_1p_on)} frame, {time.time() - t0:.1f}s, "
        f"infer_placement呼出={len(infer_hook.calls)}件)",
    )
    infer_calls_by_key = {(c.frame_idx, c.side): c for c in infer_hook.calls}
    rejected_v, persist_v = _run_measurement_1_2(
        video_stem, recs_1p_on, recs_2p_on, infer_calls_by_key,
    )
    sim = ChainSimulator()
    severity_v = _run_measurement_3(video_stem, recs_1p_on, recs_2p_on, persist_v, sim)
    _print_progress(
        f"[{video_stem}] 計測1-3完了: 却下{len(rejected_v)}件 severity{len(severity_v)}件",
    )

    _print_progress(f"[{video_stem}] 開始 (gate=OFF, A/B対照) ...")
    t1 = time.time()
    recs_1p_off, recs_2p_off, _fps_off = _collect_records(video_stem, start_sec, end_sec, False)
    _print_progress(
        f"[{video_stem}] OFF pass 完了 ({len(recs_1p_off)} frame, {time.time() - t1:.1f}s)",
    )
    _rejected_off, persist_off = _run_measurement_1_2(video_stem, recs_1p_off, recs_2p_off)
    ab_summary = _build_ab_summary(persist_v, persist_off)

    video_summary = {
        "note": note, "window": [start_sec, end_sec], "fps": fps,
        "n_rejected_cells": len(rejected_v), "n_severity_cases": len(severity_v),
        "ab_summary": ab_summary,
        "prev_state_breakdown": _build_prev_state_breakdown(rejected_v),
        "infer_status_breakdown_tsumo_fall": _build_infer_status_breakdown(rejected_v),
        "candidate_d_scope": _estimate_candidate_d_scope(rejected_v),
    }
    _write_ab_histogram(video_stem, persist_v, persist_off, OUTPUT_DIR / f"ab_histogram_{video_stem}.png")
    return rejected_v, persist_v, severity_v, video_summary


def main() -> None:
    args = _parse_args()
    smoke = args.smoke or os.environ.get("PUYO_CONFIRMED_PERSIST_SMOKE") == "1"
    windows = SMOKE_TARGET_WINDOWS if smoke else FULL_TARGET_WINDOWS
    if smoke:
        _print_progress("[SMOKE MODE] 短窓1件のみ処理します (本走行ではありません)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rejected: list[_RejectedCell] = []
    all_persist: list[_PersistRecord] = []
    all_severity: list[_SeverityCase] = []
    per_video_summary: dict[str, dict] = {}

    for video_stem, start_sec, end_sec, note in windows:
        rejected_v, persist_v, severity_v, video_summary = _process_video(
            video_stem, start_sec, end_sec, note,
        )
        all_rejected.extend(rejected_v)
        all_persist.extend(persist_v)
        all_severity.extend(severity_v)
        per_video_summary[video_stem] = video_summary

        _write_rejected_csv(rejected_v, OUTPUT_DIR / f"rejected_{video_stem}.csv")
        _write_persist_csv(persist_v, OUTPUT_DIR / f"persist_{video_stem}.csv")
        _write_severity_csv(severity_v, OUTPUT_DIR / f"severity_{video_stem}.csv")

        viz_targets = severity_v[:MAX_SEVERITY_VIZ_PER_VIDEO]
        for case in viz_targets:
            label = f"{case.trigger_t:.2f}".replace(".", "_")
            _render_severity_viz(
                video_stem, case,
                OUTPUT_DIR / f"viz_severity_{video_stem}_{case.side}_t{label}.png",
            )
        _print_progress(f"[{video_stem}] severity viz {len(viz_targets)}件 出力完了")

    overall = _build_overall_summary(all_rejected, all_persist, all_severity)
    summary = {"overall": overall, "per_video": per_video_summary, "smoke_mode": smoke}
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    text = _format_summary_text(overall, per_video_summary)
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


if __name__ == "__main__":
    main()
