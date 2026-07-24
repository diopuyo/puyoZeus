"""おじゃまドロップ修正 (案B+(a)+(b)) 効果検証 A/B スクリプト (2026-07-24)。

完全 read-only 診断スクリプト。src/ は一切変更しない。
新規ファイルのみ、labeled_win/boards への書込みもしない。

## 背景
真因診断 (scripts/_diag_ojama_fall_exit_timing_2026-07-24.py) で OJAMA_FALL
滞在時間が中央値 1 frame (0.0167s) と早すぎ、却下おじゃまセル 651 個全数が
「時間不足」由来と確定した。以下 3 点の修正を実装済 (savepoint、既定 False):
    - 案B `enable_ojama_fall_board_settle`: OJAMA_FALL 退出条件を「盤面全体の
      ぷよ数が静止するまで待つ」方式 (GravitySettle と同型) に置換。
    - (a) `enable_gravity_filter_support`: 案B適用後に判明した副作用対処。
      積もり中のおじゃまが `_apply_gravity_filter` に「浮きぷよ」誤消去
      されるのを防ぐ (F ガード board を support_board として渡す)。
    - (b) `merge_use_majority_value`: 退出 merge の単一フレームちらつき
      による却下を、多数決値書込みで解消。
タグ付け診断で却下セルの 85% (video30窓) が OJAMA_FALL 起因と確認済で、
上記 3 点がその本丸を狙う (2026-07-24 拡張、旧版は enable_ojama_fall_board_settle
単独の A/B のみだった)。

本スクリプトは同一 4 動画・同一窓を **OFF (3flag全部False=現行既定) vs
ON (3flag全部True)** の 2 構成で処理し、以下を数値+実画面フレームで出す:
    1. 却下→採用の改善 (却下おじゃまセル数 OFF vs ON、採用セル数 OFF vs ON)
    2. 浮き誤消去件数 (`_apply_gravity_filter` 起因の消去) OFF vs ON
       (計装ラップで実測、副次的に confirmed_board のおじゃまセル数変化も記録)
    3. OJAMA_FALL 滞在時間: OFF vs ON、ON側は settle-exit / timeout-exit の
       行使率ヒストグラム
    4. TSUMO_FALL 検出遅延 (回帰確認): OJAMA_FALL 退出後、次 TSUMO_FALL
       初検出までの時間 OFF vs ON
    5. 実画面フレーム A/B viz: footprint おじゃま最多の代表区間で
       OFF確定盤面 / ON確定盤面 / 実ゲーム画面 の 3 コマ montage

## 区間マッチング方式 (2026-07-24 拡張)
旧版は OFF/ON 対応付けに records の index 番号をそのまま比較していたが、
OFF/ON 構成でパイプライン挙動 (state 遷移タイミング) が変わっても
records リストは同一 frame 範囲を同一ステップで読むため index は本来
揃うはずだが、脆弱性回避のため **時刻 (`.t`) ベースの区間重複判定**
(`_match_off_segment_by_time`) に明示的に変更した (index 由来の不一致を
過去に踏んだ教訓、feedback_recognition_regression_prevention 準拠)。

## 既存資産の再利用 (方針: 重複ロジックを増やさない)
scripts/_diag_ojama_fall_exit_timing_2026-07-24.py はファイル名にハイフンを
含み通常の import 文が使えないため、importlib で動的 import し、以下の
「純粋関数・窓定義」をそのまま再利用する (計算式の重複実装をしない):
    TARGET_WINDOWS (窓定義そのまま), _FrameRec (frame 記録データ構造),
    _video_path, _find_ojama_fall_segments, _lookahead_bound,
    _natural_settle_rel_idx, _mode_board, _find_baseline, _seek_frame
「却下/採用セル数」は本スクリプト独自に、旧診断の手動 history 再現
(_reconstruct_history_and_confirmed) ではなく **実際にパイプラインが出した
confirmed_board そのもの** (records[i].confirmed_board) を比較に使う。
理由: RecognitionPipeline.load_default() は enable_stable_recovery_gate 等
旧診断が再現していない追加ゲートを多数持つため、手動再現値は近似に過ぎない。
OFF/ON 効果測定という本スクリプトの目的には実際の confirmed_board 差分の方が
真実性が高い。

## 制約
- 熱対策: cv2.setNumThreads(1)。並列はしない (呼び出し元が制御)。
- 本走行 (フル 4 動画) は別途 main Claude が detach 実行する。
  本スクリプト単体では --smoke 指定時のみ短窓 1 件で動作確認する。

Usage (本走行、WSL 経由、CLAUDE.md プロセス管理ルール準拠):
    wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \\
      setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python \\
      scripts/_diag_ojama_fall_board_settle_ab_2026-07-24.py \\
      > logs/ojama_fall_board_settle_ab_2026-07-24.log 2>&1 < /dev/null'"

Usage (スモーク、短窓 1 件・動作確認用):
    PYTHONPATH=. ./venv/bin/python \\
      scripts/_diag_ojama_fall_board_settle_ab_2026-07-24.py --smoke
    (または環境変数 PUYO_OJAMA_AB_SMOKE=1 でも同じ挙動になる)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import src.board_state_machine as board_state_machine_module  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_OJAMA  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_visual_detector import (  # noqa: E402
    OJAMA_FALL_MAX_SEC,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.visualize_recognition import (  # noqa: E402
    P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y, ROI_H, ROI_W, draw_cell_overlay,
)

# ============================
# 旧診断スクリプトの動的 import (再利用、ファイル名にハイフンを含むため
# 通常の import 文が使えない。読み取り専用利用のみ、既存ファイルは変更しない)
# ============================
_OLD_DIAG_PATH = PROJ_ROOT / "scripts" / "_diag_ojama_fall_exit_timing_2026-07-24.py"
_spec = importlib.util.spec_from_file_location(
    "_diag_ojama_fall_exit_timing_reuse", _OLD_DIAG_PATH,
)
assert _spec is not None and _spec.loader is not None
_old_diag = importlib.util.module_from_spec(_spec)
# dataclass の型解決 (sys.modules.get(cls.__module__)) が module 登録前提のため、
# exec_module 前に sys.modules へ登録しておく必要がある (未登録だと AttributeError)。
sys.modules[_spec.name] = _old_diag
_spec.loader.exec_module(_old_diag)  # 定義のみ実行 (main() は __name__ ガード済で走らない)

_FrameRec = _old_diag._FrameRec
_video_path = _old_diag._video_path
_find_ojama_fall_segments = _old_diag._find_ojama_fall_segments
_lookahead_bound = _old_diag._lookahead_bound
_natural_settle_rel_idx = _old_diag._natural_settle_rel_idx
_mode_board = _old_diag._mode_board
_find_baseline = _old_diag._find_baseline
_seek_frame = _old_diag._seek_frame
FULL_TARGET_WINDOWS: tuple = _old_diag.TARGET_WINDOWS

# ============================
# 定数
# ============================

OUTPUT_DIR: Path = (
    PROJ_ROOT / "data" / "verify" / "ojama_dropout_fix_ab_2026-07-24"
)

# スモーク専用窓: video 30 の最初の OJAMA_FALL 区間 (1P, entry_t=243.017s)。
# 旧診断の 225.0-315.0s 走行で baseline_missing=False が実測確認済 (events_30.csv
# 行2) のため、同じ開始点 225.0 を使い baseline 欠如リスクを排除する。
# 終了 246.0s は entry(243.017) から ON 側タイムアウト上限 (1.5s) + 余裕を確保。
SMOKE_TARGET_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("30", 225.0, 246.0,
     "スモーク専用: video30 最初のOJAMA_FALL区間(1P entry~243.017s)1件検証、"
     "旧診断run実測でbaseline確保済の開始点225.0を流用"),
)

# ON側 exit 種別判定の許容誤差 (フレーム境界の丸め誤差吸収、マジックナンバー回避)。
EXIT_KIND_TIMEOUT_EPSILON_FRAMES: float = 0.5

# viz を出す代表例の上限数 (動画毎、footprint おじゃま量降順)。
MAX_VIZ_PER_VIDEO_AB: int = 2

# 進捗ログの frame 間隔。
PROGRESS_LOG_INTERVAL_FRAMES: int = 1800


@dataclass(frozen=True)
class _ABFlagSet:
    """OFF/ON 構成の 3 フラグ集合 (案B+(a)+(b) 同時切替、2026-07-24 拡張)。

    3 フラグを独立検証するのではなく、user 指定通り「まとめて OFF」
    「まとめて ON」の 2 構成のみを比較する (単一フラグ寄与の分解は対象外)。
    """

    enable_ojama_fall_board_settle: bool
    enable_gravity_filter_support: bool
    merge_use_majority_value: bool


# OFF = 現行本番既定 (baseline)。ON = 3修正まとめて有効化。
OFF_FLAGS: _ABFlagSet = _ABFlagSet(False, False, False)
ON_FLAGS: _ABFlagSet = _ABFlagSet(True, True, True)


def _print_progress(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# ============================
# 浮き誤消去 (_apply_gravity_filter 起因) 計装カウント
# ============================


@dataclass
class _GravityEraseCounts:
    """`_apply_gravity_filter` 呼び出し全体で実際に消去された cell 数の内訳。"""

    total_erased: int = 0
    ojama_erased: int = 0


def _wrap_gravity_filter_for_counting(orig_fn, counts: _GravityEraseCounts):
    """`_apply_gravity_filter` を計装ラップし、消去 cell 数を counts に集計する。

    src/board_state_machine.py 自体は変更しない (read-only 制約遵守)。
    モジュール属性の一時差し替えのみで、呼び出し側が必ず finally で復元する。
    """
    def _wrapped(board: Board, *, support_board: "Board | None" = None) -> None:
        before = [
            [board.get(r, c) for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)
        ]
        orig_fn(board, support_board=support_board)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                prev_v = before[r][c]
                if prev_v != COLOR_EMPTY and board.get(r, c) == COLOR_EMPTY:
                    counts.total_erased += 1
                    if prev_v == COLOR_OJAMA:
                        counts.ojama_erased += 1
    return _wrapped


@contextmanager
def _count_gravity_erasures() -> Iterator[_GravityEraseCounts]:
    """with 節内の `_apply_gravity_filter` 呼び出しを計装カウントする。

    with を抜けると必ず元の関数に復元する (src/ を実質変更しない一時パッチ)。
    """
    counts = _GravityEraseCounts()
    orig_fn = board_state_machine_module._apply_gravity_filter
    board_state_machine_module._apply_gravity_filter = (
        _wrap_gravity_filter_for_counting(orig_fn, counts)
    )
    try:
        yield counts
    finally:
        board_state_machine_module._apply_gravity_filter = orig_fn


# ============================
# パス1: OFF/ON 2 構成でパイプライン走査
# ============================


def _collect_records_flagged(
    video_stem: str, start_sec: float, end_sec: float, flags: _ABFlagSet,
) -> tuple[list, list, float, _GravityEraseCounts]:
    """video を走査し、1P/2P それぞれの frame 記録 + 浮き誤消去数を返す (3flag指定構成)。"""
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = RecognitionPipeline.load_default(
        enable_ojama_fall_board_settle=flags.enable_ojama_fall_board_settle,
        enable_gravity_filter_support=flags.enable_gravity_filter_support,
        merge_use_majority_value=flags.merge_use_majority_value,
    )
    pipe.set_video_id(video_stem)

    recs_1p: list = []
    recs_2p: list = []
    fi = start_frame
    n_read = 0
    tag = "ON" if flags.enable_ojama_fall_board_settle else "OFF"
    with _count_gravity_erasures() as gravity_counts:
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
            if n_read % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] [{video_stem}/{tag}] t={t:.1f}s まで処理済み "
                    f"({n_read} frames)", flush=True,
                )
    cap.release()
    return recs_1p, recs_2p, fps, gravity_counts


# ============================
# baseline/footprint 計算 (旧診断 _build_event の board 版、Board オブジェクトを返す)
# ============================


def _compute_baseline_footprint_and_timing(
    records: list, entry_idx: int, exit_idx: int, fps: float,
) -> tuple:
    """baseline/footprint (Board) + 自然settle計測を返す (旧診断 _build_event と同一計算式)。

    戻り値: (baseline, footprint, natural_elapsed, natural_converged, gap_sec, exceeds_cap)
    baseline が None の場合は判定不能 (呼び出し側でスキップ)。
    """
    baseline = _find_baseline(records, entry_idx)
    if baseline is None:
        return None, None, None, False, None, False

    entry_t = records[entry_idx].t
    duration_sec = records[exit_idx].t - entry_t
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
    else:
        natural_elapsed = None
        footprint_src = [
            records[i].cnn_board
            for i in range(max(entry_idx, bound_idx - 5), bound_idx)
        ]
    footprint = _mode_board(footprint_src)

    gap_sec = (natural_elapsed - duration_sec) if natural_elapsed is not None else None
    exceeds_cap = natural_elapsed is not None and natural_elapsed > OJAMA_FALL_MAX_SEC
    return baseline, footprint, natural_elapsed, settle_abs is not None, gap_sec, exceeds_cap


def _cell_diff_counts(baseline: Board, footprint: Board, actual: "Board | None") -> tuple[int, int]:
    """footprint でお邪魔と判定された新規セルのうち、actual(confirmed_board) で
    実際にお邪魔として反映されている数 (accepted) とされていない数 (rejected) を返す。"""
    accepted = 0
    rejected = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if baseline.get(r, c) != COLOR_EMPTY or footprint.get(r, c) != COLOR_OJAMA:
                continue
            if actual is not None and actual.get(r, c) == COLOR_OJAMA:
                accepted += 1
            else:
                rejected += 1
    return accepted, rejected


def _classify_on_exit_kind(duration_sec: float, fps: float) -> str:
    """ON側 OJAMA_FALL 退出が settle-exit (自然収束) か timeout-exit (安全弁) かを分類する。"""
    epsilon_sec = EXIT_KIND_TIMEOUT_EPSILON_FRAMES / fps
    if duration_sec >= OJAMA_FALL_MAX_SEC - epsilon_sec:
        return "timeout-exit"
    return "settle-exit"


def _delay_to_next_tsumo_fall(records: list, from_idx: int) -> "float | None":
    """from_idx (退出frame) 以降、最初に TSUMO_FALL state になるまでの経過秒。
    ウィンドウ内に見つからなければ None (要フル走行での再確認)。"""
    tsumo_name = BoardState.TSUMO_FALL.name
    t0 = records[from_idx].t
    for i in range(from_idx, len(records)):
        if records[i].state == tsumo_name:
            return records[i].t - t0
    return None


def _nearest_time_idx(records: list, t: float) -> int:
    """records 内で時刻 t に最も近い index を返す (窓内のみの短い list 前提、線形探索)。"""
    best_i = 0
    best_diff = abs(records[0].t - t)
    for i, rec in enumerate(records):
        diff = abs(rec.t - t)
        if diff < best_diff:
            best_diff = diff
            best_i = i
    return best_i


def _match_off_segment_by_time(
    records_off: list, baseline: Board, footprint: Board,
    on_entry_t: float, on_exit_t: float,
) -> tuple[int, bool, int, int, float, int]:
    """OFF側 OJAMA_FALL 区間を ON基準 entry/exit **時刻** の重複で対応付ける。

    2026-07-24 拡張: 旧版は records の index 番号をそのまま比較していたが、
    index 由来の不一致を避けるため時刻ベースの区間重複判定に変更した。
    戻り値: (off_exit_idx, no_matched, accepted, rejected, total_duration_sec, n_matched)
    """
    off_segments = _find_ojama_fall_segments(records_off)
    matched = [
        (s, e) for s, e in off_segments
        if records_off[s].t < on_exit_t and records_off[e].t > on_entry_t
    ]
    no_matched = len(matched) == 0
    off_exit_idx = matched[-1][1] if matched else _nearest_time_idx(records_off, on_exit_t)
    off_actual = records_off[off_exit_idx].confirmed_board
    accepted, rejected = _cell_diff_counts(baseline, footprint, off_actual)
    total_duration = sum(records_off[e].t - records_off[s].t for s, e in matched)
    return off_exit_idx, no_matched, accepted, rejected, total_duration, len(matched)


# ============================
# ABEvent データ構造
# ============================


@dataclass
class _ABEvent:
    """1 ON基準 OJAMA_FALL 区間の OFF/ON 対比結果。"""

    video: str
    side: str
    on_entry_t: float
    on_exit_t: float
    on_duration_sec: float
    on_exit_kind: str  # "settle-exit" | "timeout-exit"
    footprint_ojama_cells: int  # ON/OFF 共通の「真」おじゃまセル数 (footprint近似)
    on_accepted_cells: int
    on_rejected_cells: int
    off_n_subsegments: int  # OFF側 同一時間帯での細切れ突入回数 (分断度合い)
    off_total_duration_sec: float  # OFF側 細切れ滞在時間合計
    off_accepted_cells: int
    off_rejected_cells: int
    off_no_matched_subsegments: bool  # 異常フラグ: OFF側で対応する突入が1つも無い (時刻重複なし)
    on_exit_idx: int  # ON側 exit の records index (viz 再利用用)
    off_exit_idx: int  # OFF側 対応 exit の records index (viz 再利用用)
    natural_settle_elapsed_sec: "float | None"
    natural_settle_converged: bool
    gap_sec: "float | None"  # ON基準: natural_settle - on_duration_sec (早すぎ秒数)
    tsumo_fall_delay_on_sec: "float | None"
    tsumo_fall_delay_off_sec: "float | None"


def _build_ab_event(
    video_stem: str, side: str,
    records_off: list, records_on: list,
    on_entry_idx: int, on_exit_idx: int, fps: float,
) -> "_ABEvent | None":
    baseline, footprint, natural_elapsed, natural_converged, gap_sec, _exceeds = (
        _compute_baseline_footprint_and_timing(records_on, on_entry_idx, on_exit_idx, fps)
    )
    if baseline is None:
        return None
    footprint_count = sum(
        1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        if baseline.get(r, c) == COLOR_EMPTY and footprint.get(r, c) == COLOR_OJAMA
    )
    on_actual = records_on[on_exit_idx].confirmed_board
    on_accepted, on_rejected = _cell_diff_counts(baseline, footprint, on_actual)

    on_entry_t = records_on[on_entry_idx].t
    on_exit_t = records_on[on_exit_idx].t
    off_exit_idx, off_no_matched, off_accepted, off_rejected, off_total_duration, off_n_seg = (
        _match_off_segment_by_time(records_off, baseline, footprint, on_entry_t, on_exit_t)
    )

    on_duration_sec = on_exit_t - on_entry_t
    exit_kind = _classify_on_exit_kind(on_duration_sec, fps)

    delay_on = _delay_to_next_tsumo_fall(records_on, on_exit_idx)
    delay_off = _delay_to_next_tsumo_fall(records_off, off_exit_idx)

    return _ABEvent(
        video=video_stem, side=side, on_entry_t=on_entry_t, on_exit_t=on_exit_t,
        on_duration_sec=on_duration_sec, on_exit_kind=exit_kind,
        footprint_ojama_cells=footprint_count,
        on_accepted_cells=on_accepted, on_rejected_cells=on_rejected,
        off_n_subsegments=off_n_seg, off_total_duration_sec=off_total_duration,
        off_accepted_cells=off_accepted, off_rejected_cells=off_rejected,
        off_no_matched_subsegments=off_no_matched,
        on_exit_idx=on_exit_idx, off_exit_idx=off_exit_idx,
        natural_settle_elapsed_sec=natural_elapsed,
        natural_settle_converged=natural_converged, gap_sec=gap_sec,
        tsumo_fall_delay_on_sec=delay_on, tsumo_fall_delay_off_sec=delay_off,
    )


# ============================
# CSV / summary 出力
# ============================


def _write_ab_events_csv(events: list[_ABEvent], out_path: Path) -> None:
    cols = [
        "video", "side", "on_entry_t", "on_exit_t", "on_duration_sec", "on_exit_kind",
        "footprint_ojama_cells", "on_accepted_cells", "on_rejected_cells",
        "off_n_subsegments", "off_total_duration_sec", "off_accepted_cells",
        "off_rejected_cells", "off_no_matched_subsegments",
        "natural_settle_elapsed_sec", "natural_settle_converged", "gap_sec",
        "tsumo_fall_delay_on_sec", "tsumo_fall_delay_off_sec",
    ]
    lines = [",".join(cols)]
    for e in events:
        vals = [
            e.video, e.side, f"{e.on_entry_t:.3f}", f"{e.on_exit_t:.3f}",
            f"{e.on_duration_sec:.3f}", e.on_exit_kind, e.footprint_ojama_cells,
            e.on_accepted_cells, e.on_rejected_cells, e.off_n_subsegments,
            f"{e.off_total_duration_sec:.3f}", e.off_accepted_cells, e.off_rejected_cells,
            e.off_no_matched_subsegments,
            "" if e.natural_settle_elapsed_sec is None else f"{e.natural_settle_elapsed_sec:.3f}",
            e.natural_settle_converged,
            "" if e.gap_sec is None else f"{e.gap_sec:.3f}",
            "" if e.tsumo_fall_delay_on_sec is None else f"{e.tsumo_fall_delay_on_sec:.3f}",
            "" if e.tsumo_fall_delay_off_sec is None else f"{e.tsumo_fall_delay_off_sec:.3f}",
        ]
        lines.append(",".join(str(v) for v in vals))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_ab_summary(events: list[_ABEvent]) -> dict:
    with_footprint = [e for e in events if e.footprint_ojama_cells > 0]
    n_settle = sum(1 for e in events if e.on_exit_kind == "settle-exit")
    n_timeout = sum(1 for e in events if e.on_exit_kind == "timeout-exit")
    on_durations = [e.on_duration_sec for e in with_footprint]
    off_durations = [e.off_total_duration_sec for e in with_footprint]
    on_accepted_total = sum(e.on_accepted_cells for e in events)
    off_accepted_total = sum(e.off_accepted_cells for e in events)
    on_rejected_total = sum(e.on_rejected_cells for e in events)
    off_rejected_total = sum(e.off_rejected_cells for e in events)
    off_subseg_counts = [e.off_n_subsegments for e in events]
    delays_on = [e.tsumo_fall_delay_on_sec for e in events if e.tsumo_fall_delay_on_sec is not None]
    delays_off = [e.tsumo_fall_delay_off_sec for e in events if e.tsumo_fall_delay_off_sec is not None]
    return {
        "n_events_total": len(events),
        "n_events_with_footprint_ojama": len(with_footprint),
        "n_off_no_matched_subsegments": sum(1 for e in events if e.off_no_matched_subsegments),
        # 要件2: settle-exit / timeout-exit 行使率ヒストグラム (アーキ必須要件)。
        "n_on_exit_settle": n_settle,
        "n_on_exit_timeout": n_timeout,
        "rate_on_exit_timeout": (n_timeout / len(events)) if events else None,
        "on_duration_sec_mean": float(np.mean(on_durations)) if on_durations else None,
        "on_duration_sec_median": float(np.median(on_durations)) if on_durations else None,
        "off_duration_sec_mean": float(np.mean(off_durations)) if off_durations else None,
        "off_duration_sec_median": float(np.median(off_durations)) if off_durations else None,
        # 要件1: 却下→採用の改善。
        "on_accepted_cells_total": on_accepted_total,
        "off_accepted_cells_total": off_accepted_total,
        "on_rejected_cells_total": on_rejected_total,
        "off_rejected_cells_total": off_rejected_total,
        "rejected_cells_improvement": off_rejected_total - on_rejected_total,
        "off_n_subsegments_mean": float(np.mean(off_subseg_counts)) if off_subseg_counts else None,
        "off_n_subsegments_median": float(np.median(off_subseg_counts)) if off_subseg_counts else None,
        # 要件3: TSUMO_FALL 検出遅延 (回帰確認)。
        "tsumo_fall_delay_on_mean_sec": float(np.mean(delays_on)) if delays_on else None,
        "tsumo_fall_delay_on_median_sec": float(np.median(delays_on)) if delays_on else None,
        "tsumo_fall_delay_off_mean_sec": float(np.mean(delays_off)) if delays_off else None,
        "tsumo_fall_delay_off_median_sec": float(np.median(delays_off)) if delays_off else None,
        "n_tsumo_fall_delay_on_unobserved": len(events) - len(delays_on),
        "n_tsumo_fall_delay_off_unobserved": len(events) - len(delays_off),
    }


def _aggregate_gravity_overall(per_video: dict[str, dict]) -> dict:
    """動画別の浮き誤消去件数 (要件2) を全体集計する (`_apply_gravity_filter` 計装値)。"""
    off_ojama = sum(s["off_gravity_ojama_erased"] for s in per_video.values())
    on_ojama = sum(s["on_gravity_ojama_erased"] for s in per_video.values())
    off_total = sum(s["off_gravity_total_erased"] for s in per_video.values())
    on_total = sum(s["on_gravity_total_erased"] for s in per_video.values())
    return {
        "off_gravity_ojama_erased_total": off_ojama,
        "on_gravity_ojama_erased_total": on_ojama,
        "gravity_ojama_erased_improvement_total": off_ojama - on_ojama,
        "off_gravity_total_erased_total": off_total,
        "on_gravity_total_erased_total": on_total,
    }


def _format_ab_summary_text(overall: dict, per_video: dict[str, dict]) -> str:
    lines = [
        "==== おじゃまドロップ修正(案B+(a)+(b)) A/B 検証サマリ (2026-07-24) ====",
        f"検出 OJAMA_FALL 区間総数 (ON基準): {overall['n_events_total']} "
        f"(footprintおじゃま有り: {overall['n_events_with_footprint_ojama']}) "
        f"OFF側対応区間なし異常: {overall['n_off_no_matched_subsegments']}",
        "--- 要件2: OJAMA_FALL滞在時間 + settle/timeoutヒストグラム ---",
        f"ON exit種別: settle-exit={overall['n_on_exit_settle']} "
        f"timeout-exit={overall['n_on_exit_timeout']} "
        f"(timeout率={overall['rate_on_exit_timeout']})",
        f"滞在時間 mean/median: ON={overall['on_duration_sec_mean']}/"
        f"{overall['on_duration_sec_median']}s  "
        f"OFF(細切れ合計)={overall['off_duration_sec_mean']}/"
        f"{overall['off_duration_sec_median']}s",
        f"OFF側 細切れ突入回数 mean/median: "
        f"{overall['off_n_subsegments_mean']}/{overall['off_n_subsegments_median']}",
        "--- 要件1: 却下→採用の改善 ---",
        f"採用セル総数: OFF={overall['off_accepted_cells_total']} "
        f"ON={overall['on_accepted_cells_total']}",
        f"却下セル総数: OFF={overall['off_rejected_cells_total']} "
        f"ON={overall['on_rejected_cells_total']} "
        f"(改善={overall['rejected_cells_improvement']})",
        "--- 要件2b: 浮き誤消去 (_apply_gravity_filter 起因) ---",
        f"おじゃま浮き消去総数: OFF={overall['off_gravity_ojama_erased_total']} "
        f"ON={overall['on_gravity_ojama_erased_total']} "
        f"(改善={overall['gravity_ojama_erased_improvement_total']}) "
        f"[全色込消去総数 OFF={overall['off_gravity_total_erased_total']} "
        f"ON={overall['on_gravity_total_erased_total']}]",
        "--- 要件3: TSUMO_FALL検出遅延 (回帰確認) ---",
        f"OJAMA_FALL退出→次TSUMO_FALL mean/median: "
        f"OFF={overall['tsumo_fall_delay_off_mean_sec']}/"
        f"{overall['tsumo_fall_delay_off_median_sec']}s "
        f"ON={overall['tsumo_fall_delay_on_mean_sec']}/"
        f"{overall['tsumo_fall_delay_on_median_sec']}s "
        f"(窓内未観測: OFF={overall['n_tsumo_fall_delay_off_unobserved']} "
        f"ON={overall['n_tsumo_fall_delay_on_unobserved']})",
        "--- 動画別 ---",
    ]
    for video, s in per_video.items():
        lines.append(
            f"  {video}: events={s['n_events_total']} footprint有={s['n_events_with_footprint_ojama']} "
            f"timeout率={s['rate_on_exit_timeout']} "
            f"却下改善={s['rejected_cells_improvement']} "
            f"浮き消去(OFF/ON)={s['off_gravity_ojama_erased']}/{s['on_gravity_ojama_erased']} "
            f"TSUMO遅延(OFF/ON)={s['tsumo_fall_delay_off_median_sec']}/"
            f"{s['tsumo_fall_delay_on_median_sec']}",
        )
    return "\n".join(lines)


# ============================
# viz: settle-exit / timeout-exit ヒストグラム
# ============================


def _write_exit_kind_histogram(events: list[_ABEvent], out_path: Path) -> None:
    """ON側 OJAMA_FALL 滞在時間の分布 + settle-exit/timeout-exit 内訳バーチャート。

    日本語ラベルは matplotlib フォント欠落 (CJK glyph missing) が既知のため ASCII のみ使用。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with_footprint = [e for e in events if e.footprint_ojama_cells > 0]
    if not with_footprint:
        return
    settle_durations = [e.on_duration_sec for e in with_footprint if e.on_exit_kind == "settle-exit"]
    timeout_durations = [e.on_duration_sec for e in with_footprint if e.on_exit_kind == "timeout-exit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(["settle-exit", "timeout-exit"], [len(settle_durations), len(timeout_durations)],
            color=["#2ca02c", "#d62728"])
    ax1.set_title("ON exit kind count")
    ax1.set_ylabel("n events")

    bins = np.linspace(0.0, max(OJAMA_FALL_MAX_SEC * 1.1, 0.1), 20)
    ax2.hist(settle_durations, bins=bins, alpha=0.7, label="settle-exit", color="#2ca02c")
    ax2.hist(timeout_durations, bins=bins, alpha=0.7, label="timeout-exit", color="#d62728")
    ax2.axvline(OJAMA_FALL_MAX_SEC, color="black", linestyle=":", label="OJAMA_FALL_MAX_SEC")
    ax2.set_xlabel("ON duration_sec")
    ax2.set_ylabel("n events")
    ax2.set_title("ON OJAMA_FALL duration distribution")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================
# viz: OFF確定盤面 vs ON確定盤面 vs 実ゲーム画面 montage
# ============================


def _roi_for_side(side: str) -> tuple[int, int]:
    return (P1_ROI_X, P1_ROI_Y) if side == "1P" else (P2_ROI_X, P2_ROI_Y)


def _label_panel(panel: np.ndarray, text: str) -> None:
    """ASCII 簡潔ラベル (matplotlib/cv2 とも CJK glyph 欠落のため日本語不使用)。"""
    cv2.putText(panel, text, (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)


def _write_ab_montage(
    video_stem: str, side: str, fps: float, t: float,
    board_off: "Board | None", board_on: "Board | None", out_path: Path,
) -> None:
    """OFF確定盤面 / ON確定盤面 / 実ゲーム画面 の 3 コマ横並び montage を書き出す。"""
    roi_x, roi_y = _roi_for_side(side)
    cap = cv2.VideoCapture(str(_video_path(video_stem)))
    fi = int(round(t * fps))
    frame = _seek_frame(cap, fi)
    cap.release()
    if frame is None:
        return
    raw_crop = frame[roi_y:roi_y + ROI_H, roi_x:roi_x + ROI_W]

    panel_off = raw_crop.copy()
    if board_off is not None:
        draw_cell_overlay(panel_off, board_off, 0, 0)
    _label_panel(panel_off, f"OFF(baseline) t={t:.2f}s")

    panel_on = raw_crop.copy()
    if board_on is not None:
        draw_cell_overlay(panel_on, board_on, 0, 0)
    _label_panel(panel_on, f"ON(B+a+b) t={t:.2f}s")

    panel_actual = raw_crop.copy()
    _label_panel(panel_actual, "ACTUAL SCREEN")

    h = raw_crop.shape[0]
    sep = np.full((h, 6, 3), (255, 255, 255), dtype=np.uint8)
    montage = np.hstack([panel_off, sep, panel_on, sep, panel_actual])
    cv2.imwrite(str(out_path), montage)


# ============================
# メイン処理: 1 動画分の OFF/ON 走査 + 側毎 ABEvent 構築
# ============================


def _process_video(
    video_stem: str, start_sec: float, end_sec: float, note: str,
) -> tuple[list[_ABEvent], dict]:
    _print_progress(f"[{video_stem}] 開始 window={start_sec:.1f}-{end_sec:.1f}s ({note})")

    t0 = time.time()
    off_1p, off_2p, fps, off_gravity = _collect_records_flagged(
        video_stem, start_sec, end_sec, OFF_FLAGS,
    )
    _print_progress(
        f"[{video_stem}] OFF pass 完了 ({len(off_1p)} frame, {time.time() - t0:.1f}s, "
        f"浮き消去={off_gravity.total_erased}(うちojama={off_gravity.ojama_erased}))",
    )
    t0 = time.time()
    on_1p, on_2p, _fps2, on_gravity = _collect_records_flagged(
        video_stem, start_sec, end_sec, ON_FLAGS,
    )
    _print_progress(
        f"[{video_stem}] ON pass 完了 ({len(on_1p)} frame, {time.time() - t0:.1f}s, "
        f"浮き消去={on_gravity.total_erased}(うちojama={on_gravity.ojama_erased}))",
    )

    video_events: list[_ABEvent] = []
    viz_sources: dict[str, list[_ABEvent]] = {}
    for side, records_off, records_on in (
        ("1P", off_1p, on_1p), ("2P", off_2p, on_2p),
    ):
        on_segments = _find_ojama_fall_segments(records_on)
        side_events: list[_ABEvent] = []
        for on_entry_idx, on_exit_idx in on_segments:
            ev = _build_ab_event(
                video_stem, side, records_off, records_on, on_entry_idx, on_exit_idx, fps,
            )
            if ev is None:
                continue
            video_events.append(ev)
            side_events.append(ev)
        viz_sources[side] = side_events

    _write_viz_for_video(video_stem, fps, viz_sources, {"1P": on_1p, "2P": on_2p}, {"1P": off_1p, "2P": off_2p})
    summary = _aggregate_ab_summary(video_events)
    summary["note"] = note
    summary["off_gravity_total_erased"] = off_gravity.total_erased
    summary["off_gravity_ojama_erased"] = off_gravity.ojama_erased
    summary["on_gravity_total_erased"] = on_gravity.total_erased
    summary["on_gravity_ojama_erased"] = on_gravity.ojama_erased
    summary["gravity_ojama_erased_improvement"] = (
        off_gravity.ojama_erased - on_gravity.ojama_erased
    )
    _print_progress(
        f"[{video_stem}] ABイベント={len(video_events)}件 "
        f"footprint有={summary['n_events_with_footprint_ojama']}件 完了",
    )
    return video_events, summary


def _write_viz_for_video(
    video_stem: str, fps: float,
    viz_sources: dict[str, list[_ABEvent]],
    records_on_by_side: dict[str, list],
    records_off_by_side: dict[str, list],
) -> None:
    """動画毎に footprint おじゃま最多の代表区間で A/B montage を出す。

    OFF/ON 側とも exit index は `_ABEvent` に保持済 (ON: on_exit_idx,
    OFF: off_exit_idx=時刻ベースマッチング済) をそのまま再利用する
    (2026-07-24 拡張、index 再計算による不一致を避けるため)。
    """
    for side, side_events in viz_sources.items():
        records_on = records_on_by_side[side]
        records_off = records_off_by_side[side]
        cands = [ev for ev in side_events if ev.footprint_ojama_cells > 0]
        cands.sort(key=lambda ev: ev.footprint_ojama_cells, reverse=True)
        for ev in cands[:MAX_VIZ_PER_VIDEO_AB]:
            label = f"{ev.on_entry_t:.2f}".replace(".", "_")
            board_on = records_on[ev.on_exit_idx].confirmed_board
            board_off = records_off[ev.off_exit_idx].confirmed_board
            _write_ab_montage(
                video_stem, side, fps, ev.on_exit_t, board_off, board_on,
                OUTPUT_DIR / f"viz_{video_stem}_{side}_t{label}_ab.png",
            )


# ============================
# エントリポイント
# ============================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="スモークモード: 短窓1件のみ処理し動作確認する (本走行禁止時に使用)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    smoke = args.smoke or os.environ.get("PUYO_OJAMA_AB_SMOKE") == "1"
    target_windows = SMOKE_TARGET_WINDOWS if smoke else FULL_TARGET_WINDOWS
    if smoke:
        _print_progress("[SMOKE MODE] 短窓1件のみ処理します (本走行ではありません)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events: list[_ABEvent] = []
    per_video_summary: dict[str, dict] = {}

    for video_stem, start_sec, end_sec, note in target_windows:
        video_events, summary = _process_video(video_stem, start_sec, end_sec, note)
        all_events.extend(video_events)
        per_video_summary[video_stem] = summary
        _write_ab_events_csv(video_events, OUTPUT_DIR / f"events_ab_{video_stem}.csv")

    overall = _aggregate_ab_summary(all_events)
    overall.update(_aggregate_gravity_overall(per_video_summary))
    _write_exit_kind_histogram(all_events, OUTPUT_DIR / "viz_settle_vs_timeout_hist.png")

    summary_all = {"overall": overall, "per_video": per_video_summary, "smoke_mode": smoke}
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary_all, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    text = _format_ab_summary_text(overall, per_video_summary)
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


if __name__ == "__main__":
    main()
