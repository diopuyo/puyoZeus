"""設置(実際に置いた瞬間) → confirmed_board反映 フレーム数精密計測 (read-only診断, 2026-07-25)。

## 背景・課題認識
既存 `_diag_c34_reflection_lag_2026-07-25.py` は起点を「TSUMO_FALL 状態を
抜けたフレーム」に取っていた。これはパイプライン state machine 自身の
遷移検出であり、それ自体が (grace period / landing vote 等で) 遅れる。
実測でも n_landed_cells==2 の「クリーン」着地が全イベントの 1〜2 割程度
しか取れておらず (例: video_c34 1P は 22 件中 2 件)、起点として不適切
だったことが分かっている。

## 本スクリプトの起点定義 (user 指定)
1. 生 CNN 観測 (state machine を経由しない `cnn_board`) で、着地候補 2 セルが
   空/UNKNOWN → 有効色 (1-5) に「初めて安定して埋まった」時刻を検出する
   (「安定」= 同色が RAW_STABLE_MIN_SEC 秒連続、fps換算)。
2. 1. の候補と NEXT 表示変化 (ツモ消費、`SideResult.next_pair` が変わる
   フレーム) が近傍 (± NEXT_CORROBORATION_WINDOW_SEC 秒) にあるかを突合し、
   「物理的に置かれたイベント」らしさを補強する (連鎖後の重力落下等
   ツモ設置以外の一斉充填ノイズを除外する目的。ただし NEXT が同一ペアを
   連続で示すケース (4色制約下では非無視できる頻度) では突合できないため、
   突合できなかったイベントも破棄せず別集計で報告する = fail-silent 回避)。
3. 終点は confirmed_board の該当 2 セルが「正しい色」(= 探索窓内の
   confirmed_board 多数決値) に一致した最初のフレーム。
4. 遅延を 2 区間に分解する:
   - 区間A (CNNが見えるまで): 設置 → 生CNNが正しい色で安定して見えるまで。
   - 区間B (見えてから確定まで): 生CNNが正しい色で見えてから confirmed_board
     に反映されるまで (state machine 側の遅延)。

## 受け入れ基準
設置 → 反映が 8 フレーム以内 (user指定の絶対基準、fps非依存の生フレーム数)。

## 対象
- video_c34 (30fps, game1 472-512s, 前後マージン込み 470-516s)
- video_30 (60fps, 対照, 225-315s)
- 設定: 現行HEAD + enable_landing_observed_color=True (レビュー候補構成と同一)。

## 制約
- read-only 診断。src/ は一切変更しない。
- 熱対策: cv2.setNumThreads(1)、並列しない (feedback_thermal_safety_mandatory)。
- --smoke で短窓の動作確認モードを用意する。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_placement_confirm_frames_2026-07-25.py
    PYTHONPATH=. ./venv/bin/python scripts/_diag_placement_confirm_frames_2026-07-25.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "placement_confirm_frames_2026-07-25"

# レビュー候補構成 (採否レビューに出す構成と同一設定)。
ENABLE_LANDING_OBSERVED_COLOR: bool = True

# 対象窓: user指定の正しい game1 境界 (472-512s) + 前後マージン。
VIDEO_C34: str = "c34"
C34_START_SEC: float = 470.0
C34_MAX_SEC: float = 46.0  # 470.0-516.0s (472-512s を完全に包含)

# 対照: video_30 (60fps)。
VIDEO_30: str = "30"
V30_START_SEC: float = 225.0
V30_MAX_SEC: float = 90.0  # 225.0-315.0s

# 「安定」判定: 同色が何秒連続すれば「安定して埋まった」とみなすか。
# 本番の収集・レンダが明示している stable_frame_count (2026-07-31)。
# collect_boards_lean.py:560 / collect_indicators_v2.py:728 /
# visualize_advantage_overlay.py:1043 / recognition_physics_review.py:213
# がいずれも 3 を渡している。RecognitionPipeline.load_default の既定は 6 で、
# 引数なしで呼ぶと**本番と違う設定を測ってしまう**。
# 盤面が**正当に確定できない**状態 (2026-07-31)。
# これらの最中に着地しても confirmed_board は更新できないので、
# その遅延は「直すべき遅延」ではなく物理的に正当な待ちである。
#   CHAIN         : 連鎖演出中 (消去→落下が進行中)
#   OJAMA_FALL    : おじゃま落下中
#   GRAVITY_SETTLE: 重力落ち着き待ち
#   MENU          : 試合外 (境界演出等)
#   EFFECT        : 全消し等の演出中
# これを混ぜて測っていたため、側の差が試合展開に振り回されていた
# (「2Pが遅い」は c34 1本の攻め/受けの偏りだった)。
BLOCKING_STATES: frozenset[str] = frozenset({
    "CHAIN", "OJAMA_FALL", "GRAVITY_SETTLE", "MENU", "EFFECT",
})

PRODUCTION_STABLE_FRAME_COUNT: int = 3

# fps換算で frame 数にする (30fps→3f, 60fps→6f、既存 stable_frame_count=6
# @60fps 相当の基準と整合)。
RAW_STABLE_MIN_SEC: float = 0.10

# 着地候補 2 セルの同時性許容 (どちらも「同一設置」由来とみなす時間差)。
PAIR_TOLERANCE_SEC: float = 0.08

# NEXT表示変化 (ツモ消費) との突合窓 (± 秒)。
NEXT_CORROBORATION_WINDOW_SEC: float = 2.0

# confirmed_board 側「真の色」判定 + 反映探索の窓 (秒)。
TRUTH_VOTE_WINDOW_SEC: float = 20.0

# 「真の色」判定の安定継続フレーム数 (confirmed_board が同色で連続する最小長)。
# 着地直後に現れる最初の安定値を「真の色」とする (窓内多数決だと、後続の
# 連鎖クリア→別ぷよ/おじゃまの再着地でセルが再利用された場合に多数決が
# 汚染され、見かけ上の巨大遅延を生む実測バグを踏んだため撤回、置換した)。
TRUTH_STABLE_MIN_FRAMES: int = 2

# 受け入れ基準 (user指定、fps非依存の生フレーム数)。
ACCEPTANCE_FRAMES: int = 8

# 計測分母修正 (2026-07-25): 走査窓終端からこのフレーム数以内で発生した
# イベントは「反映を確認しきれる前に窓が終わった」可能性がある
# (right-censored)。集計の分母/分子から除外し、n_right_censored として別掲する。
RIGHT_CENSOR_MARGIN_FRAMES: int = 30

# 「試合開始直後」判定: 走査窓開始からこの秒数未満の着地を near_match_start と
# 分類する (post-hoc実測で c34 のマッチ開始直後 (472-478s 付近) に区間B が
# 数秒〜13秒台まで膨らむ現象を確認したため、定常状態と切り分けて報告する)。
NEAR_MATCH_START_SEC: float = 15.0

# 着地セルとみなす色 (通常ツモは 1-5 の2色ペア)。
_VALID_LANDING_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})
_EMPTY_LIKE: frozenset[int] = frozenset({COLOR_EMPTY, COLOR_UNKNOWN})

SMOKE_MAX_SEC: float = 12.0


def _print_progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 (video, side, frame) 分の観測値。"""

    frame_idx: int
    t: float
    is_match_active: bool
    cnn_grid: np.ndarray
    confirmed_grid: "np.ndarray | None"
    next_pair: "tuple[int, int] | None"
    # 状態文脈 (2026-07-31 追加)。BoardState 名の文字列。
    # 「盤面が正当に確定できないフレーム」を層別するために使う。
    # 既定 "" は state を記録しない旧呼び出し用 (後方互換)。
    state: str = ""


@dataclass
class _PlacementEvent:
    """1 回の着地候補イベント (生CNN起点、NEXT突合込み)。"""

    video: str
    side: str
    cell_a: tuple[int, int]
    cell_b: tuple[int, int]
    frame_place: int
    t_place: float
    has_next_corroboration: bool
    frame_next_change: "int | None"
    raw_color_a: int
    raw_color_b: int
    truth_color_a: "int | None"
    truth_color_b: "int | None"
    color_match_at_place: "bool | None"
    frame_cnn_correct: "int | None"
    t_cnn_correct: "float | None"
    frame_reflect: "int | None"
    t_reflect: "float | None"
    reflected_within_window: bool
    delay_frames_total: "int | None"
    delay_frames_cnn_seg: "int | None"
    delay_frames_confirm_seg: "int | None"
    within_acceptance: "bool | None"
    near_match_start: bool
    # 状態文脈 (2026-07-31 追加)。イベント窓内に「盤面が正当に確定できない
    # 状態」が含まれていたか。含まれる遅延は物理的に正当な待ちなので、
    # 直すべき遅延と混ぜない。既定値付きなので末尾に置く (dataclass 制約)。
    window_had_blocking_state: bool = False
    window_blocking_states: str = ""


# ============================
# パス1: 走査
# ============================


def _video_path(video_stem: str) -> Path:
    return VIDEO_DIR / f"video_{video_stem}.mp4"


def _collect_records(
    video_stem: str, start_sec: float, max_sec: float,
    *,
    enable_landing_observed_color: bool = ENABLE_LANDING_OBSERVED_COLOR,
    enable_drift_resync_match_start_guard: bool = False,
    enable_drift_resync_hsv_gate: bool = False,
    enable_baseline_broken_reset: bool = True,
    enable_baseline_broken_grace: bool = False,
    enable_column_partial_support: bool = False,
    enable_placement_cnn_veto: bool = False,
    placement_cnn_veto_mode: str = "hold",
    enable_match_start_full_clear: bool = True,
    enable_recovery_counter_carryover: bool = False,
    enable_cnn_flicker_hsv_fallback: bool = False,
    enable_puyo_to_empty_hsv_guard: bool = False,
    stable_frame_count: int | None = None,
    recovery_min_frames: int | None = None,
    enable_side_sat_calibration: bool = False,
    pipeline_out: dict | None = None,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    """video を走査し、1P/2P それぞれの frame 記録を返す (現行既定構成)。

    enable_landing_observed_color: 着地色補正 (レビュー承認4修正の1つ)。既定
    ENABLE_LANDING_OBSERVED_COLOR (=True) = 従来通り (bit-identical)。
    False を渡すと旧構成 (この修正なし) を再現できる (2026-07-25 汎化監査用に追加)。
    enable_drift_resync_match_start_guard / enable_drift_resync_hsv_gate:
    2026-07-25 DriftDetector再同期ループ暴走ガード(commit c5bb50e)の
    効果測定用に追加。既定 False = 従来通り (bit-identical)。
    RecognitionPipeline.load_default へそのまま透過する。
    enable_baseline_broken_reset / enable_baseline_broken_grace: 2026-07-25
    baseline_broken 自己リセット制御フラグの A/B 計測用に追加。既定
    True/False = RecognitionPipeline 側既定と同一 (bit-identical)。
    RecognitionPipeline.load_default へそのまま透過する。
    enable_column_partial_support: 列ゲート緩和 (2026-07-25) の A/B 計測用に
    追加。既定 False = 従来通り (bit-identical)。
    RecognitionPipeline.load_default へそのまま透過する。
    enable_placement_cnn_veto / placement_cnn_veto_mode: 修正方針 甲
    (2026-07-25) P2 設置推論の防御的 CNN 照合の A/B 計測用に追加。既定
    False/"hold" = 従来通り (bit-identical)。
    RecognitionPipeline.load_default へそのまま透過する。
    enable_match_start_full_clear: 試合境界フルクリア (レビュー承認4修正の1つ)。
    既定 True = 従来通り (bit-identical、これまで本関数は明示せず
    RecognitionPipeline 側既定 True に委ねていたのと同じ値)。False を渡すと
    旧構成 (この修正なし) を再現できる (2026-07-25 汎化監査用に追加)。
    enable_recovery_counter_carryover: 復旧カウンタ carryover (#51,
    2026-07-26) の A/B 計測用に追加。既定 False = 従来通り (bit-identical)。
    enable_cnn_flicker_hsv_fallback: CNN 乱高下セル HSV フォールバック
    (#51 後半, 2026-07-26) の A/B 計測用に追加。既定 False = 従来通り
    (bit-identical)。
    pipeline_out: 抑制カウンタ (_drift_resync_*_suppressed_*) 観測用。
    既定 None = 従来通り (副作用なし)。dict を渡すと呼び出し後に
    pipeline_out["pipeline"] へ構築済み RecognitionPipeline を格納する。
    """
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    # 【2026-07-31】本番の収集・レンダは全て stable_frame_count=3 を明示しているのに
    # 本測定器は load_default の既定 6 を使っていた (= 本番と違う設定を測っていた)。
    # viz の7フラグ罠と同型の事故なので、未指定なら本番値に合わせた上で警告を出す。
    if stable_frame_count is None:
        stable_frame_count = PRODUCTION_STABLE_FRAME_COUNT
        print(
            f"[warn] --stable-frame-count 未指定のため本番値 "
            f"{PRODUCTION_STABLE_FRAME_COUNT} を使用 "
            f"(load_default の既定 6 ではない)"
        )
    pipe = RecognitionPipeline.load_default(
        enable_landing_observed_color=enable_landing_observed_color,
        enable_drift_resync_match_start_guard=enable_drift_resync_match_start_guard,
        enable_drift_resync_hsv_gate=enable_drift_resync_hsv_gate,
        enable_baseline_broken_reset=enable_baseline_broken_reset,
        enable_baseline_broken_grace=enable_baseline_broken_grace,
        enable_column_partial_support=enable_column_partial_support,
        enable_placement_cnn_veto=enable_placement_cnn_veto,
        placement_cnn_veto_mode=placement_cnn_veto_mode,
        enable_match_start_full_clear=enable_match_start_full_clear,
        enable_recovery_counter_carryover=enable_recovery_counter_carryover,
        enable_cnn_flicker_hsv_fallback=enable_cnn_flicker_hsv_fallback,
        enable_puyo_to_empty_hsv_guard=enable_puyo_to_empty_hsv_guard,
        **(
            {} if stable_frame_count is None
            else {"stable_frame_count": int(stable_frame_count)}
        ),
        enable_side_sat_calibration=enable_side_sat_calibration,
    )
    # 復旧ゲート閾値 (STABLE_RECOVERY_MIN_FRAMES) は load_default から届かないので
    # 構築後の state machine に直接差し込む (2026-07-31 掃引用、read-only 診断)。
    # 既定 None のときは一切触らない = 従来と完全に同じ挙動。
    if recovery_min_frames is not None:
        for _attr in ("_sm_1p", "_sm_2p"):
            _sm = getattr(pipe, _attr, None)
            if _sm is not None and hasattr(_sm, "_recovery_min_frames"):
                _sm._recovery_min_frames = max(1, int(recovery_min_frames))
    pipe.set_video_id(video_stem)
    if pipeline_out is not None:
        pipeline_out["pipeline"] = pipe

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
                frame_idx=fi, t=t, is_match_active=r.is_match_active,
                cnn_grid=side_res.cnn_board._grid.copy(),
                confirmed_grid=(
                    side_res.confirmed_board._grid.copy()
                    if side_res.confirmed_board is not None else None
                ),
                next_pair=side_res.next_pair,
                # 状態文脈 (2026-07-31)。BoardState.XXX の XXX だけを持つ。
                state=str(getattr(side_res, "state", "")).replace(
                    "BoardState.", "",
                ),
            ))
        fi += 1
        n_read += 1
        if n_read % 900 == 0:
            _print_progress(f"[{video_stem}] t={t:.1f}s まで処理済み ({n_read} frames)")
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# 計測: 生CNN 安定セルイベント検出
# ============================


def _compute_run_length(grids: np.ndarray) -> np.ndarray:
    """各セルにつき「直前フレームと同色が何フレーム連続しているか」(n,13,6)。"""
    n = grids.shape[0]
    run_len = np.ones(grids.shape, dtype=np.int32)
    for i in range(1, n):
        same = grids[i] == grids[i - 1]
        run_len[i] = np.where(same, run_len[i - 1] + 1, 1)
    return run_len


def _find_cell_settle_events(
    grids: np.ndarray, run_len: np.ndarray, k: int,
) -> list[dict]:
    """空/UNKNOWN → 有効色 に「初めて k フレーム連続で安定」したセル単位イベント。"""
    n = grids.shape[0]
    is_stable = run_len >= k
    events: list[dict] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            prev_stable = False
            for i in range(n):
                stable_now = bool(is_stable[i, r, c])
                if stable_now and not prev_stable:
                    color = int(grids[i, r, c])
                    start_idx = i - int(run_len[i, r, c]) + 1
                    before_idx = start_idx - 1
                    if (
                        color in _VALID_LANDING_COLORS and before_idx >= 0
                        and int(grids[before_idx, r, c]) in _EMPTY_LIKE
                    ):
                        events.append({
                            "r": r, "c": c, "start_idx": start_idx,
                            "confirm_idx": i, "color": color,
                        })
                prev_stable = stable_now
    return events


def _is_adjacent_tsumo_shape(a: dict, b: dict) -> bool:
    """2 セルがツモペアの形状 (縦 or 横 に隣接) として妥当か。"""
    if a["r"] == b["r"] and abs(a["c"] - b["c"]) == 1:
        return True
    if a["c"] == b["c"] and abs(a["r"] - b["r"]) == 1:
        return True
    return False


def _pair_landing_candidates(
    cell_events: list[dict], tolerance_frames: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """セル単位の着地イベントを 2 セル 1 組の着地候補にまとめる。

    Returns:
        (paired, unmatched, ambiguous)。ambiguous = 許容誤差内に候補が
        2 つ以上見つかり一意にペア化できないもの (連鎖後の重力落下等、
        ツモ設置以外の一斉充填を除外する目的で除外する)。
    """
    events_sorted = sorted(cell_events, key=lambda e: e["start_idx"])
    used = [False] * len(events_sorted)
    ambiguous_idx: set[int] = set()
    paired: list[dict] = []
    for i, ev in enumerate(events_sorted):
        if used[i] or i in ambiguous_idx:
            continue
        candidates = [
            j for j, other in enumerate(events_sorted)
            if j != i and not used[j] and j not in ambiguous_idx
            and abs(other["start_idx"] - ev["start_idx"]) <= tolerance_frames
            and _is_adjacent_tsumo_shape(ev, other)
        ]
        if not candidates:
            continue
        if len(candidates) > 1:
            ambiguous_idx.add(i)
            ambiguous_idx.update(candidates)
            continue
        j = candidates[0]
        used[i] = True
        used[j] = True
        b = events_sorted[j]
        place_idx = min(ev["start_idx"], b["start_idx"])
        paired.append({"cell_a": ev, "cell_b": b, "place_idx": place_idx})
    unmatched = [
        ev for i, ev in enumerate(events_sorted)
        if not used[i] and i not in ambiguous_idx
    ]
    ambiguous = [events_sorted[i] for i in sorted(ambiguous_idx)]
    return paired, unmatched, ambiguous


# ============================
# 計測: NEXT表示変化 (ツモ消費) 突合
# ============================


def _detect_next_change_indices(records: list[_FrameRec]) -> list[int]:
    """next_pair が前フレームと異なる (= ツモ消費) フレームの index 一覧。"""
    out: list[int] = []
    for i in range(1, len(records)):
        prev_np, cur_np = records[i - 1].next_pair, records[i].next_pair
        if prev_np is not None and cur_np is not None and prev_np != cur_np:
            out.append(i)
    return out


def _nearest_next_change(
    place_idx: int, next_change_indices: list[int], window_frames: int,
) -> tuple[bool, "int | None"]:
    """place_idx 近傍 (± window_frames) に NEXT 変化イベントがあるか。"""
    best: "int | None" = None
    best_dist: "int | None" = None
    for idx in next_change_indices:
        dist = abs(idx - place_idx)
        if dist <= window_frames and (best_dist is None or dist < best_dist):
            best, best_dist = idx, dist
    return (best is not None), best


# ============================
# 計測: confirmed_board 真の色 + 反映フレーム + CNN到達フレーム
# ============================


def _first_stable_confirmed_value(
    records: list[_FrameRec], place_idx: int, cell: tuple[int, int],
    window_end_idx: int, min_hold_frames: int,
) -> tuple["int | None", "int | None"]:
    """place_idx 以降で confirmed_board が最初に安定した非空値とその開始 index。

    窓内多数決 (旧実装) だと、連鎖クリア後に同一セルへ別ぷよ/おじゃまが
    再着地するケースで多数決が汚染され、見かけ上の巨大遅延を生む
    (実測で確認済、raw_color と truth_color が一致しない巨大遅延イベントの
    ほぼ全数がこれに該当)。着地直後に最初に安定して現れた値を「真の色」と
    する方が「その着地の反映」を表すため妥当。
    """
    # 追記 (post-hoc実測で確認): truth が COLOR_OJAMA(9) に解決されるケースが
    # video_30 で多数見つかった。ツモペアは常に有効色 (1-5) のため、真の色が
    # OJAMA になるのは「このセルが後の連鎖クリア後に別のおじゃまへ再利用された」
    # 別事象を誤って捕捉している証拠であり、「このツモの反映」ではない。
    # よって有効ツモ色 (1-5) 以外は「真の色」として採用しない (= 未反映扱い)。
    run_val: "int | None" = None
    run_start = place_idx
    for i in range(place_idx, window_end_idx):
        cg = records[i].confirmed_grid
        v = int(cg[cell]) if cg is not None else COLOR_EMPTY
        if v == COLOR_EMPTY or v != run_val:
            run_val = v
            run_start = i
        if (
            run_val in _VALID_LANDING_COLORS
            and (i - run_start + 1) >= min_hold_frames
        ):
            return run_val, run_start
    return None, None


def _resolve_truth_and_reflect(
    records: list[_FrameRec], place_idx: int, cell_a: tuple[int, int],
    cell_b: tuple[int, int], window_end_idx: int,
) -> tuple["int | None", "int | None", "int | None"]:
    """confirmed_board が着地直後に最初に安定した色を「真の色」とし、反映フレームを探す。"""
    truth_a, _ = _first_stable_confirmed_value(
        records, place_idx, cell_a, window_end_idx, TRUTH_STABLE_MIN_FRAMES,
    )
    truth_b, _ = _first_stable_confirmed_value(
        records, place_idx, cell_b, window_end_idx, TRUTH_STABLE_MIN_FRAMES,
    )
    if truth_a is None or truth_b is None:
        return truth_a, truth_b, None
    for i in range(place_idx, window_end_idx):
        cg = records[i].confirmed_grid
        if cg is None:
            continue
        if int(cg[cell_a]) == truth_a and int(cg[cell_b]) == truth_b:
            return truth_a, truth_b, i
    return truth_a, truth_b, None


def _find_cnn_correct_idx(
    records: list[_FrameRec], run_len: np.ndarray, place_idx: int, end_idx: int,
    cell_a: tuple[int, int], cell_b: tuple[int, int],
    truth_a: int, truth_b: int, k: int,
) -> "int | None":
    """place_idx 以降で、生CNNが両セルとも真の色で k フレーム安定した最初の index。"""
    for i in range(place_idx, end_idx):
        grid = records[i].cnn_grid
        if (
            int(grid[cell_a]) == truth_a and int(grid[cell_b]) == truth_b
            and int(run_len[i][cell_a]) >= k and int(run_len[i][cell_b]) >= k
        ):
            return i
    return None


# ============================
# 1 side 分のイベント構築
# ============================


def _build_placement_events(
    records: list[_FrameRec], video: str, side: str, fps: float,
    window_start_sec: float,
) -> tuple[list[_PlacementEvent], dict]:
    """1 (video, side) 分の着地イベント一覧 + 診断メタ情報 (unmatched/ambiguous 件数)。"""
    grids = np.stack([r.cnn_grid for r in records])
    k = max(2, round(fps * RAW_STABLE_MIN_SEC))
    tol_frames = max(1, round(fps * PAIR_TOLERANCE_SEC))
    next_window_frames = max(1, round(fps * NEXT_CORROBORATION_WINDOW_SEC))
    truth_window_frames = max(1, round(fps * TRUTH_VOTE_WINDOW_SEC))

    run_len = _compute_run_length(grids)
    cell_events = _find_cell_settle_events(grids, run_len, k)
    paired, unmatched, ambiguous = _pair_landing_candidates(cell_events, tol_frames)
    next_change_indices = _detect_next_change_indices(records)

    n = len(records)
    events: list[_PlacementEvent] = []
    for p in paired:
        place_idx = p["place_idx"]
        if not records[place_idx].is_match_active:
            continue
        cell_a = (p["cell_a"]["r"], p["cell_a"]["c"])
        cell_b = (p["cell_b"]["r"], p["cell_b"]["c"])
        has_corr, next_idx = _nearest_next_change(
            place_idx, next_change_indices, next_window_frames,
        )
        window_end = min(place_idx + truth_window_frames, n)
        truth_a, truth_b, reflect_idx = _resolve_truth_and_reflect(
            records, place_idx, cell_a, cell_b, window_end,
        )
        cnn_correct_idx = None
        if truth_a is not None and truth_b is not None:
            correct_end = reflect_idx if reflect_idx is not None else window_end
            cnn_correct_idx = _find_cnn_correct_idx(
                records, run_len, place_idx, correct_end, cell_a, cell_b,
                truth_a, truth_b, k,
            )
        near_start = (records[place_idx].t - window_start_sec) < NEAR_MATCH_START_SEC
        events.append(_build_one_event(
            video, side, records, cell_a, cell_b, place_idx, has_corr, next_idx,
            p["cell_a"]["color"], p["cell_b"]["color"], truth_a, truth_b,
            cnn_correct_idx, reflect_idx, near_start,
        ))
    meta = {
        "n_paired": len(paired), "n_unmatched_single_cell": len(unmatched),
        "n_ambiguous_multi_cell": len(ambiguous), "k_stable_frames": k,
        "pair_tolerance_frames": tol_frames,
    }
    return events, meta


def _build_one_event(
    video: str, side: str, records: list[_FrameRec],
    cell_a: tuple[int, int], cell_b: tuple[int, int], place_idx: int,
    has_corr: bool, next_idx: "int | None", raw_a: int, raw_b: int,
    truth_a: "int | None", truth_b: "int | None",
    cnn_correct_idx: "int | None", reflect_idx: "int | None",
    near_match_start: bool,
) -> _PlacementEvent:
    """1 件分の _PlacementEvent を組み立てる (delay 計算込み)。"""
    # 状態文脈の判定 (2026-07-31)。着地から反映 (未反映なら窓端) までの間に
    # 「盤面が正当に確定できない状態」が含まれていたかを記録する。
    # これを混ぜて測っていたため、側の差が試合展開に振り回されていた。
    _end_idx = reflect_idx if reflect_idx is not None else len(records) - 1
    _seen: list[str] = []
    for _r in records[place_idx:max(place_idx + 1, _end_idx + 1)]:
        st = _r.state
        if st in BLOCKING_STATES and st not in _seen:
            _seen.append(st)
    color_match = (
        (raw_a == truth_a and raw_b == truth_b)
        if truth_a is not None and truth_b is not None else None
    )
    delay_total = (
        records[reflect_idx].frame_idx - records[place_idx].frame_idx
        if reflect_idx is not None else None
    )
    delay_cnn_seg = (
        records[cnn_correct_idx].frame_idx - records[place_idx].frame_idx
        if cnn_correct_idx is not None else None
    )
    delay_confirm_seg = (
        records[reflect_idx].frame_idx - records[cnn_correct_idx].frame_idx
        if (reflect_idx is not None and cnn_correct_idx is not None) else None
    )
    return _PlacementEvent(
        video=video, side=side, cell_a=cell_a, cell_b=cell_b,
        frame_place=records[place_idx].frame_idx, t_place=records[place_idx].t,
        has_next_corroboration=has_corr,
        frame_next_change=(records[next_idx].frame_idx if next_idx is not None else None),
        raw_color_a=raw_a, raw_color_b=raw_b,
        truth_color_a=truth_a, truth_color_b=truth_b,
        color_match_at_place=color_match,
        frame_cnn_correct=(
            records[cnn_correct_idx].frame_idx if cnn_correct_idx is not None else None
        ),
        t_cnn_correct=(records[cnn_correct_idx].t if cnn_correct_idx is not None else None),
        frame_reflect=(records[reflect_idx].frame_idx if reflect_idx is not None else None),
        t_reflect=(records[reflect_idx].t if reflect_idx is not None else None),
        reflected_within_window=(reflect_idx is not None),
        delay_frames_total=delay_total,
        delay_frames_cnn_seg=delay_cnn_seg,
        delay_frames_confirm_seg=delay_confirm_seg,
        within_acceptance=(
            (delay_total <= ACCEPTANCE_FRAMES) if delay_total is not None else None
        ),
        near_match_start=near_match_start,
        window_had_blocking_state=bool(_seen),
        window_blocking_states=",".join(_seen),
    )


# ============================
# 集計
# ============================


def _percentile_or_none(values: list[float], q: float) -> "float | None":
    return float(np.percentile(values, q)) if values else None


def _delay_stats(
    events: list[_PlacementEvent], corroborated_only: bool,
    steady_state_only: bool = False,
    window_end_frame_idx: "int | None" = None,
) -> dict:
    """反映遅延分布統計 (8フレーム達成率込み)。

    steady_state_only=True で試合開始直後 (NEAR_MATCH_START_SEC 未満) を除外し、
    定常状態のみの分布にする (試合開始直後は別要因の遅延が支配的なため分離)。

    計測分母修正 (2026-07-25):
    - effective_pct_within_acceptance_8f: 未反映 (never_reflected) を「失敗」と
      数えた分母 (= pool 全体) に対する 8f 達成率。従来の
      pct_within_acceptance_8f (分母=反映確認できたイベントのみ) は未反映を
      暗黙に除外しており、未反映が多いほど見かけの達成率が過大評価される
      問題があったため、こちらを主指標として追加する (旧指標は互換維持で残す)。
    - window_end_frame_idx を渡すと、走査窓終端から RIGHT_CENSOR_MARGIN_FRAMES
      以内で発生したイベントを right-censored とみなし、分母/分子の両方から
      除外して n_right_censored に計上する (None なら従来通りスキップ、
      backwards compat)。
    """
    pool = [e for e in events if (not corroborated_only) or e.has_next_corroboration]
    if steady_state_only:
        pool = [e for e in pool if not e.near_match_start]

    n_right_censored = 0
    if window_end_frame_idx is not None:
        censored_ids = {
            id(e) for e in pool
            if (window_end_frame_idx - e.frame_place) <= RIGHT_CENSOR_MARGIN_FRAMES
        }
        n_right_censored = len(censored_ids)
        pool = [e for e in pool if id(e) not in censored_ids]

    resolved = [e for e in pool if e.delay_frames_total is not None]
    delays = [float(e.delay_frames_total) for e in resolved]
    cnn_segs = [float(e.delay_frames_cnn_seg) for e in resolved if e.delay_frames_cnn_seg is not None]
    confirm_segs = [
        float(e.delay_frames_confirm_seg) for e in resolved
        if e.delay_frames_confirm_seg is not None
    ]
    within_8 = [e for e in resolved if e.within_acceptance]
    # 【本来測るべき数字】状態文脈で層別する (2026-07-31)。
    # 連鎖中・おじゃま落下中・メニュー中の遅延は物理的に正当な待ちであり、
    # 「直すべき遅延」ではない。混ぜて測っていたため側の差が試合展開に
    # 振り回されていた (「2Pが遅い」は c34 1本の攻め/受けの偏りだった)。
    clean = [e for e in resolved if not e.window_had_blocking_state]
    blocked = [e for e in resolved if e.window_had_blocking_state]
    clean_delays = [
        e.delay_frames_total for e in clean if e.delay_frames_total is not None
    ]
    clean_within_8 = [e for e in clean if e.within_acceptance]
    # どの状態が遅延を作っていたかの内訳 (対処の優先順位づけに使う)
    blocking_counter: dict[str, int] = {}
    for e in blocked:
        for st in e.window_blocking_states.split(","):
            if st:
                blocking_counter[st] = blocking_counter.get(st, 0) + 1
    return {
        # --- クリーン層 (直すべき遅延) ---
        "n_clean": len(clean),
        "clean_delay_median": (
            float(np.median(clean_delays)) if clean_delays else None
        ),
        "clean_delay_p90": (
            float(np.percentile(clean_delays, 90)) if clean_delays else None
        ),
        "clean_delay_max": (max(clean_delays) if clean_delays else None),
        "clean_pct_within_acceptance_8f": (
            100.0 * len(clean_within_8) / len(clean) if clean else None
        ),
        # --- 阻害状態を含む層 (物理的に正当な待ち) ---
        "n_blocked": len(blocked),
        "blocking_state_counts": blocking_counter,
        "n_events_total": len(pool),
        "n_right_censored": n_right_censored,
        "n_resolved_within_window": len(resolved),
        "n_never_reflected": len(pool) - len(resolved),
        "n_within_acceptance_8f": len(within_8),
        "pct_within_acceptance_8f": (
            100.0 * len(within_8) / len(resolved) if resolved else None
        ),
        "effective_pct_within_acceptance_8f": (
            100.0 * len(within_8) / len(pool) if pool else None
        ),
        "delay_frames_median": (float(np.median(delays)) if delays else None),
        "delay_frames_mean": (float(np.mean(delays)) if delays else None),
        "delay_frames_p90": _percentile_or_none(delays, 90),
        "delay_frames_max": (float(np.max(delays)) if delays else None),
        "seg_cnn_visible_median": (float(np.median(cnn_segs)) if cnn_segs else None),
        "seg_confirm_median": (float(np.median(confirm_segs)) if confirm_segs else None),
        "n_color_mismatch_at_place": sum(
            1 for e in pool if e.color_match_at_place is False
        ),
    }


# ============================
# 出力
# ============================


def _write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


_CSV_COLS: list[str] = [
    "video", "side", "cell_a_r", "cell_a_c", "cell_b_r", "cell_b_c",
    "frame_place", "t_place", "has_next_corroboration", "frame_next_change",
    "raw_color_a", "raw_color_b", "truth_color_a", "truth_color_b",
    "color_match_at_place", "frame_cnn_correct", "t_cnn_correct",
    "frame_reflect", "t_reflect", "reflected_within_window",
    "delay_frames_total", "delay_frames_cnn_seg", "delay_frames_confirm_seg",
    "within_acceptance_8f", "near_match_start",
]


def _event_to_row(e: _PlacementEvent) -> list[object]:
    return [
        e.video, e.side, e.cell_a[0], e.cell_a[1], e.cell_b[0], e.cell_b[1],
        e.frame_place, f"{e.t_place:.3f}", e.has_next_corroboration, e.frame_next_change,
        e.raw_color_a, e.raw_color_b, e.truth_color_a, e.truth_color_b,
        e.color_match_at_place, e.frame_cnn_correct, e.t_cnn_correct,
        e.frame_reflect, e.t_reflect, e.reflected_within_window,
        e.delay_frames_total, e.delay_frames_cnn_seg, e.delay_frames_confirm_seg,
        e.within_acceptance, e.near_match_start,
    ]


def _write_events_csv(events: list[_PlacementEvent], path: Path) -> None:
    lines = [",".join(_CSV_COLS)]
    for e in events:
        lines.append(",".join(str(v) for v in _event_to_row(e)))
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_stats_line(label: str, s: dict) -> str:
    return (
        f"  {label}: n={s['n_events_total']} "
        f"right_censored={s.get('n_right_censored', 0)} "
        f"未反映={s['n_never_reflected']} "
        f"delay_frames(中央値/平均/p90/最大)="
        f"{s['delay_frames_median']}/{s['delay_frames_mean']}/"
        f"{s['delay_frames_p90']}/{s['delay_frames_max']} "
        f"8f以内達成率(反映済のみ)={s['pct_within_acceptance_8f']} "
        f"8f以内達成率(実効=未反映込み)={s.get('effective_pct_within_acceptance_8f')} "
        f"区間A(CNN到達)中央値={s['seg_cnn_visible_median']} "
        f"区間B(確定側)中央値={s['seg_confirm_median']} "
        f"初期色不一致={s['n_color_mismatch_at_place']}"
    )


def _format_clean_line(label: str, s: dict) -> str:
    """【本来測るべき数字】状態文脈でクリーンなイベントのみの集計 (2026-07-31)。

    連鎖中・おじゃま落下中・メニュー中を含むイベントは物理的に正当な待ちなので
    除外する。混ぜて測ると側の差が試合展開に振り回される。
    """
    counts = s.get("blocking_state_counts") or {}
    top = ", ".join(
        f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    return (
        f"  {label} [クリーン層]: n={s.get('n_clean')} "
        f"delay(中央/p90/最大)="
        f"{s.get('clean_delay_median')}/{s.get('clean_delay_p90')}/"
        f"{s.get('clean_delay_max')} "
        f"8f以内達成率={s.get('clean_pct_within_acceptance_8f')} "
        f"| 阻害状態を含む={s.get('n_blocked')} ({top or 'なし'})"
    )


# ============================
# 1 動画分の実行
# ============================


def _process_one(
    video: str, start_sec: float, max_sec: float,
    *,
    enable_landing_observed_color: bool = ENABLE_LANDING_OBSERVED_COLOR,
    enable_drift_resync_match_start_guard: bool = False,
    enable_drift_resync_hsv_gate: bool = False,
    enable_baseline_broken_reset: bool = True,
    enable_baseline_broken_grace: bool = False,
    enable_column_partial_support: bool = False,
    enable_placement_cnn_veto: bool = False,
    placement_cnn_veto_mode: str = "hold",
    enable_match_start_full_clear: bool = True,
    enable_recovery_counter_carryover: bool = False,
    enable_cnn_flicker_hsv_fallback: bool = False,
    enable_puyo_to_empty_hsv_guard: bool = False,
    stable_frame_count: int | None = None,
    recovery_min_frames: int | None = None,
    enable_side_sat_calibration: bool = False,
) -> dict:
    """1 動画分の走査 + イベント構築 + 集計。

    enable_landing_observed_color: 着地色補正 (既定 ENABLE_LANDING_OBSERVED_COLOR
    =True = 従来通り、bit-identical)。2026-07-25 汎化監査用に追加。
    enable_drift_resync_match_start_guard / enable_drift_resync_hsv_gate:
    2026-07-25 DriftDetector再同期ループ暴走ガード効果測定用 (既定 False)。
    enable_baseline_broken_reset / enable_baseline_broken_grace: 2026-07-25
    baseline_broken 自己リセット制御フラグの A/B 計測用 (既定 True/False =
    RecognitionPipeline 側既定と同一)。
    enable_column_partial_support: 列ゲート緩和 (2026-07-25) の A/B 計測用
    (既定 False)。
    enable_placement_cnn_veto / placement_cnn_veto_mode: 修正方針 甲
    (2026-07-25) P2 設置推論の防御的 CNN 照合の A/B 計測用 (既定
    False/"hold")。
    enable_match_start_full_clear: 試合境界フルクリア (既定 True = 従来通り、
    bit-identical)。2026-07-25 汎化監査用に追加。
    enable_recovery_counter_carryover / enable_cnn_flicker_hsv_fallback:
    #51 (2026-07-26) の A/B 計測用 (既定 False)。
    """
    t0 = time.time()
    _print_progress(f"[{video}] 走査開始 start={start_sec:.1f}s dur={max_sec:.1f}s")
    pipeline_out: dict = {}
    recs_1p, recs_2p, fps = _collect_records(
        video, start_sec, max_sec,
        enable_landing_observed_color=enable_landing_observed_color,
        enable_drift_resync_match_start_guard=enable_drift_resync_match_start_guard,
        enable_drift_resync_hsv_gate=enable_drift_resync_hsv_gate,
        enable_baseline_broken_reset=enable_baseline_broken_reset,
        enable_baseline_broken_grace=enable_baseline_broken_grace,
        enable_column_partial_support=enable_column_partial_support,
        enable_placement_cnn_veto=enable_placement_cnn_veto,
        placement_cnn_veto_mode=placement_cnn_veto_mode,
        enable_match_start_full_clear=enable_match_start_full_clear,
        enable_recovery_counter_carryover=enable_recovery_counter_carryover,
        enable_cnn_flicker_hsv_fallback=enable_cnn_flicker_hsv_fallback,
        enable_puyo_to_empty_hsv_guard=enable_puyo_to_empty_hsv_guard,
        stable_frame_count=stable_frame_count,
        recovery_min_frames=recovery_min_frames,
        enable_side_sat_calibration=enable_side_sat_calibration,
        pipeline_out=pipeline_out,
    )
    _print_progress(f"[{video}] 走査完了 ({time.time() - t0:.1f}s) fps={fps:.2f}")
    pipeline_obj = pipeline_out.get("pipeline")
    drift_suppressed = {
        "start_guard_suppressed_1p": getattr(
            pipeline_obj, "_drift_resync_start_guard_suppressed_1p", 0,
        ),
        "start_guard_suppressed_2p": getattr(
            pipeline_obj, "_drift_resync_start_guard_suppressed_2p", 0,
        ),
        "hsv_gate_suppressed_1p": getattr(
            pipeline_obj, "_drift_resync_hsv_gate_suppressed_1p", 0,
        ),
        "hsv_gate_suppressed_2p": getattr(
            pipeline_obj, "_drift_resync_hsv_gate_suppressed_2p", 0,
        ),
    } if pipeline_obj is not None else {}
    # baseline_broken 自己リセット 発火回数 + grace 抑制回数 (2026-07-25 A/B 計測用)。
    baseline_broken_stats = {
        "reset_count_1p": getattr(pipeline_obj, "_baseline_broken_reset_count_1p", 0),
        "reset_count_2p": getattr(pipeline_obj, "_baseline_broken_reset_count_2p", 0),
        "grace_suppressed_1p": getattr(
            pipeline_obj, "_baseline_broken_grace_suppressed_1p", 0,
        ),
        "grace_suppressed_2p": getattr(
            pipeline_obj, "_baseline_broken_grace_suppressed_2p", 0,
        ),
    } if pipeline_obj is not None else {}
    if baseline_broken_stats:
        _print_progress(f"[{video}] baseline_broken計測: {baseline_broken_stats}")
    if drift_suppressed:
        _print_progress(f"[{video}] drift_resync抑制カウンタ: {drift_suppressed}")
    # 修正方針 甲 (2026-07-25): P2 設置推論 CNN veto 保留セル数計測。
    placement_cnn_veto_stats = {
        "held_count_1p": getattr(
            pipeline_obj, "_placement_cnn_veto_held_count_1p", 0,
        ),
        "held_count_2p": getattr(
            pipeline_obj, "_placement_cnn_veto_held_count_2p", 0,
        ),
    } if pipeline_obj is not None else {}
    if placement_cnn_veto_stats:
        _print_progress(
            f"[{video}] placement_cnn_veto保留セル数: {placement_cnn_veto_stats}",
        )

    events_1p, meta_1p = _build_placement_events(recs_1p, video, "1P", fps, start_sec)
    events_2p, meta_2p = _build_placement_events(recs_2p, video, "2P", fps, start_sec)
    # 計測分母修正 (2026-07-25): 走査窓終端 frame_idx を right-censoring 判定に渡す。
    win_end_1p = recs_1p[-1].frame_idx if recs_1p else None
    win_end_2p = recs_2p[-1].frame_idx if recs_2p else None
    stats_1p_corr = _delay_stats(
        events_1p, corroborated_only=True, window_end_frame_idx=win_end_1p,
    )
    stats_2p_corr = _delay_stats(
        events_2p, corroborated_only=True, window_end_frame_idx=win_end_2p,
    )
    stats_1p_all = _delay_stats(
        events_1p, corroborated_only=False, window_end_frame_idx=win_end_1p,
    )
    stats_2p_all = _delay_stats(
        events_2p, corroborated_only=False, window_end_frame_idx=win_end_2p,
    )
    stats_1p_steady = _delay_stats(
        events_1p, corroborated_only=True, steady_state_only=True,
        window_end_frame_idx=win_end_1p,
    )
    stats_2p_steady = _delay_stats(
        events_2p, corroborated_only=True, steady_state_only=True,
        window_end_frame_idx=win_end_2p,
    )
    return {
        "video": video, "fps": fps, "start_sec": start_sec, "max_sec": max_sec,
        "events_1p": events_1p, "events_2p": events_2p,
        "meta_1p": meta_1p, "meta_2p": meta_2p,
        "stats_1p_corroborated": stats_1p_corr, "stats_2p_corroborated": stats_2p_corr,
        "stats_1p_all": stats_1p_all, "stats_2p_all": stats_2p_all,
        "stats_1p_steady_state": stats_1p_steady, "stats_2p_steady_state": stats_2p_steady,
        "drift_resync_suppressed": drift_suppressed,
        "baseline_broken_stats": baseline_broken_stats,
        "placement_cnn_veto_stats": placement_cnn_veto_stats,
    }


def _write_result_outputs(
    result: dict, output_suffix: str = "", output_dir: Path = OUTPUT_DIR,
) -> None:
    """1 動画分の CSV 出力。

    output_suffix: 2026-07-25 ガードON計測用に追加。既定 "" = 従来通りの
    ファイル名 (bit-identical)。非空を渡すと従来出力 (events_c34.csv 等) を
    上書きせず区別できる (例: "_guardon" → events_c34_guardon.csv)。
    output_dir: 既定 OUTPUT_DIR = 従来通り (bit-identical)。2026-07-25
    汎化監査用に追加、任意の出力先ディレクトリへ切り替え可能にする。
    """
    video = result["video"]
    _write_events_csv(
        result["events_1p"] + result["events_2p"],
        output_dir / f"events_{video}{output_suffix}.csv",
    )


def _build_summary(results: list[dict]) -> dict:
    """全動画分の summary.json 用 dict。"""
    summary: dict = {}
    for result in results:
        video = result["video"]
        summary[video] = {
            "fps": result["fps"],
            "meta_1p": result["meta_1p"], "meta_2p": result["meta_2p"],
            "stats_1p_corroborated": result["stats_1p_corroborated"],
            "stats_2p_corroborated": result["stats_2p_corroborated"],
            "stats_1p_all_candidates": result["stats_1p_all"],
            "stats_2p_all_candidates": result["stats_2p_all"],
            "stats_1p_steady_state": result["stats_1p_steady_state"],
            "stats_2p_steady_state": result["stats_2p_steady_state"],
            "drift_resync_suppressed": result.get("drift_resync_suppressed", {}),
            "baseline_broken_stats": result.get("baseline_broken_stats", {}),
            "placement_cnn_veto_stats": result.get("placement_cnn_veto_stats", {}),
        }
    return summary


def _format_summary_text(results: list[dict]) -> str:
    """summary.txt 用テキスト整形。"""
    lines = [
        "==== 設置(実際に置いた瞬間)→confirmed_board反映 精密計測 (2026-07-25) ====",
        f"受け入れ基準: {ACCEPTANCE_FRAMES} フレーム以内 (fps非依存の生フレーム数)",
    ]
    for result in results:
        lines.append(
            f"--- video_{result['video']} (fps={result['fps']:.1f}) "
            f"[NEXT突合済イベントのみ] ---",
        )
        lines.append(_format_stats_line("1P", result["stats_1p_corroborated"]))
        lines.append(_format_stats_line("2P", result["stats_2p_corroborated"]))
        # 【本来測るべき数字】状態文脈でクリーンな層のみ (2026-07-31)
        lines.append(_format_clean_line("1P", result["stats_1p_corroborated"]))
        lines.append(_format_clean_line("2P", result["stats_2p_corroborated"]))
        lines.append(
            f"    meta 1P: {result['meta_1p']} / meta 2P: {result['meta_2p']}",
        )
        lines.append(
            f"    baseline_broken計測: {result.get('baseline_broken_stats', {})}",
        )
        lines.append(
            f"    placement_cnn_veto保留セル数: "
            f"{result.get('placement_cnn_veto_stats', {})}",
        )
        lines.append(f"--- video_{result['video']} [全候補イベント参考] ---")
        lines.append(_format_stats_line("1P", result["stats_1p_all"]))
        lines.append(_format_stats_line("2P", result["stats_2p_all"]))
        lines.append(
            f"--- video_{result['video']} "
            f"[定常状態のみ (試合開始直後{NEAR_MATCH_START_SEC:.0f}s除外)] ---",
        )
        lines.append(_format_stats_line("1P", result["stats_1p_steady_state"]))
        lines.append(_format_stats_line("2P", result["stats_2p_steady_state"]))
    return "\n".join(lines)


# ============================
# CLI / main
# ============================


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="短窓のみ処理する動作確認モード")
    ap.add_argument(
        "--only-c34", dest="only_c34", action="store_true", default=False,
        help="既定False(c34+video_30両方処理)。指定時は c34 窓のみ処理する。",
    )
    # DriftDetector再同期ループ暴走ガード (commit c5bb50e) 効果測定用
    # (2026-07-25)。既定 False = 従来通り (bit-identical)。
    ap.add_argument(
        "--enable-drift-match-start-guard", dest="enable_drift_resync_match_start_guard",
        action="store_true", default=False,
        help="既定False。RecognitionPipeline の "
             "enable_drift_resync_match_start_guard を有効化する。",
    )
    ap.add_argument(
        "--enable-drift-hsv-gate", dest="enable_drift_resync_hsv_gate",
        action="store_true", default=False,
        help="既定False。RecognitionPipeline の "
             "enable_drift_resync_hsv_gate を有効化する。",
    )
    # baseline_broken 自己リセット 制御フラグ (2026-07-25, A/B 計測用)。
    # 既定 True/False = RecognitionPipeline 側既定と同一 (bit-identical)。
    ap.add_argument(
        "--no-baseline-broken-reset", dest="enable_baseline_broken_reset",
        action="store_false", default=True,
        help="既定True(従来通り)。指定時は RecognitionPipeline の "
             "enable_baseline_broken_reset を False にし、baseline_broken "
             "自己リセット機能全体を無効化する。",
    )
    ap.add_argument(
        "--enable-baseline-broken-grace", dest="enable_baseline_broken_grace",
        action="store_true", default=False,
        help="既定False。指定時は RecognitionPipeline の "
             "enable_baseline_broken_grace を有効化する "
             "(STABLE突入からBASELINE_BROKEN_STABLE_GRACE_SEC秒はカウンタ加算を抑制)。",
    )
    ap.add_argument(
        "--stable-frame-count", dest="stable_frame_count", type=int, default=None,
        help="着地後に STABLE 宣言するまでの連続フレーム数 (本番既定 6)。"
             "省略時はライブラリ既定 = 従来と完全に同じ挙動。"
             "小さくすると反映は速くなるが、着地しきる前の遷移フレームを "
             "STABLE 誤認する (浮きぷよ) リスクが上がる。",
    )
    ap.add_argument(
        "--side-sat-calibration", dest="enable_side_sat_calibration",
        action="store_true", default=False,
        help="側別 彩度適応較正を有効化する (2026-07-31)。",
    )
    ap.add_argument(
        "--recovery-min-frames", dest="recovery_min_frames", type=int, default=None,
        help="確定済みセルを上書きするまでの証拠フレーム数 "
             "(STABLE_RECOVERY_MIN_FRAMES、本番既定 8)。省略時は変更しない。"
             "小さくすると「誤認が治るまで」の右tailが縮むが、"
             "一時的な誤読で確定盤面が書き換わりやすくなる。",
    )
    ap.add_argument(
        "--output-suffix", dest="output_suffix", type=str, default="",
        help="既定\"\"(従来通り)。非空を渡すと出力ファイル名に付与し、"
             "従来出力 (events_c34.csv/summary.json 等) を上書きしない。",
    )
    # 列ゲート緩和 (2026-07-25, A/B 計測用)。既定 False = 従来通り (bit-identical)。
    ap.add_argument(
        "--enable-column-partial-support", dest="enable_column_partial_support",
        action="store_true", default=False,
        help="既定False。指定時は RecognitionPipeline の "
             "enable_column_partial_support を有効化する "
             "(設計C 事後復旧ゲートの安全弁C浮き判定を列カウンタ進行中セルで緩和)。",
    )
    # 修正方針 甲: P2 設置推論の防御的 CNN 照合 (2026-07-25, A/B 計測用)。
    # 既定 False = 従来通り (bit-identical)。
    ap.add_argument(
        "--enable-placement-cnn-veto", dest="enable_placement_cnn_veto",
        action="store_true", default=False,
        help="既定False。指定時は RecognitionPipeline の "
             "enable_placement_cnn_veto を有効化する "
             "(P2 infer_placement の着地セル書き込み前に現フレーム CNN 観測と "
             "照合し、不一致なら保留する)。",
    )
    ap.add_argument(
        "--placement-cnn-veto-mode", dest="placement_cnn_veto_mode",
        type=str, default="hold",
        choices=["hold", "cnn_color", "empty_hold_cnn_color"],
        help="既定'hold'。'cnn_color' で不一致セルに CNN 観測色 (有効 puyo 色の "
             "場合のみ) を採用する代替挙動を試す。",
    )
    # 着地色補正 / 試合境界フルクリア (レビュー承認4修正のうち残り2つ) の
    # OFF切替 (2026-07-25 汎化監査用に追加)。既定 True = 従来通り (bit-identical)。
    ap.add_argument(
        "--no-landing-observed-color", dest="enable_landing_observed_color",
        action="store_false", default=True,
        help="既定True(従来通り)。指定時は着地色補正 "
             "(enable_landing_observed_color) を無効化する (旧構成比較用)。",
    )
    ap.add_argument(
        "--no-match-start-full-clear", dest="enable_match_start_full_clear",
        action="store_false", default=True,
        help="既定True(従来通り)。指定時は試合境界フルクリア "
             "(enable_match_start_full_clear) を無効化する (旧構成比較用)。",
    )
    # 復旧カウンタ carryover (#51, 2026-07-26, A/B 計測用)。既定 False = 従来通り
    # (bit-identical)。
    ap.add_argument(
        "--enable-recovery-counter-carryover",
        dest="enable_recovery_counter_carryover",
        action="store_true", default=False,
        help="既定False。指定時は RecognitionPipeline の "
             "enable_recovery_counter_carryover を有効化する "
             "(STABLE→NON-STABLE 遷移時の復旧カウンタを短時間の非STABLE滞在なら"
             "引き継ぐ)。",
    )
    # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26, A/B 計測用)。
    # 既定 False = 従来通り (bit-identical)。
    ap.add_argument(
        "--enable-cnn-flicker-hsv-fallback",
        dest="enable_cnn_flicker_hsv_fallback",
        action="store_true", default=False,
        help="既定False。指定時は RecognitionPipeline の "
             "enable_cnn_flicker_hsv_fallback を有効化する "
             "(CNN 出力が直近フレームで乱高下しているセルは HSV を合意値とみなす)。",
    )
    # 色→空 HSV 照合ガード (案A, 2026-07-30, A/B 計測用)。既定 False =
    # 従来通り (bit-identical)。反映遅延受け入れ (8フレーム基準) の A/B 用。
    ap.add_argument(
        "--enable-puyo-to-empty-hsv-guard",
        dest="enable_puyo_to_empty_hsv_guard",
        action="store_true", default=False,
        help="既定False。指定時は RecognitionPipeline の "
             "enable_puyo_to_empty_hsv_guard を有効化する "
             "(NON-STABLE→STABLE 復帰 merge の色→空 遷移を HSV が色を保持する "
             "cell について退ける、案A)。",
    )
    # 任意動画・任意窓での実行 (2026-07-25 汎化監査用に追加)。既定 None =
    # 従来通り (c34+video_30 固定窓、bit-identical)。3引数はセットで指定する。
    ap.add_argument(
        "--video", dest="video", type=str, default=None,
        help="既定None(従来通りc34+video_30)。指定時はこの video stem "
             "(例: 29 → video_29.mp4) のみ単一窓で処理する。"
             "--start-sec / --max-sec と併せて指定する。",
    )
    ap.add_argument(
        "--start-sec", dest="start_sec", type=float, default=None,
        help="--video 指定時の走査開始秒。",
    )
    ap.add_argument(
        "--max-sec", dest="max_sec", type=float, default=None,
        help="--video 指定時の走査秒数。",
    )
    ap.add_argument(
        "--output-dir", dest="output_dir", type=str, default=None,
        help="既定None(data/verify/placement_confirm_frames_2026-07-25、"
             "従来通り)。指定時は出力先ディレクトリを切り替える。",
    )
    return ap.parse_args()


def main() -> None:
    cv2.setNumThreads(1)
    args = _parse_args()
    max_sec_c34 = SMOKE_MAX_SEC if args.smoke else C34_MAX_SEC
    max_sec_v30 = SMOKE_MAX_SEC if args.smoke else V30_MAX_SEC
    if args.smoke:
        _print_progress("[SMOKE MODE] 短窓のみ処理します (本走行ではありません)")

    # 出力先切替 (2026-07-25 汎化監査用に追加)。既定 None = 従来通り OUTPUT_DIR。
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    guard_kwargs = {
        "enable_landing_observed_color": args.enable_landing_observed_color,
        "enable_drift_resync_match_start_guard": args.enable_drift_resync_match_start_guard,
        "enable_drift_resync_hsv_gate": args.enable_drift_resync_hsv_gate,
        "enable_baseline_broken_reset": args.enable_baseline_broken_reset,
        "enable_baseline_broken_grace": args.enable_baseline_broken_grace,
        "enable_column_partial_support": args.enable_column_partial_support,
        "enable_placement_cnn_veto": args.enable_placement_cnn_veto,
        "placement_cnn_veto_mode": args.placement_cnn_veto_mode,
        "enable_match_start_full_clear": args.enable_match_start_full_clear,
        "enable_recovery_counter_carryover": args.enable_recovery_counter_carryover,
        "enable_cnn_flicker_hsv_fallback": args.enable_cnn_flicker_hsv_fallback,
        "enable_puyo_to_empty_hsv_guard": args.enable_puyo_to_empty_hsv_guard,
        # 2026-07-31 掃引用。None なら従来と完全に同じ挙動。
        "stable_frame_count": args.stable_frame_count,
        "recovery_min_frames": args.recovery_min_frames,
        "enable_side_sat_calibration": args.enable_side_sat_calibration,
    }

    # 任意動画・任意窓モード (2026-07-25 汎化監査用に追加)。既定 (--video 未指定)
    # では従来通り c34+video_30 固定窓を処理する (bit-identical)。
    if args.video is not None:
        if args.start_sec is None or args.max_sec is None:
            raise SystemExit(
                "--video 指定時は --start-sec と --max-sec も必須です",
            )
        results = [_process_one(args.video, args.start_sec, args.max_sec, **guard_kwargs)]
    else:
        result_c34 = _process_one(VIDEO_C34, C34_START_SEC, max_sec_c34, **guard_kwargs)
        results = [result_c34]
        if not args.only_c34:
            result_v30 = _process_one(VIDEO_30, V30_START_SEC, max_sec_v30, **guard_kwargs)
            results.append(result_v30)

    for result in results:
        _write_result_outputs(result, output_suffix=args.output_suffix, output_dir=output_dir)

    summary = _build_summary(results)
    _write_json(summary, output_dir / f"summary{args.output_suffix}.json")

    text = _format_summary_text(results)
    (output_dir / f"summary{args.output_suffix}.txt").write_text(text, encoding="utf-8")
    _print_progress(f"[DONE] 出力先: {output_dir}")
    print(text)


if __name__ == "__main__":
    main()
