"""盤面物理矛盾 全類型監査器 (#48・user要望の恒久エンジン)。

個別バグ追跡ではなく「物理的に不可能な盤面状態/遷移」を全類型スキャンし、
頻度順に潰すための常設監視の土台。read-only (src 非改変)。

confirmed_board (STABLE 専用) と estimated_board (連鎖中物理推定) を分離し、
過渡状態 (CHAIN/GRAVITY_SETTLE 中の物理推定盤面) を誤検出しない
(= board_provenance=="observed" かつ state==STABLE のフレームのみ判定対象)。

capture層 (フレーム取得・chain trigger 検出) は重複実装せず
scripts.recognition_physics_review から import して流用する。

2026-07-24 実用化修正: 既定4動画 (c62,30,35,38) の処理窓は
scripts/_diag_ojama_fall_exit_timing_2026-07-24.py の TARGET_WINDOWS
(A/B診断と同一の90秒級短窓) を動的importで再利用する
(DIAG_WINDOW_BY_STEM 参照)。c62 は matches.tsv が無く、従来は
「動画全体を1区間」に fallback して数十分を舐め長時間化していたため、
既定4動画ではこの fallback を通らないようにした。

類型一覧 (VIOLATION_TYPE_ID 参照):
    1  おじゃま会計乖離           — 骨組みのみ (未実装、TODO参照)
    2  保存則違反-消失            — 新規実装
    3  保存則違反-出現            — 新規実装
    4  色フリッカ                 — 新規実装
    5  重力違反 (浮きぷよ)        — src/self_supervised/physical_consistency 流用
    6  4連結消去漏れ              — 同上
    7  色種数超過                 — 同上
    9  窒息整合違反               — 新規実装
    16 おじゃまドロップ (#45)     — 登録のみ・担当別・本スクリプトでは検出しない
    17 設置ドロップ (#47)        — 登録のみ・担当別・本スクリプトでは検出しない

Usage:
    PYTHONPATH=. python -m scripts.physics_violation_audit --smoke
    PYTHONPATH=. python -m scripts.physics_violation_audit --videos c62,30,35,38
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, DEATH_COL, DEATH_ROW,
    HIDDEN_ROWS, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.match_end_detector import MatchEndDetector  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.score_zero import ScoreZeroDetector  # noqa: E402
from src.self_supervised.physical_consistency import (  # noqa: E402
    check_color_count, check_gravity_rule, check_no_pre_chain_4_plus_connection,
)
from scripts.recognition_physics_review import (  # noqa: E402
    _FrameRecord, _capture_frames, _new_chain_triggers,
)
# 2026-07-24 3コマmontage拡張: confirmed_board のセル色オーバーレイ描画は
# 既存 visualize_recognition.draw_cell_overlay をそのまま流用する (描画ロジック
# の重複実装を避ける、scripts/_diag_ojama_fall_board_settle_ab_2026-07-24.py の
# 3コマmontage手法と同じ再利用パターン)。
from scripts.visualize_recognition import draw_cell_overlay  # noqa: E402

# ============================
# 窓定義の再利用 (A/B診断 _diag_ojama_fall_exit_timing_2026-07-24.py と同一の
# 4動画短窓。ファイル名にハイフンを含み通常の import 文が使えないため
# importlib で動的 import する。窓定義の重複実装を避ける目的
# (scripts/_diag_ojama_fall_board_settle_ab_2026-07-24.py と同じ再利用手法)。
# ============================
_DIAG_WINDOWS_PATH = PROJ_ROOT / "scripts" / "_diag_ojama_fall_exit_timing_2026-07-24.py"
_diag_windows_spec = importlib.util.spec_from_file_location(
    "_diag_ojama_fall_exit_timing_reuse_for_audit", _DIAG_WINDOWS_PATH,
)
assert _diag_windows_spec is not None and _diag_windows_spec.loader is not None
_diag_windows_module = importlib.util.module_from_spec(_diag_windows_spec)
# dataclass の型解決が module 登録前提のため、exec_module 前に登録しておく
# (旧診断側スクリプトの同種コメントと同じ理由、AttributeError 回避)。
sys.modules[_diag_windows_spec.name] = _diag_windows_module
_diag_windows_spec.loader.exec_module(_diag_windows_module)  # 定義のみ実行 (main() は __name__ ガード済)

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
MATCH_BOUNDARIES_DIR: Path = PROJ_ROOT / "data" / "verify" / "match_boundaries_v5"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "physics_violation_audit"
# 固定名サブディレクトリ (timestamp は不要、毎回上書きでよい運用)。
RUN_DIR: Path = OUTPUT_DIR / "latest"

# フル走行対象動画 (アーキ指定): c62(マスター大連鎖多い) + 30/35/38(match_boundaries_v5確認済)。
DEFAULT_VIDEO_STEMS: tuple[str, ...] = ("c62", "30", "35", "38")

# 2026-07-24 実用化修正: DEFAULT_VIDEO_STEMS の実処理窓は A/B診断
# (_diag_ojama_fall_exit_timing_2026-07-24.TARGET_WINDOWS) と完全一致させる。
# c62 は matches.tsv 不在で従来「動画全体を1区間」に fallback し
# 数十分を舐めて長時間化した実績があるため、既定4動画は必ずこの短窓を
# 優先採用し全動画1区間フォールバックを踏ませない (_resolve_windows 参照)。
DIAG_WINDOW_BY_STEM: dict[str, tuple[float, float]] = {
    stem: (start, end - start)
    for stem, start, end, _note in _diag_windows_module.TARGET_WINDOWS
}

# 進捗ログの間隔 (game-time 秒)。既存 _diag_*.py の frame数間隔ログと違い、
# 本監査器は認識負荷が重く fps が動画により変動するため秒基準に統一する。
PROGRESS_LOG_INTERVAL_SEC: float = 30.0

# スモーク専用窓: video_30 の実測済み安全区間 (225.0-246.0s、
# scripts/_diag_ojama_fall_board_settle_ab_2026-07-24.py で baseline_missing=False
# 実測確認済みの開始点を流用し、OJAMA_FALL/CHAIN 遷移を含む短窓で動作確認する)。
SMOKE_VIDEO_STEM: str = "30"
SMOKE_START_SEC: float = 225.0
SMOKE_MAX_SEC: float = 21.0

# STABLE確定ラグ警告閾値 (秒)。GHOST_FIRST_STABLE_LAG_WARN_SEC (既存
# recognition_physics_review.py) と同じ根拠・同じ値を採用 (単純な1連鎖なら
# 数秒で STABLE 復帰するはず、それ以上は比較対象が不確かとして信頼度を下げる)。
CONFIRM_LAG_WARN_SEC: float = 5.0

# 2026-07-24 FP修正 (試合終了演出テロップ誤検出対策):
# RecognitionPipeline 内部の is_match_active は chain_in_progress 抑制
# (src/recognition_pipeline.py 付近、連鎖終了直後の誤 hard_match_off 抑制)
# と絡み合うため、演出開始タイミングが連鎖終了と重なると is_match_active が
# 正しく False にならず confirmed_board が STABLE のまま演出テロップの
# 色を盤面色として読み続ける (color_flicker FP の主因、実フレーム目視確認済)。
# 監査器側で state machine と完全独立に同一フレームへ再適用し、除外区間を得る。
TELOP_MASK_STATE_NAME: str = BoardState.MENU.name

# 類型ごとに frame crop を出力する上位件数 (montage対象外の単パネル類型に適用、
# 全動画合算の上位。既存挙動を維持、後方互換のため据え置き)。
MAX_FRAMES_PER_TYPE: int = 3

# frame crop の余白 (px)。
FRAME_CROP_MARGIN_PX: int = 30

# 2026-07-24 3コマmontage拡張 (user要望):
# 変化前STABLE/変化後STABLE/実画面 の3コマ横並びで出力する対象類型。
# 色フリッカ (色A→B, 変化前後が見えないと判定不能) が最優先、保存則違反2種・
# 4連結消去漏れも同様に前後比較が有効なため対象に含める。gravity_violation/
# death_row_inconsistency は単snapshot判定 (前後比較の意味が薄い) のため
# 単パネルのまま維持する (user明示指定)。
MONTAGE_TYPES: frozenset[str] = frozenset({
    "color_flicker", "connection_uncleared", "conservation_loss", "conservation_gain",
})

# montage対象類型は動画別に上位何件出すか (user指定「各類型 発生件数上位3〜5件/動画」
# の中央値を採用)。単パネル類型 (MAX_FRAMES_PER_TYPE) とは集計軸が異なる
# (こちらは (type, video_stem) 単位、単パネルは type 単位で全動画合算)。
MAX_MONTAGE_FRAMES_PER_TYPE_PER_VIDEO: int = 5

# montage 内の区切り線幅 (px) / 上部バナー高さ (px)。
MONTAGE_SEPARATOR_PX: int = 4
MONTAGE_BANNER_HEIGHT_PX: int = 26

# severity ラベル
SEVERITY_HIGH: str = "high"
SEVERITY_MEDIUM: str = "medium"

# 類型文字列 -> 類型番号 (報告用)
VIOLATION_TYPE_ID: dict[str, int] = {
    "ojama_accounting_rate_floor": 1,
    "conservation_loss": 2,
    "conservation_gain": 3,
    "color_flicker": 4,
    "gravity_violation": 5,
    "connection_uncleared": 6,
    "color_count_excess": 7,
    "death_row_inconsistency": 9,
    "ojama_drop_desync": 16,
    "placement_drop_desync": 17,
}

# 類型文字列 -> severity (物理的にありえない=high、認識揺れの疑い=medium)
VIOLATION_TYPE_SEVERITY: dict[str, str] = {
    "ojama_accounting_rate_floor": SEVERITY_HIGH,
    "conservation_loss": SEVERITY_HIGH,
    "conservation_gain": SEVERITY_HIGH,
    "color_flicker": SEVERITY_MEDIUM,
    "gravity_violation": SEVERITY_HIGH,
    "connection_uncleared": SEVERITY_HIGH,
    "color_count_excess": SEVERITY_MEDIUM,
    "death_row_inconsistency": SEVERITY_HIGH,
    "ojama_drop_desync": SEVERITY_MEDIUM,
    "placement_drop_desync": SEVERITY_MEDIUM,
}


@dataclass
class Violation:
    """1 件の物理矛盾違反。

    type/severity/t_sec/side/cells/detail はアーキ指定の必須フィールド。
    video_stem/frame_idx/lag_flag は動画別集計・frame crop 生成・信頼度表示に
    必要な実装上の追加フィールド (背後互換の懸念がない新規スクリプトのため追加)。

    prev_frame_idx/prev_t_sec/prev_grid/curr_grid は 2026-07-24 3コマmontage拡張で
    追加 (既定値で後方互換維持、MONTAGE_TYPES 対象の類型のみ実値を設定する)。
    prev_frame_idx=-1 は「直前のSTABLE snapshotが存在しない (最初のsnapshot等)」
    を表し、montage生成側で N/A コマにフォールバックする。
    """
    type: str
    severity: str
    t_sec: float
    side: str
    cells: list[tuple[int, int]]
    detail: str
    video_stem: str = ""
    frame_idx: int = -1
    lag_flag: bool = False
    prev_frame_idx: int = -1
    prev_t_sec: float = -1.0
    prev_grid: np.ndarray | None = None
    curr_grid: np.ndarray | None = None


# ============================
# 共通ヘルパー (STABLE observed 限定・誤検出ガード)
# ============================


def _is_stable_observed(rec: _FrameRecord) -> bool:
    """confirmed_board が STABLE かつ observed 由来 (estimated_board 除外) か。"""
    return (
        rec.state == BoardState.STABLE.name
        and rec.board_provenance == "observed"
        and rec.grid is not None
    )


def _distinct_stable_snapshots(
    records: list[_FrameRecord],
) -> list[tuple[int, _FrameRecord]]:
    """STABLE observed フレームのうち直前と異なる盤面のみ (records index, rec) で返す。"""
    out: list[tuple[int, _FrameRecord]] = []
    last_bytes: bytes | None = None
    for i, rec in enumerate(records):
        if not _is_stable_observed(rec):
            continue
        gb = rec.grid.tobytes()
        if gb == last_bytes:
            continue
        last_bytes = gb
        out.append((i, rec))
    return out


def _confirm_lag_sec(records: list[_FrameRecord], idx: int) -> float:
    """idx の STABLE 確定に至るまでの遅延秒 (直前の非STABLEフレームからの経過)。"""
    for j in range(idx - 1, -1, -1):
        if records[j].state != BoardState.STABLE.name:
            return records[idx].t_sec - records[j].t_sec
    return 0.0


def _has_chain_trigger_in_range(
    records: list[_FrameRecord], chain_idxs: list[int], t_lo: float, t_hi: float,
) -> bool:
    """(t_lo, t_hi] 区間に新規 chain trigger があるか。"""
    return any(t_lo < records[i].t_sec <= t_hi for i in chain_idxs)


def _has_placement_event_in_range(records: list[_FrameRecord], i_lo: int, i_hi: int) -> bool:
    """(i_lo, i_hi] 区間に TSUMO_FALL/OJAMA_FALL state のフレームがあるか (設置/着弾の代理検出)。"""
    target = (BoardState.TSUMO_FALL.name, BoardState.OJAMA_FALL.name)
    return any(records[k].state in target for k in range(i_lo + 1, i_hi + 1))


# 2026-07-25 Step0 (書き込み元トレース診断 kickoff): 既存 _has_placement_event_in_range
# は変更せず (後方互換維持)、CHAIN/GRAVITY_SETTLE も正当な色変化イベントとして含めた
# 拡張版を新規追加する。連鎖確定 (P3/P4 経路) による事後補正が color_flicker /
# conservation_gain として誤検出されているのではという仮説を、既存挙動を壊さずに
# 検証するため (--strict-exclusion フラグでのみ有効化、既定 False = 従来通り)。
STRICT_PLACEMENT_EVENT_STATES: tuple[str, ...] = (
    BoardState.TSUMO_FALL.name, BoardState.OJAMA_FALL.name,
    BoardState.CHAIN.name, BoardState.GRAVITY_SETTLE.name,
)


def _has_placement_event_in_range_strict(
    records: list[_FrameRecord], i_lo: int, i_hi: int,
) -> bool:
    """(i_lo, i_hi] 区間に設置/着弾/連鎖/重力settle stateのフレームがあるか (拡張版)。

    既存 _has_placement_event_in_range の CHAIN/GRAVITY_SETTLE 版。
    連鎖 (消去→重力再充填で同セルに別色着地) を挟む色変化・個数増加は
    正当なイベントとみなし除外対象に含める。
    """
    return any(
        records[k].state in STRICT_PLACEMENT_EVENT_STATES
        for k in range(i_lo + 1, i_hi + 1)
    )


def _menu_occurred_between(records: list[_FrameRecord], i_lo: int, i_hi: int) -> bool:
    """[i_lo, i_hi) 区間に MENU state のフレームがあるか (試合終了の正常検知)。"""
    return any(records[k].state == BoardState.MENU.name for k in range(i_lo, i_hi))


# ============================
# 試合外/テロップ フレーム独立検出・除外 (2026-07-24 FP修正)
# ============================
#
# RecognitionPipeline 内部の is_match_active (force_in_match=True 環境下でも
# match_end_locked/score_zero_both 自体は無効化されないが、chain_in_progress
# 抑制ロジックと絡み合い連鎖終了直後の演出開始を取りこぼす懸念がある) に
# 依存せず、監査器側で MatchEndDetector (やった!/ばたんきゅー テンプレ) と
# ScoreZeroDetector (両者スコア00000000) を同一フレームへ独立適用し、
# 試合外・演出中と判定できる区間を確実に除外する。read-only (src 非改変、
# 既存 src クラスをそのまま呼ぶのみ)。


def _scan_telop_exclusion_intervals(
    video_stem: str, start_sec: float, max_sec: float,
) -> list[tuple[float, float]]:
    """1 動画・1 窓分を独立スキャンし、試合外/演出テロップ除外区間を返す。

    MatchEndDetector.update() 内部ロックダウン (既定 5 秒) と ScoreZeroDetector
    の両者ゼロ判定を毎フレーム適用し、いずれか true の連続区間をマージして返す。
    テンプレート不在等で検出器が使えない場合は空リスト (除外なし、従来動作) を返す。
    """
    video_path = VIDEO_DIR / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)
    match_end_det = MatchEndDetector.load_default()
    try:
        score_zero_det: ScoreZeroDetector | None = ScoreZeroDetector.load_default()
    except FileNotFoundError:
        score_zero_det = None  # テンプレ不在なら score_zero 判定のみ無効化 (縮退動作)

    intervals: list[tuple[float, float]] = []
    cur_start: float | None = None
    last_t: float = start_sec
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = (start_frame + local_i) / fps
        last_t = t_sec
        excluded = bool(match_end_det.update(frame, t_sec))
        if not excluded and score_zero_det is not None:
            try:
                excluded = bool(score_zero_det.detect(frame).both_zero)
            except Exception:
                excluded = False
        if excluded and cur_start is None:
            cur_start = t_sec
        elif not excluded and cur_start is not None:
            intervals.append((cur_start, t_sec))
            cur_start = None
    if cur_start is not None:
        intervals.append((cur_start, last_t))
    cap.release()
    return intervals


def _in_any_interval(t_sec: float, intervals: list[tuple[float, float]]) -> bool:
    """t_sec がいずれかの除外区間 [t_lo, t_hi] に含まれるか。"""
    return any(t_lo <= t_sec <= t_hi for t_lo, t_hi in intervals)


def _mask_records_by_excluded_intervals(
    records: list[_FrameRecord], intervals: list[tuple[float, float]],
) -> list[_FrameRecord]:
    """除外区間内フレームの state を MENU に上書きした新リストを返す (非破壊)。

    _is_stable_observed は state==STABLE 限定のため、この上書きだけで
    全類型 (色フリッカ/保存則/重力等) の STABLE snapshot 判定から
    自動的に除外される。元の records は変更しない (dataclasses.replace)。
    """
    if not intervals:
        return records
    return [
        replace(rec, state=TELOP_MASK_STATE_NAME) if _in_any_interval(rec.t_sec, intervals)
        else rec
        for rec in records
    ]


# ============================
# 類型5/6/7: physical_consistency 流用ラッパー (STABLE observed 限定)
# ============================


def _check_gravity(records: list[_FrameRecord], video_stem: str, side: str) -> list[Violation]:
    """類型5: 重力違反 (浮きぷよ)。"""
    violations: list[Violation] = []
    for idx, rec in _distinct_stable_snapshots(records):
        board = Board.from_list(rec.grid.tolist())
        is_valid, cells = check_gravity_rule(board)
        if not is_valid:
            violations.append(Violation(
                type="gravity_violation", severity=VIOLATION_TYPE_SEVERITY["gravity_violation"],
                t_sec=rec.t_sec, side=side, cells=cells, video_stem=video_stem,
                frame_idx=rec.frame_idx,
                lag_flag=_confirm_lag_sec(records, idx) > CONFIRM_LAG_WARN_SEC,
                detail=f"浮きぷよ {len(cells)} セル (STABLE確定盤面, observed限定)",
            ))
    return violations


def _check_connection_uncleared(
    records: list[_FrameRecord], video_stem: str, side: str,
) -> list[Violation]:
    """類型6: 4連結以上消去漏れ。

    2026-07-24 3コマmontage拡張: 前後比較目視のため直前のSTABLE snapshot
    (prev_rec、検出ロジック自体は単snapshotのまま変更しない) を併記する。
    """
    violations: list[Violation] = []
    snaps = _distinct_stable_snapshots(records)
    for i, (idx, rec) in enumerate(snaps):
        board = Board.from_list(rec.grid.tolist())
        is_valid, clusters = check_no_pre_chain_4_plus_connection(board)
        if is_valid:
            continue
        lag_flag = _confirm_lag_sec(records, idx) > CONFIRM_LAG_WARN_SEC
        prev_rec = snaps[i - 1][1] if i > 0 else None
        for cluster in clusters:
            violations.append(Violation(
                type="connection_uncleared",
                severity=VIOLATION_TYPE_SEVERITY["connection_uncleared"],
                t_sec=rec.t_sec, side=side, cells=cluster["cells"], video_stem=video_stem,
                frame_idx=rec.frame_idx, lag_flag=lag_flag,
                detail=f"色{cluster['color']}の{len(cluster['cells'])}連結がSTABLEで未消去",
                prev_frame_idx=prev_rec.frame_idx if prev_rec is not None else -1,
                prev_t_sec=prev_rec.t_sec if prev_rec is not None else -1.0,
                prev_grid=prev_rec.grid if prev_rec is not None else None,
                curr_grid=rec.grid,
            ))
    return violations


def _check_color_count(records: list[_FrameRecord], video_stem: str, side: str) -> list[Violation]:
    """類型7: 色種数超過 (現状 check_color_count の5色上限のみ判定)。

    TODO: reference_four_colors_per_match (試合内実出現色集合、通常4色) で
    厳格化する余地あり。5色目が試合内で一度も出現していないのに現れた場合を
    検出できればより厳しい判定になるが、本監査器では既存 check_color_count
    (MAX_COLORS_IN_GAME=5) をそのまま用いる (誤った厳格化を入れない)。
    """
    violations: list[Violation] = []
    for idx, rec in _distinct_stable_snapshots(records):
        board = Board.from_list(rec.grid.tolist())
        is_valid, colors = check_color_count(board)
        if not is_valid:
            violations.append(Violation(
                type="color_count_excess", severity=VIOLATION_TYPE_SEVERITY["color_count_excess"],
                t_sec=rec.t_sec, side=side, cells=[], video_stem=video_stem,
                frame_idx=rec.frame_idx,
                lag_flag=_confirm_lag_sec(records, idx) > CONFIRM_LAG_WARN_SEC,
                detail=f"色種数{len(colors)} (色集合={sorted(colors)})",
            ))
    return violations


# ============================
# 類型4: 色フリッカ (新規)
# ============================


def _find_flicker_cells(
    prev_grid: np.ndarray, curr_grid: np.ndarray,
) -> list[tuple[int, int]]:
    """空を経由せず色A(非空)→色B(非空, 別色)に変化したセル一覧を返す。"""
    cells: list[tuple[int, int]] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            pv, cv = int(prev_grid[row, col]), int(curr_grid[row, col])
            if pv in (COLOR_EMPTY, COLOR_UNKNOWN) or cv in (COLOR_EMPTY, COLOR_UNKNOWN):
                continue
            if pv != cv:
                cells.append((row, col))
    return cells


def _check_color_flicker(
    records: list[_FrameRecord], video_stem: str, side: str,
    *, strict_exclusion: bool = False,
) -> list[Violation]:
    """類型4: 色フリッカ (STABLE observed snapshot間の直接色変化)。

    2026-07-24 FP修正: _check_conservation と同様に、連鎖 (消去→重力再充填で
    同セルに別色着地) または設置イベントを挟む snapshot 対は正当な色変化と
    みなし除外する (_has_chain_trigger_in_range / _has_placement_event_in_range)。

    strict_exclusion: 2026-07-25 Step0 追加 (既定 False = 従来通り後方互換)。
    True にすると設置/着弾判定を拡張版 (_has_placement_event_in_range_strict、
    CHAIN/GRAVITY_SETTLE も含む) に切り替える。連鎖確定由来の正当な色変化を
    見かけ上のフリッカから除外し、「本物のフリッカ件数」の下限推定を得る。
    """
    snaps = _distinct_stable_snapshots(records)
    chain_idxs = _new_chain_triggers(records)
    violations: list[Violation] = []
    for (prev_idx, prev_rec), (curr_idx, curr_rec) in zip(snaps, snaps[1:]):
        if _has_chain_trigger_in_range(records, chain_idxs, prev_rec.t_sec, curr_rec.t_sec):
            continue  # 連鎖 (消去→再充填) を挟む正当な色変化
        placement_excluded = (
            _has_placement_event_in_range_strict(records, prev_idx, curr_idx)
            if strict_exclusion
            else _has_placement_event_in_range(records, prev_idx, curr_idx)
        )
        if placement_excluded:
            continue  # 設置/着弾 (strict時は連鎖/重力settle含む) を挟む正当な色変化
        cells = _find_flicker_cells(prev_rec.grid, curr_rec.grid)
        if not cells:
            continue
        violations.append(Violation(
            type="color_flicker", severity=VIOLATION_TYPE_SEVERITY["color_flicker"],
            t_sec=curr_rec.t_sec, side=side, cells=cells, video_stem=video_stem,
            frame_idx=curr_rec.frame_idx,
            lag_flag=_confirm_lag_sec(records, curr_idx) > CONFIRM_LAG_WARN_SEC,
            detail=f"{len(cells)}セルで空を経由しない色変化 "
                   f"(t={prev_rec.t_sec:.2f}->{curr_rec.t_sec:.2f})",
            prev_frame_idx=prev_rec.frame_idx, prev_t_sec=prev_rec.t_sec,
            prev_grid=prev_rec.grid, curr_grid=curr_rec.grid,
        ))
    return violations


# ============================
# 類型2/3: 保存則違反 (新規)
# ============================


def _check_conservation(
    records: list[_FrameRecord], video_stem: str, side: str,
    *, strict_exclusion: bool = False,
) -> list[Violation]:
    """類型2/3: STABLE snapshot間のぷよ数増減が対応イベントなしで発生していないか。

    類型2(消失): 区間内に新規 chain trigger が無いのに減少 (相殺はぷよ数を
    変えないため対象外、連鎖消去のみが正当な減少要因)。
    類型3(出現): 区間内に TSUMO_FALL/OJAMA_FALL state (設置/着弾の代理検出) が
    無いのに増加。

    strict_exclusion: 2026-07-25 Step0 追加 (既定 False = 従来通り後方互換)。
    True にすると類型3判定の設置/着弾検出を拡張版 (CHAIN/GRAVITY_SETTLE 含む)
    に切り替える (_build_conservation_violation 参照)。
    """
    snaps = _distinct_stable_snapshots(records)
    chain_idxs = _new_chain_triggers(records)
    violations: list[Violation] = []
    for (prev_idx, prev_rec), (curr_idx, curr_rec) in zip(snaps, snaps[1:]):
        prev_n = Board.from_list(prev_rec.grid.tolist()).count_puyos()
        curr_n = Board.from_list(curr_rec.grid.tolist()).count_puyos()
        diff = curr_n - prev_n
        lag_flag = _confirm_lag_sec(records, curr_idx) > CONFIRM_LAG_WARN_SEC
        v = _build_conservation_violation(
            diff, records, chain_idxs, prev_idx, curr_idx, prev_rec, curr_rec,
            prev_n, curr_n, side, video_stem, lag_flag,
            strict_exclusion=strict_exclusion,
        )
        if v is not None:
            violations.append(v)
    return violations


def _build_conservation_violation(
    diff: int, records: list[_FrameRecord], chain_idxs: list[int],
    prev_idx: int, curr_idx: int, prev_rec: _FrameRecord, curr_rec: _FrameRecord,
    prev_n: int, curr_n: int, side: str, video_stem: str, lag_flag: bool,
    *, strict_exclusion: bool = False,
) -> Violation | None:
    """conservation 違反 1 件を判定して構築する (types 2/3 共通ロジック分離)。

    strict_exclusion: 2026-07-25 Step0 追加 (既定 False = 従来通り後方互換)。
    True で類型3 (出現) 判定を拡張版設置/着弾検出 (CHAIN/GRAVITY_SETTLE 含む)
    に切り替える。連鎖確定由来の正当な個数増加 (P3/P4経路) を除外する仮説検証用。
    """
    if diff < 0 and not _has_chain_trigger_in_range(
        records, chain_idxs, prev_rec.t_sec, curr_rec.t_sec,
    ):
        return Violation(
            type="conservation_loss", severity=VIOLATION_TYPE_SEVERITY["conservation_loss"],
            t_sec=curr_rec.t_sec, side=side, cells=[], video_stem=video_stem,
            frame_idx=curr_rec.frame_idx, lag_flag=lag_flag,
            detail=f"puyo数 {prev_n}->{curr_n} (Δ{diff}) だが区間内に連鎖イベントなし "
                   f"(t={prev_rec.t_sec:.2f}->{curr_rec.t_sec:.2f})",
            prev_frame_idx=prev_rec.frame_idx, prev_t_sec=prev_rec.t_sec,
            prev_grid=prev_rec.grid, curr_grid=curr_rec.grid,
        )
    gain_excluded = (
        _has_placement_event_in_range_strict(records, prev_idx, curr_idx)
        if strict_exclusion
        else _has_placement_event_in_range(records, prev_idx, curr_idx)
    )
    if diff > 0 and not gain_excluded:
        return Violation(
            type="conservation_gain", severity=VIOLATION_TYPE_SEVERITY["conservation_gain"],
            t_sec=curr_rec.t_sec, side=side, cells=[], video_stem=video_stem,
            frame_idx=curr_rec.frame_idx, lag_flag=lag_flag,
            detail=f"puyo数 {prev_n}->{curr_n} (Δ{diff}) だが区間内にTSUMO_FALL/OJAMA_FALL "
                   f"未検出 (t={prev_rec.t_sec:.2f}->{curr_rec.t_sec:.2f})",
            prev_frame_idx=prev_rec.frame_idx, prev_t_sec=prev_rec.t_sec,
            prev_grid=prev_rec.grid, curr_grid=curr_rec.grid,
        )
    return None


# ============================
# 類型9: 窒息整合違反 (新規)
# ============================


def _check_death_row_consistency(
    records: list[_FrameRecord], video_stem: str, side: str,
) -> list[Violation]:
    """類型9: is_dead()==True 後、MENU 遷移を経ずに STABLE が継続していないか。"""
    snaps = _distinct_stable_snapshots(records)
    violations: list[Violation] = []
    dead_active = False
    first_dead_t: float | None = None
    prev_snap_idx = 0
    for idx, rec in snaps:
        if dead_active:
            if _menu_occurred_between(records, prev_snap_idx + 1, idx):
                dead_active = False
            else:
                violations.append(_build_death_row_violation(
                    records, idx, rec, side, video_stem, first_dead_t,
                ))
                dead_active = False
        board = Board.from_list(rec.grid.tolist())
        if board.is_dead():
            dead_active = True
            first_dead_t = rec.t_sec
        prev_snap_idx = idx
    return violations


def _build_death_row_violation(
    records: list[_FrameRecord], idx: int, rec: _FrameRecord, side: str,
    video_stem: str, first_dead_t: float | None,
) -> Violation:
    """死亡整合違反 1 件を構築する。"""
    return Violation(
        type="death_row_inconsistency",
        severity=VIOLATION_TYPE_SEVERITY["death_row_inconsistency"],
        t_sec=rec.t_sec, side=side, cells=[(DEATH_ROW, DEATH_COL)],
        video_stem=video_stem, frame_idx=rec.frame_idx,
        lag_flag=_confirm_lag_sec(records, idx) > CONFIRM_LAG_WARN_SEC,
        detail=f"is_dead() at t={first_dead_t:.2f} 以降 t={rec.t_sec:.2f} でも"
               f"MENU遷移なくSTABLE継続",
    )


# ============================
# 類型1: おじゃま会計乖離 (骨組みのみ・未実装)
# ============================


def _check_ojama_accounting_rate_floor(
    records: list[_FrameRecord], video_stem: str, side: str,
) -> list[Violation]:
    """類型1: おじゃま会計 rate下限(=1)矛盾検出 (骨組みのみ・未実装)。

    本番経路 src/ojama_accounting.py (OjamaAccountingTracker + src/scoring.py
    compute_effective_rate) を基準に、「単一試合の全連鎖で rate下限(1)が
    100%発生 かつ 実経過<288s (=MARGIN_TIME_START_SEC 96s +
    (MARGIN_TIME_MAX_DECAYS-1)*MARGIN_TIME_DECAY_INTERVAL_SEC 16s 相当の
    正当な下限到達時間に満たない)」を矛盾フラグとする設計。

    未実装の理由: OjamaAccountingTracker を records から正確に再生するには
    on_state_transition への BoardState enum 変換 + on_tsumo_settled 発火
    タイミング (tsumo 着地イベント) の計装が本スクリプトの capture 層
    (_FrameRecord) に無く、誤った近似実装を入れるとかえって誤警報源になる
    (recognition_physics_review.py の会計指標が同種の理由で流用禁止と
    アーキ指定されている前例と同じ轍を踏む)。

    次の対応者への TODO:
        1. OjamaAccountingTracker を本関数内でインスタンス化し、
           records の (prev_state, curr_state, score, t_sec) を
           on_state_transition に順次投入して本番と同じ状態遷移を再生する。
        2. finalize (chain_end_triggered_p{1,2}=True) の瞬間の経過秒
           (tracker._elapsed 相当) を compute_effective_rate に通し、
           rate==OJAMA_RATE_MIN かどうかを判定する。
        3. 1試合(MENU境界)内の全 finalize で rate==OJAMA_RATE_MIN が
           100% かつ経過<288秒なら Violation を積む。
    """
    return []


# ============================
# 1 side 分の全類型実行
# ============================


def _audit_one_side(
    records: list[_FrameRecord], video_stem: str, side: str,
    *, strict_exclusion: bool = False,
) -> list[Violation]:
    """1 (video, side) 分の全類型チェックを実行する。

    strict_exclusion: 2026-07-25 Step0 追加 (既定 False = 従来通り後方互換)。
    color_flicker/conservation_gain の設置/着弾判定を拡張版
    (CHAIN/GRAVITY_SETTLE も含む) に切り替える。
    """
    violations: list[Violation] = []
    violations += _check_gravity(records, video_stem, side)
    violations += _check_connection_uncleared(records, video_stem, side)
    violations += _check_color_count(records, video_stem, side)
    violations += _check_color_flicker(records, video_stem, side, strict_exclusion=strict_exclusion)
    violations += _check_conservation(records, video_stem, side, strict_exclusion=strict_exclusion)
    violations += _check_death_row_consistency(records, video_stem, side)
    violations += _check_ojama_accounting_rate_floor(records, video_stem, side)
    return violations


# ============================
# 窓 (フル試合通し / スモーク) 解決
# ============================


def _video_duration_sec(video_stem: str) -> float:
    """動画全体の長さ (秒) を返す (matches.tsv が無い動画のフォールバック用)。"""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"video_{video_stem}.mp4"))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return n_frames / fps if fps > 0 else 0.0


def _load_full_match_windows(video_stem: str) -> list[tuple[float, float]]:
    """matches.tsv (match_boundaries_v5) から全試合区間を読む。

    DIAG_WINDOW_BY_STEM 未登録かつ matches.tsv も無い動画向けの最終フォールバック
    (動画全体を1区間として扱う)。2026-07-24 実用化修正: c62 でこの経路を通り
    数十分の動画全体を舐めて長時間化した実績があるため、既定4動画は
    _resolve_windows 側で DIAG_WINDOW_BY_STEM を優先し、この経路を通らない。
    未知動画向けの経路として残すが、事故再発防止に WARN を出す。
    """
    tsv_path = MATCH_BOUNDARIES_DIR / f"video_{video_stem}" / "matches.tsv"
    if not tsv_path.exists():
        print(
            f"[WARN] {video_stem}: matches.tsv 不在かつ DIAG_WINDOW_BY_STEM 未登録 "
            "→ 動画全体を1区間として処理します (長時間化に注意、既定4動画では通らないはずの経路)",
            flush=True,
        )
        return [(0.0, _video_duration_sec(video_stem))]
    windows: list[tuple[float, float]] = []
    with open(tsv_path, encoding="utf-8") as f:
        next(f, None)  # header 行をスキップ
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            windows.append((float(parts[1]), float(parts[2])))
    return windows


def _resolve_windows(args: argparse.Namespace) -> list[tuple[str, float, float]]:
    """CLI引数から処理対象窓 (stem, start_sec, max_sec) のリストを決定する。

    2026-07-24 実用化修正: DIAG_WINDOW_BY_STEM に登録済のstem (既定4動画) は
    A/B診断と同一の短窓を優先採用する (matches.tsv 不在動画=c62 で動画全体を
    1区間として舐める事故を防ぐ)。--start-sec/--max-sec を明示指定した場合は
    そちらを優先する (既存の任意区間診断用途を壊さない、後方互換維持)。
    """
    if args.smoke:
        return [(SMOKE_VIDEO_STEM, SMOKE_START_SEC, SMOKE_MAX_SEC)]
    stems = [s.strip() for s in args.videos.split(",") if s.strip()]
    if len(stems) == 1 and args.start_sec is not None and args.max_sec is not None:
        return [(stems[0], args.start_sec, args.max_sec)]
    windows: list[tuple[str, float, float]] = []
    for stem in stems:
        if stem in DIAG_WINDOW_BY_STEM:
            start, dur = DIAG_WINDOW_BY_STEM[stem]
            windows.append((stem, start, dur))
            continue
        for start, end in _load_full_match_windows(stem):
            windows.append((stem, start, max(0.0, end - start)))
    return windows


# ============================
# 進捗ログ計装 (2026-07-24 実用化修正)
# ============================


@dataclass
class _ProgressState:
    """1動画・1窓分の処理進捗 (直近ログ出力済み game-time 秒)。"""

    last_logged_sec: float


def _wrap_update_for_progress(
    orig_update: Callable[..., object], video_stem: str, progress: _ProgressState,
) -> Callable[..., object]:
    """RecognitionPipeline.update を計装ラップし、game-time 30秒毎に進捗ログを出す。

    src/recognition_pipeline.py 自体は変更しない (read-only制約遵守)。
    クラスメソッドの一時差し替えのみで、呼び出し側が必ず finally で復元する
    (scripts/_diag_ojama_fall_board_settle_ab_2026-07-24.py の
    _wrap_gravity_filter_for_counting と同型の計装ラップパターン)。
    """
    def _wrapped(self, frame_idx: int, time_sec: float, frame: np.ndarray):
        result = orig_update(self, frame_idx, time_sec, frame)
        if time_sec - progress.last_logged_sec >= PROGRESS_LOG_INTERVAL_SEC:
            progress.last_logged_sec = time_sec
            print(
                f"[{time.strftime('%H:%M:%S')}] [{video_stem}] "
                f"t={time_sec:.1f}s まで処理済み", flush=True,
            )
        return result
    return _wrapped


@contextmanager
def _log_progress_every_30s(video_stem: str, start_sec: float):
    """with 節内の RecognitionPipeline.update 呼び出しを進捗ログ計装する。

    with を抜けると必ず元の update に復元する (src 実質非改変の一時パッチ)。
    """
    progress = _ProgressState(last_logged_sec=start_sec)
    orig_update = RecognitionPipeline.update
    RecognitionPipeline.update = _wrap_update_for_progress(orig_update, video_stem, progress)
    try:
        yield
    finally:
        RecognitionPipeline.update = orig_update


def _format_video_type_subtotal(violations: list[Violation]) -> str:
    """1動画(1窓)分の類型別カウント小計を1行テキストにする (進捗ログ用)。"""
    by_type: dict[str, int] = {}
    for v in violations:
        by_type[v.type] = by_type.get(v.type, 0) + 1
    if not by_type:
        return "類型別小計: (違反なし)"
    ranked = sorted(by_type.items(), key=lambda kv: -kv[1])
    return "類型別小計: " + ", ".join(f"{name}={n}" for name, n in ranked)


# ============================
# frame crop 出力 (実画素 + 赤枠ハイライト)
# ============================


def _crop_board_region(frame: np.ndarray, region: object) -> tuple[np.ndarray, int, int]:
    """盤面領域 (隠し段含む余白付き) を crop する。戻り値 (crop, x1, y1)、x1/y1はframe上のcrop左上座標。

    2026-07-24 3コマmontage拡張: 元 _crop_board_with_highlight から crop 計算部分のみ
    切り出した (ハイライト描画と分離し、montage側のconfirmed_boardオーバーレイでも
    同一座標系を再利用するため)。
    """
    cell_h = int(region.cell_height)  # type: ignore[attr-defined]
    x1 = region.x - FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    y1 = region.y - cell_h - FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    x2 = region.x + region.width + FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    y2 = region.y + region.height + FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    return frame[y1:y2, x1:x2].copy(), x1, y1


def _draw_cell_highlight_boxes(
    crop: np.ndarray, region: object, cells: list[tuple[int, int]], x1: int, y1: int,
) -> None:
    """矛盾セルを赤枠でハイライトする (crop に in-place 描画、x1/y1はcrop左上のframe座標)。"""
    half_w = int(region.cell_width / 2)  # type: ignore[attr-defined]
    half_h = int(region.cell_height / 2)  # type: ignore[attr-defined]
    for row, col in cells:
        cx, cy = region.cell_center(row, col)  # type: ignore[attr-defined]
        rx1, ry1 = cx - half_w - x1, cy - half_h - y1
        rx2, ry2 = cx + half_w - x1, cy + half_h - y1
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)


def _crop_board_with_highlight(
    frame: np.ndarray, region: object, cells: list[tuple[int, int]],
) -> np.ndarray:
    """盤面領域 (隠し段含む) を crop し、矛盾セルを赤枠でハイライトする (単パネル用、既存動作維持)。"""
    crop, x1, y1 = _crop_board_region(frame, region)
    _draw_cell_highlight_boxes(crop, region, cells, x1, y1)
    return crop


def _draw_confirmed_board_overlay(
    crop: np.ndarray, grid: np.ndarray | None, region: object, x1: int, y1: int,
) -> None:
    """crop 上に confirmed_board の認識色シンボルを重畳する (in-place)。

    visualize_recognition.draw_cell_overlay をそのまま流用する (描画ロジックの
    重複実装を避ける)。draw_cell_overlay は roi_x/roi_y を「描画先画像内での
    ROI原点」として使うだけなので、frame全体でなくcropを渡してもcrop内座標系
    (region.x - x1, region.y - y1) を指定すれば正しく描画される。
    """
    if grid is None:
        return
    board = Board.from_list(grid.tolist())
    roi_x_local = int(region.x - x1)  # type: ignore[attr-defined]
    roi_y_local = int(region.y - y1)  # type: ignore[attr-defined]
    draw_cell_overlay(crop, board, roi_x_local, roi_y_local)


def _label_montage_panel(panel: np.ndarray, text: str) -> None:
    """montageコマ内の簡潔ラベル (ASCII、CJKグリフ欠落のため日本語不使用)。"""
    cv2.putText(
        panel, text, (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5,
        (0, 255, 255), 1, cv2.LINE_AA,
    )


def _write_one_violation_frame(v: Violation, frames_dir: Path) -> bool:
    """1 件の違反に対応する実画面 crop を書き出す。"""
    video_path = VIDEO_DIR / f"video_{v.video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = v.frame_idx if v.frame_idx >= 0 else int(round(v.t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    region = DEFAULT_P1_REGION if v.side == "1P" else DEFAULT_P2_REGION
    crop = _crop_board_with_highlight(frame, region, v.cells)
    label = f"{v.t_sec:.2f}".replace(".", "_")
    fname = f"{v.type}_{v.video_stem}_{v.side}_t{label}.png"
    cv2.imwrite(str(frames_dir / fname), crop)
    return True


# ============================
# 2026-07-24 3コマmontage拡張: BEFORE(STABLE)/AFTER(STABLE)/ACTUAL 出力
# ============================


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """cap から frame_idx の1枚を読み込み、1920x1080に正規化して返す (読めなければNone)。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _panel_prev_stable(
    prev_frame: np.ndarray | None, v: Violation, region: object,
) -> np.ndarray | None:
    """左コマ「変化前STABLE」(confirmed_board重畳+赤枠) を作る。prev不在ならNoneを返す。"""
    if prev_frame is None:
        return None
    crop, x1, y1 = _crop_board_region(prev_frame, region)
    _draw_confirmed_board_overlay(crop, v.prev_grid, region, x1, y1)
    _draw_cell_highlight_boxes(crop, region, v.cells, x1, y1)
    _label_montage_panel(crop, f"BEFORE(STABLE) t={v.prev_t_sec:.2f}s")
    return crop


def _panel_curr_stable(curr_frame: np.ndarray, v: Violation, region: object) -> np.ndarray:
    """中コマ「変化後STABLE」(confirmed_board重畳+赤枠) を作る。"""
    crop, x1, y1 = _crop_board_region(curr_frame, region)
    _draw_confirmed_board_overlay(crop, v.curr_grid, region, x1, y1)
    _draw_cell_highlight_boxes(crop, region, v.cells, x1, y1)
    _label_montage_panel(crop, f"AFTER(STABLE) t={v.t_sec:.2f}s")
    return crop


def _panel_actual_screen(curr_frame: np.ndarray, v: Violation, region: object) -> np.ndarray:
    """右コマ「実画面」(オーバーレイなし、赤枠のみ) を作る。"""
    crop, x1, y1 = _crop_board_region(curr_frame, region)
    _draw_cell_highlight_boxes(crop, region, v.cells, x1, y1)
    _label_montage_panel(crop, f"ACTUAL SCREEN t={v.t_sec:.2f}s")
    return crop


def _na_panel(like: np.ndarray) -> np.ndarray:
    """prev snapshot不在時 (最初のsnapshot等) の空欄コマ (N/A表示、安全なフォールバック)。"""
    panel = np.zeros_like(like)
    cv2.putText(
        panel, "N/A (no prev snapshot)", (6, panel.shape[0] // 2),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def _stack_montage_panels(panels: list[np.ndarray]) -> np.ndarray:
    """3コマを同一高さ/幅にpaddingし、区切り線付きで横並びにする。"""
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)

    def _pad(img: np.ndarray) -> np.ndarray:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[: img.shape[0], : img.shape[1]] = img
        return out

    sep = np.full((h, MONTAGE_SEPARATOR_PX, 3), (255, 255, 255), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i > 0:
            parts.append(sep)
        parts.append(_pad(p))
    return np.hstack(parts)


def _build_montage_banner_text(v: Violation) -> str:
    """montage上部の英数字サマリラベル (類型・動画・時刻・prev/curr色、CJKグリフ欠落回避のためASCIIのみ)。"""
    text = (
        f"{v.type} id={VIOLATION_TYPE_ID.get(v.type, -1)} vid={v.video_stem} "
        f"{v.side} t={v.t_sec:.2f}s n_cells={len(v.cells)}"
    )
    if v.cells and v.prev_grid is not None and v.curr_grid is not None:
        row, col = v.cells[0]
        prev_c, curr_c = int(v.prev_grid[row, col]), int(v.curr_grid[row, col])
        text += f" cell0(row{row},col{col})_color:{prev_c}->{curr_c}"
    return text


def _write_one_violation_montage(v: Violation, frames_dir: Path) -> bool:
    """1件の違反に対応する3コマmontage (BEFORE STABLE/AFTER STABLE/ACTUAL) を書き出す。

    prev snapshot が無い場合は左コマを N/A にする (安全なフォールバック、user制約準拠)。
    """
    video_path = VIDEO_DIR / f"video_{v.video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    region = DEFAULT_P1_REGION if v.side == "1P" else DEFAULT_P2_REGION
    curr_frame_idx = v.frame_idx if v.frame_idx >= 0 else int(round(v.t_sec * fps))
    curr_frame = _read_frame_at(cap, curr_frame_idx)
    prev_frame = _read_frame_at(cap, v.prev_frame_idx) if v.prev_frame_idx >= 0 else None
    cap.release()
    if curr_frame is None:
        return False

    panel_curr = _panel_curr_stable(curr_frame, v, region)
    panel_actual = _panel_actual_screen(curr_frame, v, region)
    panel_prev = _panel_prev_stable(prev_frame, v, region)
    if panel_prev is None:
        panel_prev = _na_panel(panel_curr)

    montage = _stack_montage_panels([panel_prev, panel_curr, panel_actual])
    banner = np.zeros((MONTAGE_BANNER_HEIGHT_PX, montage.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner, _build_montage_banner_text(v), (6, 18),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )
    out = np.vstack([banner, montage])
    label = f"{v.t_sec:.2f}".replace(".", "_")
    fname = f"{v.type}_{v.video_stem}_{v.side}_t{label}_montage.png"
    cv2.imwrite(str(frames_dir / fname), out)
    return True


def _write_top_violation_frames(violations: list[Violation], frames_dir: Path) -> int:
    """類型ごとに発生件数上位の frame crop を出力する。

    2026-07-24 3コマmontage拡張: MONTAGE_TYPES 対象は (type, video_stem) 単位で
    上位 MAX_MONTAGE_FRAMES_PER_TYPE_PER_VIDEO 件、それ以外は既存通り type 単位
    (全動画合算) で上位 MAX_FRAMES_PER_TYPE 件を出す (既存挙動を維持)。
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    montage_group: dict[tuple[str, str], list[Violation]] = {}
    single_group: dict[str, list[Violation]] = {}
    for v in violations:
        if v.type in MONTAGE_TYPES:
            montage_group.setdefault((v.type, v.video_stem), []).append(v)
        else:
            single_group.setdefault(v.type, []).append(v)
    n_written = 0
    for vlist in montage_group.values():
        for v in vlist[:MAX_MONTAGE_FRAMES_PER_TYPE_PER_VIDEO]:
            if _write_one_violation_montage(v, frames_dir):
                n_written += 1
    for vlist in single_group.values():
        for v in vlist[:MAX_FRAMES_PER_TYPE]:
            if _write_one_violation_frame(v, frames_dir):
                n_written += 1
    return n_written


# ============================
# サマリ集計・出力
# ============================


def _build_summary(
    violations: list[Violation], duration_min_by_key: dict[str, float],
) -> dict:
    """類型別 count/rate_per_min/lag_flag_rate を発生件数降順ランクで集計する。"""
    total_min = sum(duration_min_by_key.values()) or 1e-6
    by_type: dict[str, list[Violation]] = {}
    for v in violations:
        by_type.setdefault(v.type, []).append(v)
    ranked = [_summarize_one_type(vtype, vlist, total_min) for vtype, vlist in by_type.items()]
    ranked.sort(key=lambda r: -r["count"])
    return {
        "total_duration_min": total_min, "n_total_violations": len(violations),
        "known_unimplemented_types": [
            "ojama_accounting_rate_floor(#1,骨組みのみ)",
            "ojama_drop_desync(#16,#45,担当別)", "placement_drop_desync(#17,#47,担当別)",
        ],
        "by_type_ranked": ranked,
    }


def _summarize_one_type(vtype: str, vlist: list[Violation], total_min: float) -> dict:
    """1 類型分の集計行 (count/rate_per_min/lag_flag_rate/動画別内訳) を作る。"""
    n = len(vlist)
    n_lag = sum(1 for v in vlist if v.lag_flag)
    by_video: dict[str, int] = {}
    for v in vlist:
        by_video[v.video_stem] = by_video.get(v.video_stem, 0) + 1
    return {
        "type": vtype, "type_id": VIOLATION_TYPE_ID.get(vtype, -1),
        "severity": VIOLATION_TYPE_SEVERITY.get(vtype, "unknown"),
        "count": n, "rate_per_min": n / total_min,
        "lag_flag_rate": (n_lag / n) if n else 0.0,
        "by_video": by_video,
    }


def _format_summary_text(summary: dict, n_frames_written: int) -> str:
    """サマリを人間可読なテキストに整形する。"""
    lines = [
        "==== 盤面物理矛盾 全類型監査 サマリ (#48) ====",
        f"総処理時間(分): {summary['total_duration_min']:.2f}",
        f"総違反件数: {summary['n_total_violations']}",
        f"フレーム出力: {n_frames_written} 件",
        "--- 類型別 (発生件数降順) ---",
    ]
    for r in summary["by_type_ranked"]:
        lines.append(
            f"  類型{r['type_id']:>2} {r['type']:28s} severity={r['severity']:6s} "
            f"count={r['count']:4d} rate/min={r['rate_per_min']:.3f} "
            f"lag_flag_rate={r['lag_flag_rate']:.2f} by_video={r['by_video']}",
        )
    lines.append("--- 未実装/担当別 (登録のみ、本スクリプトでは検出しない) ---")
    lines.extend(f"  {name}" for name in summary["known_unimplemented_types"])
    return "\n".join(lines)


# ============================
# CLI / main
# ============================


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする。"""
    ap = argparse.ArgumentParser(description="盤面物理矛盾 全類型監査器 (#48)")
    ap.add_argument(
        "--videos", type=str, default=",".join(DEFAULT_VIDEO_STEMS),
        help="対象動画stemのカンマ区切り (既定: c62,30,35,38)。既定4動画は"
             "DIAG_WINDOW_BY_STEM (A/B診断と同一の短窓) を優先採用する。"
             "未登録stemはmatches.tsvから全試合区間を読む (無ければ動画全体を1区間"
             "扱い、長時間化に注意)。",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help=f"スモークモード: video_{SMOKE_VIDEO_STEM} の短窓 "
             f"({SMOKE_START_SEC}-{SMOKE_START_SEC + SMOKE_MAX_SEC}s) のみ同期実行する。",
    )
    ap.add_argument(
        "--start-sec", type=float, default=None, dest="start_sec",
        help="--videos に単一stemを指定した場合の開始秒上書き (任意区間診断用)。",
    )
    ap.add_argument(
        "--max-sec", type=float, default=None, dest="max_sec",
        help="--start-sec と併用する処理秒数。",
    )
    ap.add_argument(
        "--strict-exclusion", action="store_true", dest="strict_exclusion",
        help="2026-07-25 Step0 追加: color_flicker/conservation_gain の設置/着弾"
             "判定に CHAIN/GRAVITY_SETTLE も含める拡張版を使う (既定 False = "
             "従来通り TSUMO_FALL/OJAMA_FALL のみ)。連鎖確定由来の正当な色変化を"
             "除外し「本物のフリッカ件数」下限推定を得るための比較用フラグ。",
    )
    ap.add_argument(
        "--enable-placement-color-cnn-check", action="store_true",
        dest="enable_placement_color_cnn_check",
        help="2026-07-25 甲修正の効果測定用: RecognitionPipeline の設置時色CNN照合 "
             "(src/recognition_pipeline.py enable_placement_color_cnn_check) を "
             "有効化してフレーム収集する (既定 False = 従来通り無効、read-only診断)。"
             "color_flicker 等の件数が甲修正でどう変わるかを比較する。",
    )
    return ap.parse_args()


def main() -> None:
    """メイン処理: 対象窓を処理し、違反を検出して summary.json/txt + frames を出力する。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない (アーキ指定)
    args = _parse_args()
    windows = _resolve_windows(args)
    print(
        f"[INFO] 対象 {len(windows)} 窓 (smoke={args.smoke}, "
        f"enable_placement_color_cnn_check={args.enable_placement_color_cnn_check})",
    )
    all_violations: list[Violation] = []
    duration_min_by_key: dict[str, float] = {}
    for stem, start_sec, max_sec in windows:
        print(
            f"[{time.strftime('%H:%M:%S')}] [{stem}] 開始 "
            f"start={start_sec:.1f}s dur={max_sec:.1f}s を処理中...", flush=True,
        )
        t_video_start = time.time()
        with _log_progress_every_30s(stem, start_sec):
            by_side = _capture_frames(
                stem, start_sec, max_sec,
                enable_placement_color_cnn_check=args.enable_placement_color_cnn_check,
            )
        # 2026-07-24 FP修正: 試合外/演出テロップ区間を独立検出し、両 side 共通で
        # 除外する (テロップは画面全体に表示されるため side 非依存)。
        telop_intervals = _scan_telop_exclusion_intervals(stem, start_sec, max_sec)
        if telop_intervals:
            n_masked_sec = sum(hi - lo for lo, hi in telop_intervals)
            print(
                f"[{time.strftime('%H:%M:%S')}] [{stem}] 試合外/演出テロップ除外区間 "
                f"{len(telop_intervals)} 件 (計 {n_masked_sec:.1f}s): {telop_intervals}",
                flush=True,
            )
        video_violations: list[Violation] = []
        for side in ("1P", "2P"):
            records = _mask_records_by_excluded_intervals(by_side[side], telop_intervals)
            if not records:
                continue
            side_violations = _audit_one_side(
                records, stem, side, strict_exclusion=args.strict_exclusion,
            )
            video_violations += side_violations
            all_violations += side_violations
            dur_min = max(1e-6, (records[-1].t_sec - records[0].t_sec) / 60.0)
            key = f"{stem}_{side}"
            duration_min_by_key[key] = duration_min_by_key.get(key, 0.0) + dur_min
        print(
            f"[{time.strftime('%H:%M:%S')}] [{stem}] 完了 "
            f"({time.time() - t_video_start:.1f}s) {_format_video_type_subtotal(video_violations)}",
            flush=True,
        )
    summary = _build_summary(all_violations, duration_min_by_key)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    n_frames = _write_top_violation_frames(all_violations, RUN_DIR / "frames")
    (RUN_DIR / "summary.json").write_text(
        json.dumps({"summary": summary}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    text = _format_summary_text(summary, n_frames)
    (RUN_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"\n[DONE] {RUN_DIR} に保存しました (frames: {n_frames} 件)")
    print(text)


if __name__ == "__main__":
    main()
