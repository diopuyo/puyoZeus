"""盤面物理矛盾 全類型監査器 (#48・user要望の恒久エンジン)。

個別バグ追跡ではなく「物理的に不可能な盤面状態/遷移」を全類型スキャンし、
頻度順に潰すための常設監視の土台。read-only (src 非改変)。

confirmed_board (STABLE 専用) と estimated_board (連鎖中物理推定) を分離し、
過渡状態 (CHAIN/GRAVITY_SETTLE 中の物理推定盤面) を誤検出しない
(= board_provenance=="observed" かつ state==STABLE のフレームのみ判定対象)。

capture層 (フレーム取得・chain trigger 検出) は重複実装せず
scripts.recognition_physics_review から import して流用する。

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
import json
import os
import sys
from dataclasses import dataclass, field
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
from src.self_supervised.physical_consistency import (  # noqa: E402
    check_color_count, check_gravity_rule, check_no_pre_chain_4_plus_connection,
)
from scripts.recognition_physics_review import (  # noqa: E402
    _FrameRecord, _capture_frames, _new_chain_triggers,
)

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

# 類型ごとに frame crop を出力する上位件数。
MAX_FRAMES_PER_TYPE: int = 3

# frame crop の余白 (px)。
FRAME_CROP_MARGIN_PX: int = 30

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


def _menu_occurred_between(records: list[_FrameRecord], i_lo: int, i_hi: int) -> bool:
    """[i_lo, i_hi) 区間に MENU state のフレームがあるか (試合終了の正常検知)。"""
    return any(records[k].state == BoardState.MENU.name for k in range(i_lo, i_hi))


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
    """類型6: 4連結以上消去漏れ。"""
    violations: list[Violation] = []
    for idx, rec in _distinct_stable_snapshots(records):
        board = Board.from_list(rec.grid.tolist())
        is_valid, clusters = check_no_pre_chain_4_plus_connection(board)
        if is_valid:
            continue
        lag_flag = _confirm_lag_sec(records, idx) > CONFIRM_LAG_WARN_SEC
        for cluster in clusters:
            violations.append(Violation(
                type="connection_uncleared",
                severity=VIOLATION_TYPE_SEVERITY["connection_uncleared"],
                t_sec=rec.t_sec, side=side, cells=cluster["cells"], video_stem=video_stem,
                frame_idx=rec.frame_idx, lag_flag=lag_flag,
                detail=f"色{cluster['color']}の{len(cluster['cells'])}連結がSTABLEで未消去",
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


def _check_color_flicker(records: list[_FrameRecord], video_stem: str, side: str) -> list[Violation]:
    """類型4: 色フリッカ (STABLE observed snapshot間の直接色変化)。"""
    snaps = _distinct_stable_snapshots(records)
    violations: list[Violation] = []
    for (_, prev_rec), (curr_idx, curr_rec) in zip(snaps, snaps[1:]):
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
        ))
    return violations


# ============================
# 類型2/3: 保存則違反 (新規)
# ============================


def _check_conservation(records: list[_FrameRecord], video_stem: str, side: str) -> list[Violation]:
    """類型2/3: STABLE snapshot間のぷよ数増減が対応イベントなしで発生していないか。

    類型2(消失): 区間内に新規 chain trigger が無いのに減少 (相殺はぷよ数を
    変えないため対象外、連鎖消去のみが正当な減少要因)。
    類型3(出現): 区間内に TSUMO_FALL/OJAMA_FALL state (設置/着弾の代理検出) が
    無いのに増加。
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
        )
        if v is not None:
            violations.append(v)
    return violations


def _build_conservation_violation(
    diff: int, records: list[_FrameRecord], chain_idxs: list[int],
    prev_idx: int, curr_idx: int, prev_rec: _FrameRecord, curr_rec: _FrameRecord,
    prev_n: int, curr_n: int, side: str, video_stem: str, lag_flag: bool,
) -> Violation | None:
    """conservation 違反 1 件を判定して構築する (types 2/3 共通ロジック分離)。"""
    if diff < 0 and not _has_chain_trigger_in_range(
        records, chain_idxs, prev_rec.t_sec, curr_rec.t_sec,
    ):
        return Violation(
            type="conservation_loss", severity=VIOLATION_TYPE_SEVERITY["conservation_loss"],
            t_sec=curr_rec.t_sec, side=side, cells=[], video_stem=video_stem,
            frame_idx=curr_rec.frame_idx, lag_flag=lag_flag,
            detail=f"puyo数 {prev_n}->{curr_n} (Δ{diff}) だが区間内に連鎖イベントなし "
                   f"(t={prev_rec.t_sec:.2f}->{curr_rec.t_sec:.2f})",
        )
    if diff > 0 and not _has_placement_event_in_range(records, prev_idx, curr_idx):
        return Violation(
            type="conservation_gain", severity=VIOLATION_TYPE_SEVERITY["conservation_gain"],
            t_sec=curr_rec.t_sec, side=side, cells=[], video_stem=video_stem,
            frame_idx=curr_rec.frame_idx, lag_flag=lag_flag,
            detail=f"puyo数 {prev_n}->{curr_n} (Δ{diff}) だが区間内にTSUMO_FALL/OJAMA_FALL "
                   f"未検出 (t={prev_rec.t_sec:.2f}->{curr_rec.t_sec:.2f})",
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


def _audit_one_side(records: list[_FrameRecord], video_stem: str, side: str) -> list[Violation]:
    """1 (video, side) 分の全類型チェックを実行する。"""
    violations: list[Violation] = []
    violations += _check_gravity(records, video_stem, side)
    violations += _check_connection_uncleared(records, video_stem, side)
    violations += _check_color_count(records, video_stem, side)
    violations += _check_color_flicker(records, video_stem, side)
    violations += _check_conservation(records, video_stem, side)
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
    """matches.tsv (match_boundaries_v5) から全試合区間を読む (フル試合通し前提)。

    無い動画 (例: c62、既に1試合分にクリップ済み) は動画全体を1区間として扱う。
    """
    tsv_path = MATCH_BOUNDARIES_DIR / f"video_{video_stem}" / "matches.tsv"
    if not tsv_path.exists():
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
    """CLI引数から処理対象窓 (stem, start_sec, max_sec) のリストを決定する。"""
    if args.smoke:
        return [(SMOKE_VIDEO_STEM, SMOKE_START_SEC, SMOKE_MAX_SEC)]
    stems = [s.strip() for s in args.videos.split(",") if s.strip()]
    if len(stems) == 1 and args.start_sec is not None and args.max_sec is not None:
        return [(stems[0], args.start_sec, args.max_sec)]
    windows: list[tuple[str, float, float]] = []
    for stem in stems:
        for start, end in _load_full_match_windows(stem):
            windows.append((stem, start, max(0.0, end - start)))
    return windows


# ============================
# frame crop 出力 (実画素 + 赤枠ハイライト)
# ============================


def _crop_board_with_highlight(
    frame: np.ndarray, region: object, cells: list[tuple[int, int]],
) -> np.ndarray:
    """盤面領域 (隠し段含む) を crop し、矛盾セルを赤枠でハイライトする。"""
    cell_h = int(region.cell_height)  # type: ignore[attr-defined]
    x1 = region.x - FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    y1 = region.y - cell_h - FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    x2 = region.x + region.width + FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    y2 = region.y + region.height + FRAME_CROP_MARGIN_PX  # type: ignore[attr-defined]
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    crop = frame[y1:y2, x1:x2].copy()
    for row, col in cells:
        cx, cy = region.cell_center(row, col)  # type: ignore[attr-defined]
        half_w = int(region.cell_width / 2)  # type: ignore[attr-defined]
        half_h = int(region.cell_height / 2)  # type: ignore[attr-defined]
        rx1, ry1 = cx - half_w - x1, cy - half_h - y1
        rx2, ry2 = cx + half_w - x1, cy + half_h - y1
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
    return crop


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


def _write_top_violation_frames(violations: list[Violation], frames_dir: Path) -> int:
    """類型ごとに発生件数上位 MAX_FRAMES_PER_TYPE 件の frame crop を出力する。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[Violation]] = {}
    for v in violations:
        by_type.setdefault(v.type, []).append(v)
    n_written = 0
    for vlist in by_type.values():
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
        help="対象動画stemのカンマ区切り (既定: c62,30,35,38)。フル試合通し前提、"
             "各stemのmatches.tsvから全試合区間を読む (無ければ動画全体を1区間扱い)。",
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
    return ap.parse_args()


def main() -> None:
    """メイン処理: 対象窓を処理し、違反を検出して summary.json/txt + frames を出力する。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない (アーキ指定)
    args = _parse_args()
    windows = _resolve_windows(args)
    print(f"[INFO] 対象 {len(windows)} 窓 (smoke={args.smoke})")
    all_violations: list[Violation] = []
    duration_min_by_key: dict[str, float] = {}
    for stem, start_sec, max_sec in windows:
        print(f"  {stem}: start={start_sec:.1f}s dur={max_sec:.1f}s を処理中...")
        by_side = _capture_frames(stem, start_sec, max_sec)
        for side in ("1P", "2P"):
            records = by_side[side]
            if not records:
                continue
            all_violations += _audit_one_side(records, stem, side)
            dur_min = max(1e-6, (records[-1].t_sec - records[0].t_sec) / 60.0)
            key = f"{stem}_{side}"
            duration_min_by_key[key] = duration_min_by_key.get(key, 0.0) + dur_min
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
