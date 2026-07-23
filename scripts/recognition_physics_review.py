"""物理推論による認識品質の自動レビュー機構(人の目の代わりの客観指標)。

背景: 「連鎖終了後に消えたぷよが残像で残る=連鎖後盤面が即反映されない」
基幹認識バグ(user目視確定)を直す際、改善したかを物理整合性で自動判定する
ための計測器。src/ は一切変更しない(読み取り専用、
scripts/measure_opponent_estimate_accuracy.py と同じ外部消費パターン)。

計測する物理整合性違反 (全て「低いほど良い」で統一):
    1. 残像/連鎖後不一致 (最重要): 連鎖後 最初の STABLE confirmed_board が
       ChainSimulator.simulate(連鎖前盤面).final_board と一致するか。
       不一致セル数・不一致率 + 「連鎖終了〜正しい盤面反映までの遅延秒数」
       (残像持続時間)。
    2. 浮きぷよ (重力整合): src/self_supervised/physical_consistency.py
       の check_gravity_rule を流用。
    3. 色ワープ/変色: 既存 PuyoErasureMonitor/StableTransitionMonitor が
       自動生成する erasure_alerts / transition_drop_alerts を集計。
    4. お邪魔会計整合: ChainEvent.ojama_sent (発火側の期待送り量) と
       相手盤面の可視お邪魔増加 (実測着弾量、measure_ojama_landing_delay.py
       の着弾検出ロジックを流用) の乖離。
    5. is_match_active 誤 False の発生率: CHAIN 状態中に confirmed_board が
       None になる割合 (Stage 0 で発見した既知の穴、cf.
       measure_opponent_estimate_accuracy.py)。

使い方:
    PYTHONPATH=. python -m scripts.recognition_physics_review
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠。
# 7時間自律運転中のため特に厳守)。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board, COLOR_OJAMA, HIDDEN_ROWS  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.self_supervised.physical_consistency import check_gravity_rule  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.measure_ojama_landing_delay import (  # noqa: E402
    _visible_ojama_counts, _find_landing, OPP_GAP_THRESHOLD_SEC,
    MAX_LANDING_SEARCH_SEC,
)

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_physics_review"

# 対象動画・処理窓 (stem, start_sec, max_sec)。c62(マスター、大連鎖多い)必須 +
# c82(S級) + c11(チャレンジャー、ティアバランス)。既知の実連鎖を含む窓を選ぶ。
TARGET_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("c62", 895.0, 65.0),
    ("c82", 960.0, 50.0),
    ("c11", 585.0, 60.0),
)

# 残像持続時間の探索上限秒 (これを超えても正しい盤面に一致しなければ
# "未解消" として扱う、打ち切り値としても使う)。
GHOST_SEARCH_MAX_SEC: float = 15.0

# データ品質フラグ: 連鎖発火〜「最初の STABLE」までの遅延がこれを超えたら
# lag_flag=True とする (単純な1連鎖なら数秒で STABLE 復帰するはずのため、
# それ以上かかった場合は「別の後続手番を挟んだ後の STABLE」を誤って比較対象に
# している疑いがあり、mismatch を額面通り受け取れない可能性を明示する)。
GHOST_FIRST_STABLE_LAG_WARN_SEC: float = 5.0

# 盤面セル総数 (13行×6列)
TOTAL_CELLS: int = 13 * 6


@dataclass
class _FrameRecord:
    """1 (video, side, frame) 分の観測値。"""
    frame_idx: int
    t_sec: float
    state: str
    grid: np.ndarray | None
    score: int | None
    chain_trigger_sec: float | None
    chain_before_grid: np.ndarray | None
    chain_ojama_sent: int | None
    n_erasure_alerts: int
    n_transition_drop_alerts: int
    # 反復4 (2026-07-23): confirmed_board=None の理由分類 (診断計装、
    # src.recognition_pipeline.SideResult.board_none_reason をそのまま透過)。
    # None(非該当) / "cold_start" / "menu_reset" / "chain_hold_none" / "other"。
    board_none_reason: str | None = None


def _capture_frames(
    video_stem: str, start_sec: float, max_sec: float,
    enable_chain_exit_warmup: bool = False,
    enable_chain_exit_next_signal: bool = False,
) -> dict[str, list[_FrameRecord]]:
    """1 動画・1 窓分を RecognitionPipeline で処理し、side別に記録を返す。

    enable_chain_exit_warmup / enable_chain_exit_next_signal: 2026-07-23
    P1 実験用の optional 引数 (既定 False = 従来通り、後方互換)。
    src/recognition_pipeline.py の既存 runtime flag をそのまま透過する
    (src 無改修、既存 default False で挙動不変)。
    """
    video_path = VIDEO_DIR / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}", file=sys.stderr)
        return {"1P": [], "2P": []}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
    )
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(video_stem)

    records: dict[str, list[_FrameRecord]] = {"1P": [], "2P": []}
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        for side, side_result in (("1P", result.p1), ("2P", result.p2)):
            records[side].append(_build_record(fi, t_sec, side_result))
    cap.release()
    return records


def _build_record(fi: int, t_sec: float, side_result: object) -> _FrameRecord:
    """SideResult から 1 frame 分の _FrameRecord を組み立てる。"""
    grid = side_result.confirmed_board._grid.copy() if side_result.confirmed_board else None
    ce = side_result.chain_event
    return _FrameRecord(
        frame_idx=fi, t_sec=t_sec, state=side_result.state.name, grid=grid,
        score=side_result.score,
        chain_trigger_sec=float(ce.trigger_sec) if ce is not None else None,
        chain_before_grid=ce.before_board._grid.copy() if ce is not None else None,
        chain_ojama_sent=int(ce.ojama_sent) if ce is not None else None,
        n_erasure_alerts=len(side_result.erasure_alerts or []),
        n_transition_drop_alerts=len(side_result.transition_drop_alerts or []),
        # getattr: 反復4 計装フィールド未搭載の古い SideResult でも動く安全策。
        board_none_reason=getattr(side_result, "board_none_reason", None),
    )


# ============================
# 1. 残像/連鎖後不一致
# ============================


def _new_chain_triggers(records: list[_FrameRecord]) -> list[int]:
    """chain_event が新規出現したフレームの index 一覧を返す (trigger_sec 変化で検出)。"""
    idxs: list[int] = []
    last_trigger: float | None = None
    for i, rec in enumerate(records):
        if rec.chain_trigger_sec is not None and rec.chain_trigger_sec != last_trigger:
            idxs.append(i)
            last_trigger = rec.chain_trigger_sec
        elif rec.chain_trigger_sec is None:
            last_trigger = None
    return idxs


def _find_first_stable_after(records: list[_FrameRecord], start_idx: int) -> int | None:
    """start_idx 以降で最初に state==STABLE かつ grid が有効なフレームの index を返す。"""
    for i in range(start_idx, len(records)):
        if records[i].state == BoardState.STABLE.name and records[i].grid is not None:
            return i
    return None


def _find_ghost_resolution(
    records: list[_FrameRecord], first_stable_idx: int, expected: np.ndarray,
) -> tuple[float, bool]:
    """first_stable_idx 以降で expected と一致する最初のフレームまでの遅延秒を返す。

    Returns: (ghost_duration_sec, resolved)。GHOST_SEARCH_MAX_SEC 以内に
    一致しなければ (GHOST_SEARCH_MAX_SEC, False) を打ち切り値として返す。
    """
    t0 = records[first_stable_idx].t_sec
    for i in range(first_stable_idx, len(records)):
        rec = records[i]
        if rec.t_sec - t0 > GHOST_SEARCH_MAX_SEC:
            break
        if rec.grid is not None and np.array_equal(rec.grid, expected):
            return rec.t_sec - t0, True
    return GHOST_SEARCH_MAX_SEC, False


def _measure_ghost_mismatch(records: list[_FrameRecord], sim: ChainSimulator) -> list[dict]:
    """各連鎖イベントの残像/連鎖後不一致を計測する。"""
    results: list[dict] = []
    for idx in _new_chain_triggers(records):
        rec = records[idx]
        try:
            before = Board.from_list(rec.chain_before_grid.tolist())
            sim_result = sim.simulate(before)
        except Exception:
            continue
        if sim_result.chain_count < 1:
            continue  # 実連鎖でない (疑似イベント) は対象外
        expected = sim_result.final_board._grid
        first_stable_idx = _find_first_stable_after(records, idx)
        if first_stable_idx is None:
            continue
        actual = records[first_stable_idx].grid
        mismatch_cells = int((actual != expected).sum())
        ghost_sec, resolved = (0.0, True) if mismatch_cells == 0 else \
            _find_ghost_resolution(records, first_stable_idx, expected)
        first_stable_lag = records[first_stable_idx].t_sec - rec.chain_trigger_sec
        results.append({
            "t_trigger": rec.chain_trigger_sec, "chain_count": sim_result.chain_count,
            "mismatch_cells": mismatch_cells,
            "mismatch_rate": mismatch_cells / TOTAL_CELLS,
            "ghost_duration_sec": ghost_sec, "resolved": resolved,
            "first_stable_lag_sec": first_stable_lag,
            # lag_flag=True: 別の後続手番を挟んだ後の STABLE を比較対象にした
            # 疑いがあり、mismatch を額面通り受け取れない可能性がある
            # (GHOST_FIRST_STABLE_LAG_WARN_SEC 参照)。
            "lag_flag": first_stable_lag > GHOST_FIRST_STABLE_LAG_WARN_SEC,
        })
    return results


# ============================
# 2. 浮きぷよ (重力整合)
# ============================


def _measure_floating_puyo(records: list[_FrameRecord]) -> dict:
    """distinct STABLE snapshot ごとに check_gravity_rule を適用し違反率を返す。"""
    last_bytes: bytes | None = None
    n_snapshots = 0
    n_with_violation = 0
    total_violation_cells = 0
    for rec in records:
        if rec.state != BoardState.STABLE.name or rec.grid is None:
            continue
        gb = rec.grid.tobytes()
        if gb == last_bytes:
            continue
        last_bytes = gb
        n_snapshots += 1
        board = Board.from_list(rec.grid.tolist())
        is_valid, violations = check_gravity_rule(board)
        if not is_valid:
            n_with_violation += 1
            total_violation_cells += len(violations)
    rate = n_with_violation / n_snapshots if n_snapshots > 0 else 0.0
    return {
        "n_snapshots": n_snapshots, "n_with_violation": n_with_violation,
        "violation_rate": rate, "total_violation_cells": total_violation_cells,
    }


# ============================
# 3. 色ワープ/変色 (既存 monitor 流用)
# ============================


def _measure_color_warp(records: list[_FrameRecord]) -> dict:
    """erasure_alerts / transition_drop_alerts を集計し分間レートを返す。"""
    n_erasure = sum(r.n_erasure_alerts for r in records)
    n_drop = sum(r.n_transition_drop_alerts for r in records)
    duration_min = max(1e-6, (records[-1].t_sec - records[0].t_sec) / 60.0) if records else 1e-6
    return {
        "n_erasure_alerts": n_erasure, "n_transition_drop_alerts": n_drop,
        "erasure_alerts_per_min": n_erasure / duration_min,
        "transition_drop_alerts_per_min": n_drop / duration_min,
    }


# ============================
# 4. お邪魔会計整合 (期待送り量 vs 実測着弾量)
# ============================


def _measure_ojama_accounting(
    records: list[_FrameRecord], opp_records: list[_FrameRecord],
) -> list[dict]:
    """発火側 chain_event.ojama_sent (期待) と相手盤面の実測着弾量の乖離を計測する。

    grid が None (未確定) のフレームは _find_landing の gap 検知が正しく
    働くよう除外してから配列化する (欠損を 0 埋めすると偽の着弾検知を招くため)。
    """
    valid_opp = [r for r in opp_records if r.grid is not None]
    if not valid_opp:
        return []
    opp_t = np.array([r.t_sec for r in valid_opp], dtype=np.float64)
    opp_counts = _visible_ojama_counts(np.stack([r.grid for r in valid_opp]))
    results: list[dict] = []
    for idx in _new_chain_triggers(records):
        rec = records[idx]
        if rec.chain_ojama_sent is None or rec.chain_ojama_sent <= 0:
            continue
        landing = _find_landing(
            opp_t, opp_counts, rec.chain_trigger_sec,
            OPP_GAP_THRESHOLD_SEC, MAX_LANDING_SEARCH_SEC,
        )
        if landing.status != "landed":
            continue
        landed_idx = landing.landed_idx
        base_idx = int(np.searchsorted(opp_t, rec.chain_trigger_sec, side="right")) - 1
        actual_delta = int(opp_counts[landed_idx]) - int(opp_counts[max(0, base_idx)])
        results.append({
            "t_trigger": rec.chain_trigger_sec, "expected_ojama_sent": rec.chain_ojama_sent,
            "actual_landed": actual_delta,
            "abs_diff": abs(rec.chain_ojama_sent - actual_delta),
        })
    return results


# ============================
# 5. is_match_active 誤 False (CHAIN 中の confirmed_board 消失率)
# ============================


def _measure_match_active_false_negative(records: list[_FrameRecord]) -> dict:
    """CHAIN 状態フレームのうち confirmed_board が None の割合を返す。"""
    chain_frames = [r for r in records if r.state == BoardState.CHAIN.name]
    if not chain_frames:
        return {"n_chain_frames": 0, "false_negative_rate": 0.0}
    n_none = sum(1 for r in chain_frames if r.grid is None)
    return {
        "n_chain_frames": len(chain_frames),
        "n_none": n_none,
        "false_negative_rate": n_none / len(chain_frames),
    }


# ============================
# 5b. 反復4 (2026-07-23): CHAIN/GRAVITY_SETTLE 中 grid=None の理由内訳
# ============================
#
# 診断計装 (修正ではない): confirmed_board=None が
#   - "menu_reset"      : is_match_active→MENU 経路 (P3 が狙った経路)
#   - "chain_hold_none" : それ以外 (CHAIN/GRAVITY_SETTLE 中に別要因で None)
#   - "cold_start"       : この試合でまだ一度も STABLE 確定していない
#   - "other"            : 上記以外 (fail-silent 防止の受け皿)
# のどれに起因するかを内訳集計する。真因が menu_reset 主体なら P3 (反復3) が
# 効くはず、chain_hold_none 主体なら P3 は無力で別経路の調査が必要。


def _measure_board_none_reason_breakdown(
    records: list[_FrameRecord], states: tuple[str, ...],
) -> dict:
    """指定 state 群のフレームのうち grid=None の理由内訳を集計する。"""
    target_frames = [r for r in records if r.state in states]
    none_frames = [r for r in target_frames if r.grid is None]
    counts: dict[str, int] = {}
    for r in none_frames:
        key = r.board_none_reason or "unclassified"
        counts[key] = counts.get(key, 0) + 1
    n_total = len(target_frames)
    n_none = len(none_frames)
    return {
        "n_frames": n_total,
        "n_none": n_none,
        "none_rate": (n_none / n_total) if n_total > 0 else 0.0,
        "reason_counts": counts,
        "reason_rates": (
            {k: v / n_none for k, v in counts.items()} if n_none > 0 else {}
        ),
    }


# ============================
# 1 動画分の集計
# ============================


def _review_one_video(
    video_stem: str, start_sec: float, max_sec: float, sim: ChainSimulator,
    enable_chain_exit_warmup: bool = False,
    enable_chain_exit_next_signal: bool = False,
) -> dict:
    """1 動画・1 窓分の全メトリクスを計測する。"""
    print(f"  {video_stem}: start={start_sec}s max={max_sec}s を処理中...")
    by_side = _capture_frames(
        video_stem, start_sec, max_sec,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
    )
    out: dict = {"video_stem": video_stem, "start_sec": start_sec, "max_sec": max_sec, "sides": {}}
    for side, opp_side in (("1P", "2P"), ("2P", "1P")):
        records = by_side[side]
        if not records:
            continue
        ghost = _measure_ghost_mismatch(records, sim)
        floating = _measure_floating_puyo(records)
        warp = _measure_color_warp(records)
        acct = _measure_ojama_accounting(records, by_side[opp_side])
        active_fn = _measure_match_active_false_negative(records)
        board_none_chain = _measure_board_none_reason_breakdown(
            records, (BoardState.CHAIN.name,),
        )
        board_none_settle = _measure_board_none_reason_breakdown(
            records, (BoardState.GRAVITY_SETTLE.name,),
        )
        out["sides"][side] = {
            "ghost_mismatch_events": ghost, "floating_puyo": floating,
            "color_warp": warp, "ojama_accounting": acct,
            "match_active_false_negative": active_fn,
            "board_none_reason_chain": board_none_chain,
            "board_none_reason_gravity_settle": board_none_settle,
        }
    print(f"    -> {sum(len(v) for v in by_side.values())} frame*side 記録")
    return out


# ============================
# サマリ集計・出力
# ============================


def _summarize(video_reports: list[dict]) -> dict:
    """全動画の主要スコアを 1 つの dict に集約する (低いほど良いスカラー中心)。"""
    ghost_rates: list[float] = []
    ghost_durations: list[float] = []
    ghost_unresolved: list[bool] = []
    ghost_lag_flags: list[bool] = []
    floating_rates: list[float] = []
    erasure_rates: list[float] = []
    acct_diffs: list[float] = []
    active_fn_rates: list[float] = []
    for rep in video_reports:
        for side_data in rep["sides"].values():
            for ev in side_data["ghost_mismatch_events"]:
                ghost_rates.append(ev["mismatch_rate"])
                ghost_durations.append(ev["ghost_duration_sec"])
                ghost_unresolved.append(not ev["resolved"])
                ghost_lag_flags.append(ev["lag_flag"])
            floating_rates.append(side_data["floating_puyo"]["violation_rate"])
            erasure_rates.append(side_data["color_warp"]["erasure_alerts_per_min"])
            for a in side_data["ojama_accounting"]:
                acct_diffs.append(a["abs_diff"])
            active_fn_rates.append(side_data["match_active_false_negative"]["false_negative_rate"])
    return {
        "n_chain_events": len(ghost_rates),
        "ghost_mismatch_rate_mean": _safe_mean(ghost_rates),
        "ghost_duration_sec_mean": _safe_mean(ghost_durations),
        "ghost_unresolved_rate": _safe_mean([float(x) for x in ghost_unresolved]),
        # データ品質フラグ: 別の後続手番を挟んだ後の STABLE を誤って比較対象に
        # した疑いがある連鎖イベントの割合 (高いと上記 mismatch 系の信頼度が下がる)。
        "ghost_lag_flag_rate": _safe_mean([float(x) for x in ghost_lag_flags]),
        "floating_puyo_violation_rate_mean": _safe_mean(floating_rates),
        "color_warp_alerts_per_min_mean": _safe_mean(erasure_rates),
        "ojama_accounting_abs_diff_mean": _safe_mean(acct_diffs),
        "match_active_false_negative_rate_mean": _safe_mean(active_fn_rates),
        # 反復4: CHAIN/GRAVITY_SETTLE 中 grid=None の理由内訳 (全動画・全 side 集計)。
        "board_none_reason_chain": _aggregate_reason_breakdown(
            video_reports, "board_none_reason_chain",
        ),
        "board_none_reason_gravity_settle": _aggregate_reason_breakdown(
            video_reports, "board_none_reason_gravity_settle",
        ),
    }


def _aggregate_reason_breakdown(video_reports: list[dict], key: str) -> dict:
    """全動画・全 side の board_none_reason 内訳 (件数) を合算する。

    反復4: is_match_active→MENU 経路 (menu_reset) が真因か、それ以外
    (chain_hold_none 等) が真因かを動画横断で切り分けるための集計。
    """
    total_frames = 0
    total_none = 0
    counts: dict[str, int] = {}
    for rep in video_reports:
        for side_data in rep["sides"].values():
            bd = side_data.get(key)
            if bd is None:
                continue
            total_frames += bd["n_frames"]
            total_none += bd["n_none"]
            for reason, n in bd["reason_counts"].items():
                counts[reason] = counts.get(reason, 0) + n
    return {
        "n_frames": total_frames,
        "n_none": total_none,
        "none_rate": (total_none / total_frames) if total_frames > 0 else 0.0,
        "reason_counts": counts,
        "reason_rates": (
            {k: v / total_none for k, v in counts.items()} if total_none > 0 else {}
        ),
    }


def _safe_mean(values: list[float]) -> float:
    """空リストなら 0.0 を返す平均 (nan 汚染回避)。"""
    return float(np.mean(values)) if values else 0.0


def _print_summary_table(summary: dict) -> None:
    """低いほど良いスカラー指標を表形式で出力する。"""
    print("\n[サマリ] (全て低いほど良い)")
    print(f"  対象連鎖イベント数              : {summary['n_chain_events']}")
    print(f"  1. 残像/連鎖後 不一致率 (mean)   : {summary['ghost_mismatch_rate_mean']:.4f}")
    print(f"     残像持続時間 秒 (mean)        : {summary['ghost_duration_sec_mean']:.2f}")
    print(f"     未解消率 (>{GHOST_SEARCH_MAX_SEC:.0f}秒経過)      : {summary['ghost_unresolved_rate']:.4f}")
    print(f"     [品質注意] lag_flag率(信頼度低下): {summary['ghost_lag_flag_rate']:.4f}")
    print(f"  2. 浮きぷよ違反率 (mean)          : {summary['floating_puyo_violation_rate_mean']:.4f}")
    print(f"  3. 色ワープ/変色 件/分 (mean)     : {summary['color_warp_alerts_per_min_mean']:.4f}")
    print(f"  4. お邪魔会計 絶対差 個 (mean)    : {summary['ojama_accounting_abs_diff_mean']:.2f}")
    print(f"  5. is_match_active誤False率(mean): {summary['match_active_false_negative_rate_mean']:.4f}")
    print("\n[反復4 診断] CHAIN/GRAVITY_SETTLE 中 grid=None の理由内訳")
    _print_reason_breakdown("CHAIN", summary["board_none_reason_chain"])
    _print_reason_breakdown("GRAVITY_SETTLE", summary["board_none_reason_gravity_settle"])


def _print_reason_breakdown(label: str, bd: dict) -> None:
    """board_none_reason 内訳を 1 state 分表示する。"""
    print(
        f"  [{label}] n_frames={bd['n_frames']} n_none={bd['n_none']} "
        f"none_rate={bd['none_rate']:.4f}",
    )
    for reason, rate in sorted(
        bd["reason_rates"].items(), key=lambda kv: -kv[1],
    ):
        count = bd["reason_counts"][reason]
        print(f"      - {reason:16s}: {count:6d} 件 ({rate:.4f})")


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする (P1 実験用 optional flag、既定 False = 従来通り)。"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--enable-chain-exit-warmup", action="store_true",
        dest="enable_chain_exit_warmup",
        help="機能C: CHAIN→STABLE遷移直後の confirmed 凍結 (src既存 runtime flag "
             "を透過、既定False)。",
    )
    ap.add_argument(
        "--enable-chain-exit-next-signal", action="store_true",
        dest="enable_chain_exit_next_signal",
        help="案X: NextSlide signal によるCHAIN即終了 (0.5秒版、既定False)。",
    )
    ap.add_argument(
        "--label", default="",
        help="出力ファイル名に付与する識別ラベル (例: warmup_on)。",
    )
    return ap.parse_args()


def main() -> None:
    """メイン処理: 対象動画を処理し JSON 保存 + サマリ出力する。"""
    args = _parse_args()
    print(f"[INFO] 対象 {len(TARGET_WINDOWS)} 動画・窓 (物理整合性レビュー) "
          f"warmup={args.enable_chain_exit_warmup} "
          f"next_signal={args.enable_chain_exit_next_signal}")
    sim = ChainSimulator()
    video_reports: list[dict] = []
    for stem, start_sec, max_sec in TARGET_WINDOWS:
        video_reports.append(_review_one_video(
            stem, start_sec, max_sec, sim,
            enable_chain_exit_warmup=args.enable_chain_exit_warmup,
            enable_chain_exit_next_signal=args.enable_chain_exit_next_signal,
        ))

    summary = _summarize(video_reports)
    summary["condition"] = {
        "enable_chain_exit_warmup": args.enable_chain_exit_warmup,
        "enable_chain_exit_next_signal": args.enable_chain_exit_next_signal,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.label}" if args.label else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{timestamp}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": video_reports}, f,
                   ensure_ascii=False, indent=2, default=str)
    print(f"\n[DONE] {out_path} に保存しました")
    _print_summary_table(summary)


if __name__ == "__main__":
    main()
