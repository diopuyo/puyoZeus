"""打ち合い計測器 Stage 0: 相手盤面「安い道」の精度検証。

アーキの発見(前提、2026-07-22): 相手盤面は動作中(NON-STABLE)も「見れない」
わけではない。(1) confirmed_board は動作中も毎フレーム凍結保持される
(src/board_state_machine.py)。(2) 連鎖は着地の瞬間に物理シミュで即解決され、
着地フレームで既に連鎖後の正しい盤面になっている可能性がある
(src/placement_inferrer.py resolve_after_placement、
src/recognition_pipeline.py の CHAIN→STABLE 復帰時 chain_event 解決)。
真の欠損源は収集層 (scripts/collect_boards_lean.py:311-322 の _should_emit)
が STABLE 以外を捨てているだけ、という仮説。

本スクリプトはこの仮説を「NON-STABLE中に既に得られている confirmed_board」と
「その後実際に STABLE 復帰した時の confirmed_board (ground truth)」を
cell 単位で比較することで検証する。

⚠️ 読み取り専用: RecognitionPipeline / BoardStateMachine は一切変更しない
(scripts/collect_indicators_v2.py の外部消費パターンに倣い、PipelineResult
を消費するだけ)。src/ は無改修。

使い方:
    PYTHONPATH=. python -m scripts.measure_opponent_estimate_accuracy
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board, BOARD_COLS, BOARD_ROWS  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================

# 認識は 1920x1080 前提 (既存収集スクリプトと同条件)
TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0

# 盤面セル総数 (13行×6列)
TOTAL_CELLS: int = BOARD_ROWS * BOARD_COLS

# 検証対象動画・処理窓 (stem, start_sec, max_sec)。
# 既知の大連鎖(scripts/measure_exchange_dynamics.py の回帰確認で確認済み)を
# 含む区間を選び、TSUMO_FALL/CHAIN/OJAMA_FALL の状態を混在させる。
# ティア混在: c62=マスター(11連鎖), c82=S級(12連鎖), c5=チャレンジャー(14連鎖)。
TARGET_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("c62", 895.0, 60.0),
    ("c82", 960.0, 55.0),
    ("c5", 805.0, 55.0),
)

VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUTPUT_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "opponent_estimate_accuracy.csv"


@dataclass
class _FrameRecord:
    """1 (video, side, frame) 分の観測値。"""
    video_stem: str
    side: str
    frame_idx: int
    t_sec: float
    state: str
    grid: np.ndarray | None  # (13,6) int、None は MENU 等で confirmed 無し
    chain_before_grid: np.ndarray | None  # chain_event.before_board (無ければ None)


def _capture_frames(video_stem: str, start_sec: float, max_sec: float) -> list[_FrameRecord]:
    """1 動画・1 窓分の全 side・全 frame を RecognitionPipeline で処理し記録する。"""
    video_path = VIDEO_DIR / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}", file=sys.stderr)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)

    # collect_indicators_v2.py と同条件 (chain_tracker/next_detector 有効)
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(video_stem)

    records: list[_FrameRecord] = []
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        for side, side_result in (("1P", result.p1), ("2P", result.p2)):
            grid = side_result.confirmed_board._grid.copy() if side_result.confirmed_board else None
            chain_grid = (
                side_result.chain_event.before_board._grid.copy()
                if side_result.chain_event is not None else None
            )
            records.append(_FrameRecord(
                video_stem=video_stem, side=side, frame_idx=fi, t_sec=t_sec,
                state=side_result.state.name, grid=grid, chain_before_grid=chain_grid,
            ))
    cap.release()
    return records


# ============================
# NON-STABLE ランの抽出 + 一致率計算
# ============================


def _find_non_stable_runs(
    records: list[_FrameRecord],
) -> list[tuple[int, int, int]]:
    """時系列(同一 side)から (run_start_idx, run_end_idx, post_stable_idx) を抽出する。

    STABLE で挟まれた連続 NON-STABLE 区間のみを対象にする (窓の先頭/末尾で
    STABLE が確認できない不完全な区間は ground truth 不明のため除外)。
    """
    runs: list[tuple[int, int, int]] = []
    i = 0
    n = len(records)
    while i < n:
        if records[i].state != BoardState.STABLE.name:
            i += 1
            continue
        j = i + 1
        while j < n and records[j].state != BoardState.STABLE.name:
            j += 1
        if j < n and j > i + 1:
            runs.append((i + 1, j - 1, j))
        i = j
    return runs


def _cell_match_rate(a: np.ndarray | None, b: np.ndarray) -> float | None:
    """2 グリッドの cell 一致率 (0〜1)。a が None (confirmed 無し) なら None。"""
    if a is None or a.shape != b.shape:
        return None
    return float((a == b).sum()) / float(TOTAL_CELLS)


def _measure_side_runs(
    records: list[_FrameRecord], video_stem: str, side: str,
) -> list[dict]:
    """1 (video, side) 分の NON-STABLE ラン全てを計測し、行データ一覧を返す。"""
    rows: list[dict] = []
    for start_i, end_i, post_i in _find_non_stable_runs(records):
        post_board = records[post_i].grid
        pre_board = records[start_i - 1].grid
        if post_board is None or pre_board is None:
            continue
        run_len = end_i - start_i + 1
        for k, idx in enumerate(range(start_i, end_i + 1)):
            rec = records[idx]
            rows.append({
                "video_stem": video_stem, "side": side, "frame_idx": rec.frame_idx,
                "state": rec.state, "position_in_run": k, "run_len": run_len,
                "sec_before_stable": records[post_i].t_sec - rec.t_sec,
                "match_rate_live": _cell_match_rate(rec.grid, post_board),
                "match_rate_naive_freeze": _cell_match_rate(pre_board, post_board),
                "match_rate_chain_sim": None,
            })
    return rows


# ============================
# 連鎖ケース専用: chain_event.before_board を独立simulateして比較
# ============================


def _measure_chain_event_cases(
    records: list[_FrameRecord], video_stem: str, side: str, sim: ChainSimulator,
) -> list[dict]:
    """CHAIN 状態で chain_event が現れた各フレームについて、独立simulateした
    final_board が「その後の実 STABLE」と一致するかを計測する。
    """
    rows: list[dict] = []
    runs = _find_non_stable_runs(records)
    for start_i, end_i, post_i in runs:
        post_board = records[post_i].grid
        if post_board is None:
            continue
        for idx in range(start_i, end_i + 1):
            rec = records[idx]
            if rec.state != BoardState.CHAIN.name or rec.chain_before_grid is None:
                continue
            try:
                before = Board.from_list(rec.chain_before_grid.tolist())
                result = sim.simulate(before)
                final = result.final_board._grid if result.final_board is not None else None
            except Exception:
                final = None
            rows.append({
                "video_stem": video_stem, "side": side, "frame_idx": rec.frame_idx,
                "state": rec.state, "position_in_run": idx - start_i, "run_len": end_i - start_i + 1,
                "sec_before_stable": records[post_i].t_sec - rec.t_sec,
                "match_rate_live": _cell_match_rate(rec.grid, post_board),
                "match_rate_naive_freeze": None,
                "match_rate_chain_sim": _cell_match_rate(final, post_board),
            })
    return rows


# ============================
# レポート集計
# ============================


def _print_state_summary(df: pd.DataFrame) -> None:
    """state別の一致率 (live推定 vs naive frozen) を出力する。"""
    print("\n[state別 一致率] (live=NON-STABLE中に読める confirmed_board の値)")
    g = df.dropna(subset=["match_rate_live"]).groupby("state").agg(
        n=("match_rate_live", "size"),
        live_mean=("match_rate_live", "mean"),
        live_perfect_rate=("match_rate_live", lambda s: float((s >= 0.999).mean())),
        naive_mean=("match_rate_naive_freeze", "mean"),
    )
    print(g.to_string())


def _print_chain_summary(df_chain: pd.DataFrame) -> None:
    """連鎖ケース (chain_event.before_board simulate) の一致率を出力する。"""
    print("\n[連鎖ケース: chain_event.before_board を独立simulateした一致率]")
    if df_chain.empty:
        print("  (対象イベントなし)")
        return
    valid = df_chain.dropna(subset=["match_rate_chain_sim"])
    print(f"  n={len(valid)} 一致率mean={valid['match_rate_chain_sim'].mean():.4f} "
          f"完全一致率={float((valid['match_rate_chain_sim'] >= 0.999).mean()):.4f}")
    print(valid[["video_stem", "side", "frame_idx", "sec_before_stable", "match_rate_chain_sim"]]
          .head(10).to_string(index=False))


def _print_column_breakdown(records_all: dict[str, list[_FrameRecord]]) -> None:
    """列別の崩壊偏りを確認する (fail-silent警戒)。"""
    print("\n[列別ズレ確認 (fail-silent警戒)]")
    for key, records in records_all.items():
        runs = _find_non_stable_runs(records)
        mism_by_col = np.zeros(BOARD_COLS, dtype=np.int64)
        total = 0
        for start_i, end_i, post_i in runs:
            post = records[post_i].grid
            if post is None:
                continue
            for idx in range(start_i, end_i + 1):
                live = records[idx].grid
                if live is None or live.shape != post.shape:
                    continue
                mism_by_col += (live != post).sum(axis=0)
                total += 1
        if total > 0:
            print(f"  {key}: n_frame={total} 列別不一致率={np.round(mism_by_col / (total * BOARD_ROWS), 3).tolist()}")


def main() -> None:
    """メイン処理: 対象動画を処理し state別一致率レポートを出力する。"""
    print(f"[INFO] 対象 {len(TARGET_WINDOWS)} 動画・窓")
    sim = ChainSimulator()
    all_rows: list[dict] = []
    all_chain_rows: list[dict] = []
    records_all: dict[str, list[_FrameRecord]] = {}
    for stem, start_sec, max_sec in TARGET_WINDOWS:
        print(f"  {stem}: start={start_sec}s max={max_sec}s を処理中...")
        records = _capture_frames(stem, start_sec, max_sec)
        for side in ("1P", "2P"):
            side_records = [r for r in records if r.side == side]
            records_all[f"{stem}_{side}"] = side_records
            all_rows.extend(_measure_side_runs(side_records, stem, side))
            all_chain_rows.extend(_measure_chain_event_cases(side_records, stem, side, sim))
        print(f"    -> {len(records)} frame*side 記録")

    if not all_rows:
        print("[ERROR] 記録が0件でした。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] {len(df)} 行を {OUTPUT_CSV} に保存しました")

    _print_state_summary(df)
    _print_chain_summary(pd.DataFrame(all_chain_rows))
    _print_column_breakdown(records_all)


if __name__ == "__main__":
    main()
