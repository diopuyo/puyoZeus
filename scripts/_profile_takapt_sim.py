"""takapt 定石 30-sim のプロファイル計測スクリプト。

STABLE スナップショットを video_124_4min.mp4 から収集し
_takapt_best_drop の実行時間 (キャッシュ有/無) を計測する。

使い方:
    python -m scripts._profile_takapt_sim --video data/frames/video_124_4min.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.indicators_v2 import _takapt_best_drop, IGNITION_TRIAL_COLORS  # noqa: E402
from src.board import BOARD_COLS  # noqa: E402

TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0

# プロファイル上限 (STABLE スナップショット数)
MAX_SNAPSHOTS: int = 80
# 処理動画の最大秒数
MAX_SEC: float = 120.0


def _collect_stable_boards(video_path: Path, max_sec: float) -> list[Board]:
    """STABLE 盤面スナップショットを収集する。"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(max_sec * fps))

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=False,
        enable_chain_tracker=False,
        temporal_smoothing=1,
        load_next_detector=False,
        force_in_match=True,
    )

    boards: list[Board] = []
    prev_grid_p1: bytes | None = None
    prev_grid_p2: bytes | None = None

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        result = pipeline.update(fi, fi / fps, frame)

        for side, prev_grid in [
            (result.p1, prev_grid_p1),
            (result.p2, prev_grid_p2),
        ]:
            if side.state != BoardState.STABLE:
                continue
            b = side.confirmed_board
            if b is None or b.count_puyos() == 0:
                continue
            g = b._grid.tobytes()
            if g == prev_grid:
                continue
            boards.append(b)
            if side is result.p1:
                prev_grid_p1 = g
            else:
                prev_grid_p2 = g
        if len(boards) >= MAX_SNAPSHOTS:
            break

    cap.release()
    return boards


def _profile_takapt(boards: list[Board]) -> None:
    """_takapt_best_drop のキャッシュ有/無のタイムを計測して表示する。"""
    n = len(boards)
    print(f"[profile] STABLE スナップショット数: {n}")

    # --- キャッシュ有 (初回: キャッシュ cold) ---
    sim_cold = ChainSimulator(cache_enabled=True)
    t0 = time.perf_counter()
    results_cold = [_takapt_best_drop(b, sim_cold) for b in boards]
    t1 = time.perf_counter()
    elapsed_cold = (t1 - t0) * 1000.0  # ms

    # --- キャッシュ有 (2回目: キャッシュ warm) ---
    t2 = time.perf_counter()
    results_warm = [_takapt_best_drop(b, sim_cold) for b in boards]
    t3 = time.perf_counter()
    elapsed_warm = (t3 - t2) * 1000.0  # ms

    # --- キャッシュ無 ---
    sim_nocache = ChainSimulator(cache_enabled=False)
    t4 = time.perf_counter()
    results_nocache = [_takapt_best_drop(b, sim_nocache) for b in boards]
    t5 = time.perf_counter()
    elapsed_nocache = (t5 - t4) * 1000.0  # ms

    # --- 結果集計 ---
    chains_cold = [r[0] for r in results_cold]
    avg_chain = sum(chains_cold) / max(1, len(chains_cold))
    max_chain = max(chains_cold) if chains_cold else 0
    nonzero = sum(1 for c in chains_cold if c > 0)

    print(f"\n=== タイム計測 (n={n} スナップショット) ===")
    print(f"  キャッシュ cold (初回)  : {elapsed_cold:.1f} ms 合計 "
          f"= {elapsed_cold / max(1, n):.2f} ms/snapshot")
    print(f"  キャッシュ warm (2回目) : {elapsed_warm:.1f} ms 合計 "
          f"= {elapsed_warm / max(1, n):.3f} ms/snapshot")
    print(f"  キャッシュ無し          : {elapsed_nocache:.1f} ms 合計 "
          f"= {elapsed_nocache / max(1, n):.2f} ms/snapshot")
    print(f"\n=== 連鎖数統計 (cold) ===")
    print(f"  平均: {avg_chain:.2f} 連鎖, 最大: {max_chain} 連鎖")
    print(f"  非ゼロ率: {nonzero}/{n} ({100.0 * nonzero / max(1, n):.1f}%)")

    # 具体値サンプル (最初の 5 件)
    print(f"\n=== 最初の 5 スナップショット ===")
    for i, (b, (chain, best_b)) in enumerate(zip(boards[:5], results_cold[:5])):
        puyo_count = b.count_puyos()
        print(f"  [{i}] puyos={puyo_count}, max_chain={chain}, "
              f"best_board={'あり' if best_b else 'なし'}")

    # STABLE 0.5s 間隔での許容判定
    threshold_ms = 50.0
    per_snap_cold = elapsed_cold / max(1, n)
    verdict = "OK (< 50ms)" if per_snap_cold < threshold_ms else "NG (>= 50ms → 5色→3色削減を検討)"
    print(f"\n=== 判定 ===")
    print(f"  cold {per_snap_cold:.2f} ms/snapshot: {verdict}")
    if per_snap_cold >= threshold_ms:
        # 3色版 (RED/BLUE/GREEN のみ) の速度を参考計測
        from src.board import COLOR_RED, COLOR_BLUE, COLOR_GREEN
        three_colors = (COLOR_RED, COLOR_BLUE, COLOR_GREEN)
        sim_3 = ChainSimulator(cache_enabled=True)
        t6 = time.perf_counter()
        for b in boards:
            best_chain_3 = 0
            best_board_3 = None
            for col in range(BOARD_COLS):
                for color in three_colors:
                    from src.indicators_v2 import _drop_one_color
                    dropped = _drop_one_color(b, col, color)
                    if dropped is None:
                        continue
                    res = sim_3.simulate(dropped)
                    if res.chain_count > best_chain_3:
                        best_chain_3 = res.chain_count
                        best_board_3 = dropped
        t7 = time.perf_counter()
        elapsed_3 = (t7 - t6) * 1000.0
        print(f"  3色版 cold: {elapsed_3 / max(1, n):.2f} ms/snapshot "
              f"(5色比 {elapsed_3/max(1e-6, elapsed_cold):.1%})")


def main() -> int:
    parser = argparse.ArgumentParser(description="takapt 30-sim プロファイル")
    parser.add_argument(
        "--video", type=Path,
        default=Path("data/frames/video_124_4min.mp4"),
        help="入力動画パス",
    )
    args = parser.parse_args()
    print(f"[profile] 動画: {args.video}")
    boards = _collect_stable_boards(args.video, MAX_SEC)
    if not boards:
        print("[ERROR] STABLE スナップショットを収集できませんでした")
        return 1
    _profile_takapt(boards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
