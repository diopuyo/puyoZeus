"""軽量 board 抽出パス — SiameseBoardCNN 学習用 npz を高速収集する。

collect_indicators_v2 の重い処理(全指標計算・お邪魔会計・ojama_disruption 等)を
省略し、confirmed_board グリッドと勝敗 won ラベルのみを蓄積する。

## 省略する処理
- 全指標計算 (indicators_v2 モジュール呼び出しなし)
- お邪魔会計 (OjamaAccountingTracker 不使用)
- ojama_disruption (モンテカルロ計算なし)
- NextDetector (load_next_detector=False)
- VideoChainTracker (enable_chain_tracker=False)

## 出力 npz 形式
collect_indicators_v2 --board-npz と同形式 + won 列を追加:
  grids      : (N, 13, 6) int8
  video_id   : (N,) str
  side       : (N,) str  "1P" / "2P"
  t_sec      : (N,) float32
  game_idx   : (N,) int32
  frame_idx  : (N,) int32
  won        : (N,) float32  1P視点の勝敗 (1.0/0.0/NaN)

## 勝敗 won の自己ラベル付け
score のリセット(前値 - 現値 >= SCORE_RESET_THRESHOLD)でゲーム境界を検知し
game_idx を振る。動画末尾で最終 score が大きい side を勝者とし、
そのゲームの各 snapshot に 1P 視点 won を付与する。
(1P 盤面なら 1P 勝ち=1、1P 負け=0 / 2P 盤面は逆転)

## 使い方
    python -m scripts.collect_boards_lean \\
        --video data/frames/video_29.mp4 \\
        --out-npz /tmp/lean29.npz \\
        --max-sec 30
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================

# 出力解像度 (認識は 1920x1080 前提)
TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0

# 試合境界検知: score がこの値以上減少したら新しい試合とみなす
SCORE_RESET_THRESHOLD: int = 500

# 勝敗ラベルが付与できない試合の won 値
WON_UNKNOWN: float = float("nan")


# ============================
# 蓄積バッファ
# ============================

@dataclass
class _LeanNpzAccumulator:
    """board グリッド + won ラベル蓄積バッファ。

    confirmed_board と score 情報を蓄積し、動画末尾で won を付与して npz 保存する。
    """
    grids: list[np.ndarray] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)
    t_secs: list[float] = field(default_factory=list)
    game_idxs: list[int] = field(default_factory=list)
    frame_idxs: list[int] = field(default_factory=list)
    # won は後付け (動画末尾で付与する)
    wons: list[float] = field(default_factory=list)

    def append(
        self,
        grid: np.ndarray,
        video_id: str,
        side: str,
        t_sec: float,
        game_idx: int,
        frame_idx: int,
    ) -> None:
        """1 STABLE snapshot を追加する。won は NaN で仮置き。"""
        self.grids.append(grid.copy())
        self.video_ids.append(video_id)
        self.sides.append(side)
        self.t_secs.append(t_sec)
        self.game_idxs.append(game_idx)
        self.frame_idxs.append(frame_idx)
        self.wons.append(WON_UNKNOWN)

    def assign_won_labels(
        self,
        game_final_scores: dict[int, dict[str, int | None]],
    ) -> None:
        """各 game_idx の最終 score から 1P 視点 won を付与する。

        Args:
            game_final_scores: {game_idx: {"1P": score_int|None, "2P": score_int|None}}
        """
        # game_idx ごとの勝者 side を判定
        winner_by_game: dict[int, str | None] = {}
        for gidx, scores in game_final_scores.items():
            s1 = scores.get("1P")
            s2 = scores.get("2P")
            if s1 is not None and s2 is not None and s1 != s2:
                winner_by_game[gidx] = "1P" if s1 > s2 else "2P"
            else:
                winner_by_game[gidx] = None  # 判定不能

        for i in range(len(self.wons)):
            gidx = self.game_idxs[i]
            winner = winner_by_game.get(gidx)
            if winner is None:
                continue
            # 1P 視点: 自 side が勝者なら 1、負けなら 0
            self.wons[i] = 1.0 if self.sides[i] == winner else 0.0

    def save(self, path: Path) -> None:
        """npz 形式で保存する。grids=(N,13,6) int8、won=(N,) float32。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            grids=np.array(self.grids, dtype=np.int8) if self.grids
                  else np.array([], dtype=np.int8),
            video_id=np.array(self.video_ids),
            side=np.array(self.sides),
            t_sec=np.array(self.t_secs, dtype=np.float32),
            game_idx=np.array(self.game_idxs, dtype=np.int32),
            frame_idx=np.array(self.frame_idxs, dtype=np.int32),
            won=np.array(self.wons, dtype=np.float32),
        )


# ============================
# 1 side の状態管理
# ============================

@dataclass
class _SideState:
    """1 side の間引き・ゲーム境界管理用状態。"""
    game_idx: int = 0
    prev_score: int | None = None
    last_emitted_grid: bytes | None = None
    # game_idx ごとの最終 score を追跡
    final_scores: dict[int, int | None] = field(default_factory=dict)


def _update_game_boundary(state: _SideState, score: int | None) -> None:
    """score リセット検知で game_idx を進める。"""
    if score is not None:
        state.final_scores[state.game_idx] = score
    if score is not None and state.prev_score is not None:
        if state.prev_score - score >= SCORE_RESET_THRESHOLD:
            state.game_idx += 1
    if score is not None:
        state.prev_score = score


def _should_emit(state: _SideState, board: Board, bstate: BoardState) -> bool:
    """STABLE かつ重複でない盤面かを判定する。"""
    if bstate != BoardState.STABLE or board is None:
        return False
    # 全消し直後 / 試合開始直後 (盤面ぷよ 0) は除外
    if board.count_puyos() == 0:
        return False
    # 直前と同一盤面なら間引き
    grid_bytes = board._grid.tobytes()
    if grid_bytes == state.last_emitted_grid:
        return False
    return True


# ============================
# メイン収集ループ
# ============================

def collect_lean(
    video_path: Path,
    out_npz: Path,
    max_sec: float = 0.0,
    start_sec: float = 0.0,
) -> int:
    """1 動画を処理して盤面 npz を出力する。指標計算は一切行わない。

    Args:
        video_path: 入力動画パス。
        out_npz: 出力 npz パス。
        max_sec: 処理最大秒数 (0=全長)。
        start_sec: 処理開始オフセット秒。

    Returns:
        蓄積した snapshot 数。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[lean] cannot open: {video_path}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps) if start_sec > 0.0 else 0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    if max_sec > 0:
        end_frame = min(total_frames, start_frame + int(max_sec * fps))
    else:
        end_frame = total_frames
    n_frames = max(0, end_frame - start_frame)

    video_id = video_path.stem
    # NextDetector / ChainTracker を OFF にして高速化
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=False,
        temporal_smoothing=1,
        load_next_detector=False,
        force_in_match=True,
    )
    # 動画 ID をセット (per-video HSV プロファイル自動ロード用)
    vid_match = __import__("re").search(r"(v\d+|video_\d+)", video_path.name)
    if vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(vid_match.group(1))

    acc = _LeanNpzAccumulator()
    state_p1 = _SideState()
    state_p2 = _SideState()

    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)

        _process_side_lean(
            acc, state_p1, "1P", result.p1.confirmed_board,
            result.p1.state, result.p1.score, video_id, t_sec, fi,
        )
        _process_side_lean(
            acc, state_p2, "2P", result.p2.confirmed_board,
            result.p2.state, result.p2.score, video_id, t_sec, fi,
        )
    cap.release()

    # 勝敗ラベルを付与して保存
    combined_final = _merge_final_scores(state_p1, state_p2)
    acc.assign_won_labels(combined_final)
    acc.save(out_npz)
    return len(acc.grids)


def _process_side_lean(
    acc: _LeanNpzAccumulator,
    state: _SideState,
    side_label: str,
    board: Optional[Board],
    bstate: BoardState,
    score: int | None,
    video_id: str,
    t_sec: float,
    frame_idx: int,
) -> None:
    """1 side の STABLE snapshot を蓄積する。指標計算は行わない。"""
    _update_game_boundary(state, score)
    if board is None or not _should_emit(state, board, bstate):
        return
    acc.append(
        board._grid, video_id, side_label,
        round(t_sec, 3), state.game_idx, frame_idx,
    )
    state.last_emitted_grid = board._grid.tobytes()


def _merge_final_scores(
    state_p1: _SideState,
    state_p2: _SideState,
) -> dict[int, dict[str, int | None]]:
    """両 side の final_scores を game_idx をキーに統合する。

    Returns:
        {game_idx: {"1P": score_or_None, "2P": score_or_None}}
    """
    all_games: set[int] = set(state_p1.final_scores) | set(state_p2.final_scores)
    result: dict[int, dict[str, int | None]] = {}
    for gidx in all_games:
        result[gidx] = {
            "1P": state_p1.final_scores.get(gidx),
            "2P": state_p2.final_scores.get(gidx),
        }
    return result


# ============================
# CLI エントリポイント
# ============================

def main() -> int:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description="軽量 board 抽出 (SiameseBoardCNN 学習用)")
    parser.add_argument("--video", type=Path, required=True, help="入力動画パス")
    parser.add_argument("--out-npz", type=Path, required=True, help="出力 npz パス")
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="処理する最大秒数 (0=全長)",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="処理開始オフセット秒",
    )
    args = parser.parse_args()
    n = collect_lean(
        args.video, args.out_npz,
        max_sec=args.max_sec,
        start_sec=args.start_sec,
    )
    print(f"[lean] {args.video.name} -> {args.out_npz} : {n} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
