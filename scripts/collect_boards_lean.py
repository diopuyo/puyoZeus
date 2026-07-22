"""軽量 board 抽出パス — SiameseBoardCNN 学習用 npz を高速収集する。

collect_indicators_v2 の重い処理(全指標計算・お邪魔会計・ojama_disruption 等)を
省略し、confirmed_board グリッドと勝敗 won ラベルのみを蓄積する。

## 省略する処理
- 全指標計算 (indicators_v2 モジュール呼び出しなし)
- お邪魔会計 (OjamaAccountingTracker 不使用)
- ojama_disruption (モンテカルロ計算なし)
- NextDetector (load_next_detector=False。--with-next 指定時のみ有効化)
- VideoChainTracker (enable_chain_tracker=False)

## 出力 npz 形式
collect_indicators_v2 --board-npz と同形式 + won / score 列を追加:
  grids      : (N, 13, 6) int8
  video_id   : (N,) str
  side       : (N,) str  "1P" / "2P"
  t_sec      : (N,) float32
  game_idx   : (N,) int32
  frame_idx  : (N,) int32
  won        : (N,) float32  1P視点の勝敗 (1.0/0.0/NaN)
  score      : (N,) int32    スコア (-1 = None)
  next1_a    : (N,) int8     現ネクスト軸ぷよ色 (1-5、未検出/未取得は -1)
  next1_b    : (N,) int8     現ネクスト子ぷよ色 (1-5、未検出/未取得は -1)
  dnext_a    : (N,) int8     ダブルネクスト軸ぷよ色 (1-5、未検出/未取得は -1)
  dnext_b    : (N,) int8     ダブルネクスト子ぷよ色 (1-5、未検出/未取得は -1)

  ⚠️ next1_*/dnext_* は --with-next を指定した収集時のみ実値が入る。
  未指定 (既定) の場合は NextDetector が無効なため全て -1 (後方互換、
  既存 boards_lean_fixed の再利用に影響なし)。

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

## --sample-interval による高速化
    --sample-interval 0.1 を指定すると fps*0.1 フレームに 1 回だけ
    pipeline.update を呼ぶ (collect_indicators_v2 と同じ間引き方式)。
    cap.read() は毎フレーム呼んでデコードし、間引き対象フレームは continue
    でスキップする。

    スコアリセット検知への影響:
      score は STABLE snapshot 取得時にのみ読む設計のため、
      間引きで STABLE でない短命フレームを飛ばしても実害なし。
      ゲーム終了時の score リセットは STABLE 直後の数フレームで起こるが
      0.1 秒≒3 フレーム間引きなら次の STABLE フレームで検知できる。
      (worst-case: 間引き幅 * fps フレーム = 約 0.2 秒の検知遅延)

    推奨値: 0.1〜0.2 秒 (≒3×〜6× 高速化、snapshot 数ほぼ変わらず)。
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

# next_pair/dnext_pair が None (未検出 / NextDetector 無効) の場合の埋め値。
# ぷよ色は 1-5 のため -1 は安全な sentinel。
NEXT_COLOR_UNKNOWN: int = -1


# ============================
# 蓄積バッファ
# ============================

@dataclass
class _LeanNpzAccumulator:
    """board グリッド + won ラベル蓄積バッファ。

    confirmed_board と score 情報を蓄積し、動画末尾で won を付与して npz 保存する。
    score を保存することで、収集後にオフラインで何度でも勝者ラベルを再作成できる。
    """
    grids: list[np.ndarray] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)
    t_secs: list[float] = field(default_factory=list)
    game_idxs: list[int] = field(default_factory=list)
    frame_idxs: list[int] = field(default_factory=list)
    # won は後付け (動画末尾で付与する)
    wons: list[float] = field(default_factory=list)
    # score を保存: オフライン再ラベル付けを可能にする (None は -1 として保存)
    scores: list[int] = field(default_factory=list)
    # ネクスト情報 (指標①本命版検証用、2026-07 追加)。
    # None は NEXT_COLOR_UNKNOWN (-1) として保存する。既存キー・既存呼び出しの
    # 後方互換のため append() では末尾の optional 引数として追加する。
    next1_as: list[int] = field(default_factory=list)
    next1_bs: list[int] = field(default_factory=list)
    dnext_as: list[int] = field(default_factory=list)
    dnext_bs: list[int] = field(default_factory=list)

    def append(
        self,
        grid: np.ndarray,
        video_id: str,
        side: str,
        t_sec: float,
        game_idx: int,
        frame_idx: int,
        score: int | None = None,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
    ) -> None:
        """1 STABLE snapshot を追加する。won は NaN で仮置き。

        Args:
            grid: 確定盤面グリッド (13, 6)。
            video_id: 動画 ID 文字列。
            side: "1P" または "2P"。
            t_sec: タイムスタンプ (秒)。
            game_idx: ゲーム境界カウンタ。
            frame_idx: フレーム絶対番号。
            score: スコア OCR 値。None は -1 に変換して保存。
            next_pair: (軸ぷよ色, 子ぷよ色)。None は NEXT_COLOR_UNKNOWN で保存
                (後方互換: 省略時は既存呼び出しと同じ挙動)。
            dnext_pair: ダブルネクストの (軸ぷよ色, 子ぷよ色)。同上。
        """
        self.grids.append(grid.copy())
        self.video_ids.append(video_id)
        self.sides.append(side)
        self.t_secs.append(t_sec)
        self.game_idxs.append(game_idx)
        self.frame_idxs.append(frame_idx)
        self.wons.append(WON_UNKNOWN)
        self.scores.append(score if score is not None else -1)
        n_a, n_b = next_pair if next_pair is not None else (NEXT_COLOR_UNKNOWN, NEXT_COLOR_UNKNOWN)
        d_a, d_b = dnext_pair if dnext_pair is not None else (NEXT_COLOR_UNKNOWN, NEXT_COLOR_UNKNOWN)
        self.next1_as.append(int(n_a))
        self.next1_bs.append(int(n_b))
        self.dnext_as.append(int(d_a))
        self.dnext_bs.append(int(d_b))

    def assign_won_labels(
        self,
        game_final_scores: dict[int, dict[str, int | None]],
    ) -> None:
        """各 game_idx の最終 score から 1P 視点 won を付与する。

        スコア判定を主とし、スコアが同点または欠損の場合のみ
        _winner_by_survival フォールバックで窒息判定を補助する。

        Args:
            game_final_scores: {game_idx: {"1P": score_int|None, "2P": score_int|None}}
        """
        winner_by_game: dict[int, str | None] = {}
        for gidx, scores in game_final_scores.items():
            s1 = scores.get("1P")
            s2 = scores.get("2P")
            if s1 is not None and s2 is not None and s1 != s2:
                # スコアで判定できる場合: 高得点側が勝者
                winner_by_game[gidx] = "1P" if s1 > s2 else "2P"
            else:
                # スコア同点・欠損時: 窒息フォールバック
                winner_by_game[gidx] = _winner_by_survival(self, gidx)

        for i in range(len(self.wons)):
            gidx = self.game_idxs[i]
            winner = winner_by_game.get(gidx)
            if winner is None:
                continue
            # 1P 視点: 自 side が勝者なら 1、負けなら 0
            self.wons[i] = 1.0 if self.sides[i] == winner else 0.0

    def save(self, path: Path) -> None:
        """npz 形式で保存する。grids=(N,13,6) int8、won=(N,) float32、score=(N,) int32。

        next1_a/next1_b/dnext_a/dnext_b (int8) を追加保存する (既存キーは不変、
        後方互換)。--with-next 未指定の収集では全て NEXT_COLOR_UNKNOWN (-1)。
        """
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
            score=np.array(self.scores, dtype=np.int32),
            next1_a=np.array(self.next1_as, dtype=np.int8),
            next1_b=np.array(self.next1_bs, dtype=np.int8),
            dnext_a=np.array(self.dnext_as, dtype=np.int8),
            dnext_b=np.array(self.dnext_bs, dtype=np.int8),
        )


# ============================
# 窒息フォールバック判定ヘルパ
# ============================

# 窒息判定: 3列目(index=2)の画面内最上段(row=1、隠し段row0は除く)にぷよがあれば窒息。
# 2026-07-22 ルール是正: 旧 row=0(隠し段)は窒息検知漏れ(完全オーバーフローしないと発火せず)。
# board.py DEATH_ROW と同じ定義に統一。
_DEATH_ROW: int = 1
_DEATH_COL: int = 2


def _winner_by_survival(
    acc: "_LeanNpzAccumulator",
    game_idx: int,
) -> str | None:
    """スコア判定不能時のフォールバック: 窒息していない側を勝者とする。

    各 game_idx の末尾 snapshot の grid で窒息セル (row=_DEATH_ROW=1, col=2 != 0) を確認する。
    どちらも窒息なし / 両方窒息 / snapshot なしの場合は None を返す。

    Args:
        acc: スナップショット蓄積バッファ。
        game_idx: 対象ゲームのインデックス。

    Returns:
        "1P" / "2P" / None (判定不能)
    """
    # game_idx に属するインデックスを side 別に収集
    idx_by_side: dict[str, list[int]] = {"1P": [], "2P": []}
    for i, (gidx, side) in enumerate(zip(acc.game_idxs, acc.sides)):
        if gidx == game_idx and side in idx_by_side:
            idx_by_side[side].append(i)

    def _is_suffocated(indices: list[int]) -> bool | None:
        """末尾 snapshot で窒息しているか判定する。"""
        if not indices:
            return None
        last_i = max(indices, key=lambda i: acc.t_secs[i])
        return bool(acc.grids[last_i][_DEATH_ROW, _DEATH_COL] != 0)

    suf_1p = _is_suffocated(idx_by_side["1P"])
    suf_2p = _is_suffocated(idx_by_side["2P"])

    if suf_1p is None or suf_2p is None:
        return None  # どちらかの snapshot がない
    if suf_1p and not suf_2p:
        return "2P"  # 1P が窒息 → 2P 勝ち
    if suf_2p and not suf_1p:
        return "1P"  # 2P が窒息 → 1P 勝ち
    return None  # 両方窒息・両方生存は判定不能


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
    """score リセット検知で game_idx を進める。旧ゲームの最終 score は
    リセット直前の prev_score (高値) を記録する。

    バグ修正: 旧実装はリセット発生フレームで final_scores に ≈0 の低値を
    書き込んでから game_idx を進めていたため、旧ゲームの最終スコアが
    リセット後低値で上書きされ勝者判定が全て None になっていた。
    """
    if score is None:
        return
    is_reset = (
        state.prev_score is not None
        and state.prev_score - score >= SCORE_RESET_THRESHOLD
    )
    if is_reset:
        # 旧ゲームの最終スコア = リセット直前の高値を確定
        state.final_scores[state.game_idx] = state.prev_score
        state.game_idx += 1
    # 現ゲームの暫定最終スコア (次フレームで上書きされ続け、最後は真の最終値)
    state.final_scores[state.game_idx] = score
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
    sample_interval_sec: float = 0.0,
    capture_next: bool = False,
) -> int:
    """1 動画を処理して盤面 npz を出力する。指標計算は一切行わない。

    Args:
        video_path: 入力動画パス。
        out_npz: 出力 npz パス。
        max_sec: 処理最大秒数 (0=全長)。
        start_sec: 処理開始オフセット秒。
        sample_interval_sec: フレーム間引き間隔 (秒)。0 = 全フレーム処理
            (従来挙動)。collect_indicators_v2 と同じ間引き方式を採用:
            cap.read() は毎フレーム呼び、sample_interval_frames おきに
            pipeline.update を呼ぶ。
        capture_next: True で NextDetector を有効化し next1_a/next1_b/
            dnext_a/dnext_b を実値で記録する (指標①本命版検証用)。
            既定 False = 従来挙動 (NextDetector 無効、全て -1 で保存、
            後方互換)。

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

    # --- フレーム間引き設定 (collect_indicators_v2 と同じ計算式) ---
    # sample_interval_sec=0.0 の場合は全フレーム処理 (step=1)
    sample_interval_frames: int = max(1, int(round(sample_interval_sec * fps))) \
        if sample_interval_sec > 0.0 else 1

    # NextDetector / ChainTracker を OFF にして高速化
    # (capture_next=True の場合のみ NextDetector を有効化、指標①本命版検証用)
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=False,
        temporal_smoothing=1,
        load_next_detector=capture_next,
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
        # --- フレーム間引き: sample_interval_frames おきに pipeline.update を呼ぶ ---
        # cap.read() は毎フレーム呼んでデコードし、間引き対象フレームはスキップ。
        # (collect_indicators_v2 と同じ方式)
        if local_i % sample_interval_frames != 0:
            continue
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)

        _process_side_lean(
            acc, state_p1, "1P", result.p1.confirmed_board,
            result.p1.state, result.p1.score, video_id, t_sec, fi,
            next_pair=result.p1.next_pair, dnext_pair=result.p1.dnext_pair,
        )
        _process_side_lean(
            acc, state_p2, "2P", result.p2.confirmed_board,
            result.p2.state, result.p2.score, video_id, t_sec, fi,
            next_pair=result.p2.next_pair, dnext_pair=result.p2.dnext_pair,
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
    next_pair: tuple[int, int] | None = None,
    dnext_pair: tuple[int, int] | None = None,
) -> None:
    """1 side の STABLE snapshot を蓄積する。指標計算は行わない。

    next_pair/dnext_pair は capture_next=False (既定) の呼び出しでは常に
    None (SideResult 既定値) となり、acc.append 側で -1 埋めされる
    (後方互換)。
    """
    _update_game_boundary(state, score)
    if board is None or not _should_emit(state, board, bstate):
        return
    acc.append(
        board._grid, video_id, side_label,
        round(t_sec, 3), state.game_idx, frame_idx,
        score=score, next_pair=next_pair, dnext_pair=dnext_pair,
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
    parser.add_argument(
        "--sample-interval", type=float, default=0.0,
        dest="sample_interval",
        help=(
            "フレーム間引き間隔 (秒)。0 = 全フレーム処理 (既定)。"
            "0.1 で約 3×、0.2 で約 6× 高速化。"
            "STABLE 検出・勝者判定には影響しない。"
        ),
    )
    parser.add_argument(
        "--with-next", action="store_true", dest="with_next",
        help=(
            "NextDetector を有効化し next1_a/next1_b/dnext_a/dnext_b を"
            "実値で記録する (指標①本命版検証用)。既定は無効 (-1 埋め、後方互換)。"
        ),
    )
    args = parser.parse_args()
    n = collect_lean(
        args.video, args.out_npz,
        max_sec=args.max_sec,
        start_sec=args.start_sec,
        sample_interval_sec=args.sample_interval,
        capture_next=args.with_next,
    )
    print(f"[lean] {args.video.name} -> {args.out_npz} : {n} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
