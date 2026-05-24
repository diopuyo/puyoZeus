"""
複数フレームから連鎖発火イベントを検出するモジュール。

原理:
    連鎖が発火すると盤面の puyo 数が「消去→落下→消去→落下」で段階的に減る。
    単一フレームでは静的盤面しか見えないが、連続フレームの盤面差分を取ると
    連鎖の段数・消去数を後付けで特定できる。

アプローチ:
    1. 試合開始後、一定間隔 (interval) ごとに盤面分類を実行
    2. 連続するフレーム間で puyo 総数が大きく減ったら「消去イベント」
    3. 消去前の盤面を `ChainSimulator.simulate()` で full simulate して連鎖確定
    4. シミュレーション結果から得点とおじゃま送出を計算

使い方:
    from src.chain_detector import VideoChainTracker
    tracker = VideoChainTracker()
    for t, board in frame_stream:
        event = tracker.update(t, board)
        if event:
            print(f"連鎖: {event.chain_count}連鎖 score={event.total_score}")

制約:
    - interval が粗いと複数連鎖を 1 回とみなす可能性（ok, 後で simulate が解決）
    - 連鎖中のエフェクト画面では CNN が誤分類する可能性
      → 直前の stable board を使う
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, Board
from src.chain import ChainSimulator
from src.scoring import (
    ALL_CLEAR_BONUS,
    ChainScoreResult,
    OJAMA_RATE_STANDARD,
    calculate_chain_score,
    compute_effective_rate,
    score_to_ojama,
)

# 消去イベント検出の最小減少数（4 = MIN_ERASE_COUNT）
ERASURE_MIN_DROP: int = 4

# 連鎖中フレームは board が不安定になりやすいので、
# このフレーム数だけ前を「発火前 board」とみなす
SNAPSHOT_LOOKBACK: int = 2

# ChainFineTuner が出力する calibration ファイル (Phase I)
DEFAULT_CALIBRATION_PATH: Path = Path("data/verify/chain_tracker_calibration.json")

# calibration cache: (path, mtime) -> dict
_CALIBRATION_CACHE: dict[tuple[str, float], dict] = {}


def _load_calibration(path: Path) -> dict | None:
    """chain_tracker_calibration.json を mtime キャッシュ付きで読込む.

    silent fallback: 読込失敗・不在は None を返す.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = (str(path), float(mtime))
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    _CALIBRATION_CACHE[key] = data
    return data


def count_non_empty(board: Board) -> int:
    """盤面内の非空セル数（おじゃま含む）。"""
    total = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board.get(r, c) != COLOR_EMPTY:
                total += 1
    return total


@dataclass(frozen=True)
class ChainEvent:
    """
    検出した 1 回の連鎖発火イベント。

    Attributes:
        trigger_sec: 連鎖発火直前の時刻
        end_sec: 連鎖が落ち着いた時刻
        before_board: 発火前盤面（simulate 入力）
        chain_count: 連鎖数
        total_erased: 消去総数（おじゃま除く）
        total_score: 合計得点（直前全消しの持ち越しボーナス込み）
        base_score: 持ち越し抜きの素点（連鎖そのものの得点）
        all_clear_bonus_applied: 直前の全消しから持ち越されたボーナスが今回加算されたか
        ojama_sent: 相手に送るおじゃま数
        leftover_score: 繰越余り
        is_all_clear: 連鎖終了後盤面が全消し（次回ボーナスフラグ）
    """
    trigger_sec: float
    end_sec: float
    before_board: Board
    chain_count: int
    total_erased: int
    total_score: int
    base_score: int
    all_clear_bonus_applied: int
    ojama_sent: int
    leftover_score: int
    is_all_clear: bool


class VideoChainTracker:
    """
    連続する (時刻, Board) を受け取り、連鎖イベントを検出する。

    Usage:
        tracker = VideoChainTracker(elapsed_sec_provider=lambda t: t - match_start)
        for t, board in stream:
            ev = tracker.update(t, board)
            if ev: ...

    状態:
        - last_stable_board: 直近で「消去なし」の安定盤面
        - history: 最近の N フレーム盤面（snapshot_lookback 用）
    """

    def __init__(
        self,
        erasure_min_drop: int = ERASURE_MIN_DROP,
        snapshot_lookback: int = SNAPSHOT_LOOKBACK,
        ojama_rate_base: int = OJAMA_RATE_STANDARD,
        match_start_sec: float = 0.0,
        prev_leftover: int = 0,
        prev_all_clear_pending: bool = False,
        apply_calibration: bool = False,
        calibration_path: Path | str | None = None,
    ) -> None:
        # apply_calibration=True なら data/verify/chain_tracker_calibration.json
        # の値で erasure_min_drop / snapshot_lookback を上書きする。
        # silent fallback: 不在・破損ならコンストラクタ引数値を維持。
        if apply_calibration:
            cal_path = (
                Path(calibration_path)
                if calibration_path is not None
                else DEFAULT_CALIBRATION_PATH
            )
            data = _load_calibration(cal_path)
            if data is not None:
                v_emin = data.get("erasure_min_drop")
                v_look = data.get("snapshot_lookback")
                if isinstance(v_emin, int) and v_emin > 0:
                    erasure_min_drop = v_emin
                if isinstance(v_look, int) and v_look > 0:
                    snapshot_lookback = v_look
        self._erasure_min_drop = erasure_min_drop
        self._lookback = max(1, snapshot_lookback)
        self._rate_base = ojama_rate_base
        self._match_start_sec = match_start_sec
        self._leftover = prev_leftover
        # 前試合からの全消し持ち越しがあれば True
        self._all_clear_pending = prev_all_clear_pending

        self._simulator = ChainSimulator()
        self._history: list[tuple[float, Board]] = []
        self._last_stable_count: int | None = None
        self._last_stable_board: Board | None = None
        self._last_stable_t: float | None = None

    # ============================
    # 状態更新
    # ============================

    def update(self, t_sec: float, board: Board) -> ChainEvent | None:
        """
        新フレームを 1 枚受け取る。連鎖イベントが確定したら ChainEvent を返す。

        検出ロジック:
            - 前回 stable 時点と比較して非空セル数が erasure_min_drop 以上減少
            - → stable 時点の board を発火前として simulate
            - simulate 結果の chain_count が 0 なら誤検出（noise）として破棄
        """
        current_count = count_non_empty(board)
        self._history.append((t_sec, board))
        # 履歴は必要最小限だけ残す
        if len(self._history) > self._lookback + 2:
            self._history.pop(0)

        event: ChainEvent | None = None

        if self._last_stable_count is None:
            # 初回
            self._last_stable_count = current_count
            self._last_stable_board = board
            self._last_stable_t = t_sec
            return None

        drop = self._last_stable_count - current_count
        if drop >= self._erasure_min_drop:
            # 消去発生疑い
            assert self._last_stable_board is not None
            assert self._last_stable_t is not None
            sim = self._simulator.simulate(self._last_stable_board)
            if sim.chain_count >= 1:
                scored = calculate_chain_score(sim)
                # 前回全消しのボーナスを今回に持ち越し（公式仕様: 次連鎖発火時加算）
                bonus = ALL_CLEAR_BONUS if self._all_clear_pending else 0
                effective_score = scored.total_score + bonus
                # 持ち越しフラグを「今回が全消しなら次回」に更新
                self._all_clear_pending = scored.is_all_clear

                elapsed = max(0.0, self._last_stable_t - self._match_start_sec)
                ojama_r = score_to_ojama(
                    effective_score,
                    prev_leftover=self._leftover,
                    elapsed_sec=elapsed,
                    rate_base=self._rate_base,
                )
                event = ChainEvent(
                    trigger_sec=self._last_stable_t,
                    end_sec=t_sec,
                    before_board=self._last_stable_board,
                    chain_count=sim.chain_count,
                    total_erased=sim.total_erased,
                    total_score=effective_score,
                    base_score=scored.total_score,
                    all_clear_bonus_applied=bonus,
                    ojama_sent=ojama_r.ojama_count,
                    leftover_score=ojama_r.leftover_score,
                    is_all_clear=scored.is_all_clear,
                )
                self._leftover = ojama_r.leftover_score

        # 現在のフレームを stable として更新（次の比較用）
        self._last_stable_count = current_count
        self._last_stable_board = board
        self._last_stable_t = t_sec
        return event

    # ============================
    # ヘルパ
    # ============================

    @staticmethod
    def _is_all_clear(board: Board) -> bool:
        return count_non_empty(board) == 0

    def reset_leftover(self, value: int = 0) -> None:
        """試合切り替え時に繰越をクリア。"""
        self._leftover = value

    def reset_all_clear_pending(self, value: bool = False) -> None:
        """試合切り替え時に全消し持ち越しをクリア（通常 False）。"""
        self._all_clear_pending = value

    @property
    def current_leftover(self) -> int:
        return self._leftover

    @property
    def all_clear_pending(self) -> bool:
        return self._all_clear_pending


# ============================
# バッチ処理用ユーティリティ
# ============================


def track_chains(
    frames: Iterable[tuple[float, Board]],
    match_start_sec: float = 0.0,
    prev_leftover: int = 0,
) -> list[ChainEvent]:
    """
    フレーム列から検出された全連鎖イベントを一括で返す。

    Args:
        frames: (時刻 [秒], Board) の iterable。時刻は単調増加前提。
        match_start_sec: 試合開始秒（マージンタイム計算用）。
        prev_leftover: 前試合からの繰越（通常 0）。

    Returns:
        list[ChainEvent]
    """
    tracker = VideoChainTracker(
        match_start_sec=match_start_sec,
        prev_leftover=prev_leftover,
    )
    events: list[ChainEvent] = []
    for t, board in frames:
        ev = tracker.update(t, board)
        if ev is not None:
            events.append(ev)
    return events
