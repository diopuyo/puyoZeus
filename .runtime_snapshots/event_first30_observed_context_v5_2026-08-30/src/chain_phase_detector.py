"""連鎖中判定モジュール (Phase Z-1)。

score 差分 + 直前盤面の 4+ 同色クラスタ存在から「連鎖中フェーズ」かを
推定する。判定が True の区間では観測盤面を信用せず、ChainSimulator
予測盤面を suspicious 判定の reference にする。

設計方針:
    - **連鎖発火**: 直近 N=3 frame で score 増分 >= SCORE_DELTA_FIRE
    - **連鎖継続**: 連鎖発火後、score 増加が止まり安定するまで継続中扱い
    - 1P / 2P 独立に判定する (片側だけ連鎖していることが大半)
    - 連鎖中の盤面は ChainSimulator(prev_stable_board) で予測

使い方:
    detector = ChainPhaseDetector()
    for state in state_sequence:
        is_chain_p1, is_chain_p2 = detector.update(state)
        # is_chain_p* が True なら 連鎖中、観測棄却して予測盤面を使う
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import Board, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN
from src.chain import ChainSimulator

# score 1 連鎖 = 概ね 40 点 (連結ボーナス無し最低)、現実的な発火閾値
# Z-3A: 連鎖中の score OCR 欠損で発火検出を取りこぼすため緩和 (200→80)
SCORE_DELTA_FIRE: int = 80
# 連鎖中とみなす最大持続時間 (秒)
# Z-3A: 連鎖アニメ中の score 欠損で early termination するため延長 (1.5→3.0)
# Z-3K で 1.8 に微調整を試行したが v13 単発で悪化 → 3.0 維持
CHAIN_HOLD_SEC: float = 3.0
# 連鎖完了とみなす score 静止フレーム数 (0.1s ごとに 3 frame = 0.3s)
SCORE_STILL_FRAMES: int = 3
# Z-3B: prev → cur で puyo cell 数がこれ以上減少したら連鎖中扱い (score OCR 欠損対応)
# Z-3K で 6 に厳格化を試行したが v13 単発で悪化 → 4 維持
PUYO_DIFF_FIRE: int = 4
# Z-3C: 連鎖完了後にさらに連鎖中扱いを継続する秒数 (落下完了までの境界対策)
CHAIN_TAIL_BUFFER_SEC: float = 0.5
# cycle 71d (案 D6): puyo_diff のみで発火した chain は、 この秒数以内に
# cur_puyo_count が baseline (= 発火前 last_puyo_count) に戻ったら認識ノイズ判定で取り消し.
# β 系 偽発火対策 (= cnn 振動 32↔27 で 1 frame だけ -5 のスパイクで発火するのを抑制).
NOISE_RECOVERY_WINDOW_SEC: float = 0.20
# 取り消し判定の cur >= baseline 許容下限 (= baseline - NOISE_RECOVERY_TOLERANCE).
# 連鎖直後でも CNN ゆらぎがあるため厳密一致でなく ±1 cell の余裕を持たせる.
NOISE_RECOVERY_TOLERANCE: int = 1


@dataclass
class _SidePhase:
    """片側 (1P or 2P) の連鎖フェーズ状態。"""
    is_chain: bool = False
    chain_started_at: float = -1.0
    last_score: int | None = None
    still_count: int = 0
    last_stable_board: Board | None = None
    predicted_board: Board | None = None
    history_score: list[tuple[float, int]] = field(default_factory=list)
    # Z-3B: 直前 frame の puyo cell 数 (board diff 連鎖検出用)
    last_puyo_count: int = 0
    # Z-3C: 連鎖完了後のバッファ期間用、完了時刻を記録
    chain_finished_at: float = -1.0
    # cycle 71d (案 D6): puyo_diff のみで発火した場合の baseline と発火経路.
    # 発火後 NOISE_RECOVERY_WINDOW_SEC 以内に cur_puyo_count が baseline まで回復したら
    # 「CNN 認識ノイズ単発スパイク」 と判定し is_chain を取り消す (= β 系偽発火対策).
    fire_baseline_puyo: int = 0
    fire_via_diff_only: bool = False


@dataclass(frozen=True)
class ChainPhaseResult:
    """update() の返り値。"""
    is_chain_p1: bool
    is_chain_p2: bool
    predicted_p1: Board | None  # 連鎖中の simulator 予測 (連鎖開始時の prev board ベース)
    predicted_p2: Board | None


class ChainPhaseDetector:
    """連鎖発火 → 持続 → 完了 を score 差分で追跡する。

    連鎖中判定の要件:
        - score 急増 (delta >= SCORE_DELTA_FIRE) を 1 frame でも観測
        - 連続 SCORE_STILL_FRAMES frame で score 変化なし → 完了
        - 強制タイムアウト (CHAIN_HOLD_SEC 経過) で完了
    """

    def __init__(
        self,
        score_delta_fire: int = SCORE_DELTA_FIRE,
        chain_hold_sec: float = CHAIN_HOLD_SEC,
        score_still_frames: int = SCORE_STILL_FRAMES,
        puyo_diff_fire: int = PUYO_DIFF_FIRE,
        chain_tail_buffer_sec: float = CHAIN_TAIL_BUFFER_SEC,
    ) -> None:
        self._score_delta_fire = int(score_delta_fire)
        self._chain_hold_sec = float(chain_hold_sec)
        self._score_still_frames = int(score_still_frames)
        self._puyo_diff_fire = int(puyo_diff_fire)
        self._chain_tail_buffer_sec = float(chain_tail_buffer_sec)
        self._sim = ChainSimulator()
        self._p1 = _SidePhase()
        self._p2 = _SidePhase()

    def reset(self) -> None:
        self._p1 = _SidePhase()
        self._p2 = _SidePhase()

    def update(
        self,
        t_sec: float,
        board_p1: Board,
        board_p2: Board,
        score_p1: int | None,
        score_p2: int | None,
    ) -> ChainPhaseResult:
        """1 frame 分の状態を入力し、連鎖中フラグと予測盤面を返す。"""
        is_p1 = self._update_side(self._p1, t_sec, board_p1, score_p1)
        is_p2 = self._update_side(self._p2, t_sec, board_p2, score_p2)
        return ChainPhaseResult(
            is_chain_p1=is_p1,
            is_chain_p2=is_p2,
            predicted_p1=self._p1.predicted_board if is_p1 else None,
            predicted_p2=self._p2.predicted_board if is_p2 else None,
        )

    def _update_side(
        self,
        phase: _SidePhase,
        t_sec: float,
        board: Board,
        score: int | None,
    ) -> bool:
        # score 履歴を更新
        if score is not None:
            phase.history_score.append((t_sec, score))
            if len(phase.history_score) > 30:
                phase.history_score = phase.history_score[-30:]

        # 連鎖中なら完了判定優先
        if phase.is_chain:
            # cycle 71d (案 D6): puyo_diff のみで発火した chain は、
            # 発火後 NOISE_RECOVERY_WINDOW_SEC 以内に cur_puyo_count が baseline まで
            # 戻ったら認識ノイズ単発スパイク判定で chain を取り消す.
            # 真の連鎖時は puyo cells が継続的に減少するので baseline 復帰しない.
            if (
                phase.fire_via_diff_only
                and t_sec - phase.chain_started_at <= NOISE_RECOVERY_WINDOW_SEC
            ):
                cur_n = self._count_puyo(board)
                if cur_n >= phase.fire_baseline_puyo - NOISE_RECOVERY_TOLERANCE:
                    phase.is_chain = False
                    phase.predicted_board = None
                    phase.fire_via_diff_only = False
                    phase.last_stable_board = board.copy()
                    phase.last_puyo_count = cur_n
                    return False
            if self._should_finish(phase, t_sec, score):
                phase.is_chain = False
                phase.predicted_board = None
                phase.last_stable_board = board.copy()
                phase.chain_finished_at = t_sec  # Z-3C: 完了時刻記録
                phase.fire_via_diff_only = False
                return False
            return True

        # Z-3C: 連鎖完了**後**の tail buffer 期間中は連鎖中扱い (完了 frame は除外)
        if (phase.chain_finished_at >= 0
                and 0 < t_sec - phase.chain_finished_at
                <= self._chain_tail_buffer_sec):
            phase.last_puyo_count = self._count_puyo(board)
            return True

        # 連鎖中でない: 発火検出
        # Z-3A: _has_erasable チェック削除。score 急増 OR Z-3B board diff のみ。
        # Z-3B: score OCR 欠損中でも、prev → cur で puyo cell 数が急減したら
        # 連鎖中扱い (chain animation で観測盤面の puyo が消える)。
        delta = self._compute_delta(phase, score)
        cur_puyo_count = self._count_puyo(board)
        puyo_diff = phase.last_puyo_count - cur_puyo_count
        score_fired = delta >= self._score_delta_fire
        diff_fired = (
            phase.last_puyo_count > 0
            and puyo_diff >= self._puyo_diff_fire
        )
        if score_fired or diff_fired:
            phase.is_chain = True
            phase.chain_started_at = t_sec
            phase.still_count = 0
            # cycle 71d (案 D6): 取り消し判定用 baseline と発火経路を記録.
            # score_fired は信頼度高い (= 実際に score 増加) ので取り消し対象外.
            phase.fire_baseline_puyo = phase.last_puyo_count
            phase.fire_via_diff_only = bool(diff_fired) and not bool(score_fired)
            try:
                if (phase.last_stable_board is not None
                        and self._has_erasable(phase.last_stable_board)):
                    result = self._sim.simulate(
                        phase.last_stable_board.copy(),
                    )
                    phase.predicted_board = result.final_board
                else:
                    phase.predicted_board = None
            except Exception:
                phase.predicted_board = None
            phase.last_score = score
            phase.last_puyo_count = cur_puyo_count
            return True

        # 連鎖中でない: 観測盤面を「直前安定盤面」として保持
        phase.last_stable_board = board.copy()
        phase.last_puyo_count = cur_puyo_count
        if score is not None:
            phase.last_score = score
        return False

    def _should_finish(
        self, phase: _SidePhase, t_sec: float, score: int | None,
    ) -> bool:
        """連鎖完了判定: score 静止 N frame か timeout で完了。"""
        if t_sec - phase.chain_started_at >= self._chain_hold_sec:
            return True
        if score is not None and phase.last_score is not None:
            if score == phase.last_score:
                phase.still_count += 1
            else:
                phase.still_count = 0
                phase.last_score = score
            if phase.still_count >= self._score_still_frames:
                return True
        return False

    def _compute_delta(
        self, phase: _SidePhase, score: int | None,
    ) -> int:
        """直近 3 frame の最大 score 増分を返す。"""
        if score is None or phase.last_score is None:
            return 0
        delta = score - phase.last_score
        return max(0, delta)

    @staticmethod
    def _has_erasable(board: Board | None) -> bool:
        """盤面に 4+ 同色クラスタがあるか (発火可能性チェック)。"""
        if board is None:
            return False
        try:
            sim = ChainSimulator()
            return len(sim.find_erasable_groups(board)) > 0
        except Exception:
            return False

    @staticmethod
    def _count_puyo(board: Board) -> int:
        """盤面の puyo cell 数 (EMPTY/UNKNOWN 以外)。"""
        from src.board import (
            BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN,
        )
        n = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    n += 1
        return n


__all__ = [
    "ChainPhaseDetector",
    "ChainPhaseResult",
    "SCORE_DELTA_FIRE",
    "CHAIN_HOLD_SEC",
]
