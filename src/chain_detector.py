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
from src.production_config import GHOST_CHAIN_RULE_ENABLED
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

# 修正C (2026-07-24): 偽連鎖イベント抑制用 debounce 確認フレーム数。
# 1 = 既定・従来通り即時確定 (bit-identical, backwards compat)。
# 2 以上を指定すると、drop 検出が debounce_confirm_frames 回連続で
# 観測されるまで ChainEvent 確定を保留する（1 フレームの色フリッカーが
# 作る見かけの減少を「本物の連鎖」と誤認しないようにする）。
# 詳細は VideoChainTracker._update_with_debounce の docstring 参照。
DEBOUNCE_CONFIRM_FRAMES: int = 1

# ChainFineTuner が出力する calibration ファイル (Phase I)
DEFAULT_CALIBRATION_PATH: Path = Path("data/verify/chain_tracker_calibration.json")

# 発火検知経路識別子 (ChainEvent.mechanism、2026-08-02 Step0検証を受けて追加)。
# Step0 実測 (logs/step0_diag/aggregate_result.log): formula が大半 (89.7%) を
# 先着検知するが、baseline のみが捕まえる「難しい連鎖」(5.9%) は
# オフライン発火クラスタリング側の board_ref_index と grid_match 0% だった。
# mechanism を記録し Step2 (仮想盤面再構成) で経路別の信頼度を扱えるようにする。
CHAIN_MECHANISM_BASELINE: str = "baseline"      # ① VideoChainTracker (puyo数減少検知)
CHAIN_MECHANISM_SCORE_JUMP: str = "score_jump"  # ② score急増早期発火 (機能B)
CHAIN_MECHANISM_FORMULA: str = "formula"        # ③ 掛け算式検知早期発火 (機能D)
CHAIN_MECHANISM_LANDING: str = "landing"        # ④ 着地直後即時連鎖判定

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
        mechanism: 発火を捕まえた検知経路 (CHAIN_MECHANISM_* のいずれか)。
            2026-08-02 追加、既定 None (後方互換)。既存呼び出し (mechanism 未指定)
            は None のままで挙動不変。Step2 (仮想盤面再構成) で経路別の
            信頼度を扱うために追加した optional 属性。
        score_estimated: total_score/base_score が実測 (score OCR 差分) でなく
            ChainSimulator によるシミュレーション推定であることを示すフラグ
            (根治①, 2026-08-13, docs/KNOWN_WEAKNESSES.md W7)。formula/landing
            経路の疑似 ChainEvent は本来スコアを直接観測できず、従来は 0
            固定だった (W7 実測: 全連鎖の6.14%が該当し、先読み評価#9の
            起動ゲートを阻害)。enable_pseudo_chain_score_fill=True で
            calculate_chain_score(検証済みChainResult) を充填した場合に
            True。既定 False (未充填、旧挙動 bit-identical、後方互換)。
            推定値は認識の連結欠損 (W1) により実際より低く出る場合がある
            点に注意 (simulate は真値を過小評価しがちという既知の弱点)。
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
    mechanism: str | None = None
    score_estimated: bool = False


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
        debounce_confirm_frames: int = DEBOUNCE_CONFIRM_FRAMES,
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

        # 幽霊連鎖ルール (2026-08-10 本番ON採用): production_config.py が単一情報源。
        self._simulator = ChainSimulator(
            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        )
        self._history: list[tuple[float, Board]] = []
        self._last_stable_count: int | None = None
        self._last_stable_board: Board | None = None
        self._last_stable_t: float | None = None
        # 修正C: debounce 確認カウンタ。0 = 候補なし（通常状態）。
        # debounce_confirm_frames<=1 の既定経路では未使用（値は常に 0 のまま）。
        self._debounce_frames = max(1, int(debounce_confirm_frames))
        self._pending_confirm_count: int = 0

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
            - debounce_confirm_frames > 1 の場合、上記の減少判定が
              その回数分連続で観測されて初めて ChainEvent を確定する
              （修正C: 1 フレームの色フリッカーによる偽イベント抑制）。
              既定値 1 では従来通り即時確定（bit-identical）。
        """
        current_count = count_non_empty(board)
        self._history.append((t_sec, board))
        # 履歴は必要最小限だけ残す
        if len(self._history) > self._lookback + 2:
            self._history.pop(0)

        if self._last_stable_count is None:
            # 初回
            self._advance_stable(current_count, board, t_sec)
            return None

        drop_detected = (
            self._last_stable_count - current_count >= self._erasure_min_drop
        )

        if self._debounce_frames <= 1:
            # 既定経路（debounce 無効）: 従来通り即時判定。bit-identical。
            event = self._try_emit_event(t_sec) if drop_detected else None
            self._advance_stable(current_count, board, t_sec)
            return event

        return self._update_with_debounce(t_sec, current_count, board, drop_detected)

    def _advance_stable(self, count: int, board: Board, t_sec: float) -> None:
        """現フレームを新しい stable 基準点として記録する（次回比較用）。"""
        self._last_stable_count = count
        self._last_stable_board = board
        self._last_stable_t = t_sec

    def _try_emit_event(self, t_sec: float) -> ChainEvent | None:
        """last_stable_board を simulate し、有効な連鎖なら ChainEvent を返す。

        simulate 結果の chain_count が 0（= 4連結消去対象なし）の場合は
        誤検出（noise）として None を返す。
        """
        assert self._last_stable_board is not None
        assert self._last_stable_t is not None
        sim = self._simulator.simulate(self._last_stable_board)
        if sim.chain_count < 1:
            return None
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
            mechanism=CHAIN_MECHANISM_BASELINE,
        )
        self._leftover = ojama_r.leftover_score
        return event

    def _update_with_debounce(
        self,
        t_sec: float,
        current_count: int,
        board: Board,
        drop_detected: bool,
    ) -> ChainEvent | None:
        """修正C: debounce_confirm_frames > 1 時の候補確認ロジック。

        drop_detected が debounce_frames 回連続で観測されて初めて
        （凍結した last_stable_board を simulate して）ChainEvent を確定する。
        途中で drop が消えた（=色フリッカーが1フレームで解消した）場合は
        候補を破棄し、そのフレームを新しい stable 基準として採用する
        （偽イベントを握りつぶす）。本物の連鎖は drop が継続 or 拡大する
        ため debounce_frames 回以内に確定し、速い追撃連鎖も取りこぼさない。
        """
        if self._pending_confirm_count == 0:
            if drop_detected:
                # 候補として保留（last_stable はまだ更新しない）
                self._pending_confirm_count = 1
                return None
            self._advance_stable(current_count, board, t_sec)
            return None

        if not drop_detected:
            # drop が消失 = フリッカー解消。候補を破棄し現フレームを採用。
            self._pending_confirm_count = 0
            self._advance_stable(current_count, board, t_sec)
            return None

        self._pending_confirm_count += 1
        if self._pending_confirm_count < self._debounce_frames:
            return None  # まだ確認フレーム数未達、last_stable は保留のまま

        event = self._try_emit_event(t_sec)
        self._pending_confirm_count = 0
        self._advance_stable(current_count, board, t_sec)
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

    @property
    def debounce_confirm_frames(self) -> int:
        """修正C debounce 設定値（呼出元が reset 時の再構築に使う）。"""
        return self._debounce_frames


# ============================
# バッチ処理用ユーティリティ
# ============================


def track_chains(
    frames: Iterable[tuple[float, Board]],
    match_start_sec: float = 0.0,
    prev_leftover: int = 0,
    debounce_confirm_frames: int = DEBOUNCE_CONFIRM_FRAMES,
) -> list[ChainEvent]:
    """
    フレーム列から検出された全連鎖イベントを一括で返す。

    Args:
        frames: (時刻 [秒], Board) の iterable。時刻は単調増加前提。
        match_start_sec: 試合開始秒（マージンタイム計算用）。
        prev_leftover: 前試合からの繰越（通常 0）。
        debounce_confirm_frames: 修正C debounce（既定 1 = 従来通り）。

    Returns:
        list[ChainEvent]
    """
    tracker = VideoChainTracker(
        match_start_sec=match_start_sec,
        prev_leftover=prev_leftover,
        debounce_confirm_frames=debounce_confirm_frames,
    )
    events: list[ChainEvent] = []
    for t, board in frames:
        ev = tracker.update(t, board)
        if ev is not None:
            events.append(ev)
    return events
