"""BoardStateMachine — 盤面状態遷移管理 (Phase B-1).

ぷよぷよの 1 プレイヤー側盤面の状態 (STABLE / TSUMO_FALL / CHAIN /
OJAMA_FALL / EFFECT / MENU) を保持し、各状態で「認識を信用するか / 物理推論
を使うか」を切り替えるためのフレームワーク。

新方針 (project_recognition_strategy_pivot):
    - STABLE 状態のみ CNN 認識結果を盤面確定に使う。
    - それ以外の state ではアクション中なので CNN 出力は drift detector
      参照のみ。盤面確定は ChainSimulator / ネクスト履歴 / score OCR で行う。

このモジュールは骨組み (state enum + StateContext + 遷移ルール) のみ提供する。
各 detector の実装は Phase B-2 で `StateTransitionDetector` Protocol を満たす
形で個別に追加する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, Board

# ============================
# 定数
# ============================

# 平常時 = 連続 N frame で同一盤面が観測されたら STABLE 確定
DEFAULT_STABLE_FRAME_COUNT: int = 6

# F = STABLE 復帰ゲート (2026-05-21 cycle 56):
# NON-STABLE 中の cnn_board を最大 N frame 保持し、 STABLE 復帰時に
# 多数決を取って「瞬間的な背景誤認」 を抑制する。
# ユーザー指摘「置く直前の推論で背景がぷよ誤認、 2 秒残る」 への対策。
# 2026-05-23: G 拡張 (= N=8/votes=5) は v89m7 で逆効果 (= flicker +8%)、
# 8 動画でも c56_v3b 単独不採用と合わせ「baseline 維持」 判定 → revert。
DEFAULT_NON_STABLE_HISTORY_SIZE: int = 5
# EMPTY → 色 遷移 cell を採用するための NON-STABLE 中の最低観測 frame 数。
# N=5 中 3 frame 以上で同色観測 = 信頼。 それ以下は瞬間的誤認とみなし baseline 維持。
DEFAULT_EMPTY_TO_COLOR_MIN_VOTES: int = 3

# B1 (M1 warmup guard): STABLE 判定直後に confirmed 更新を N frame スキップ。
# NON-STABLE → STABLE 遷移直後は CNN 出力がまだ不安定 (= エフェクト残光、
# 背景誤認の溶け残り) であり、この期間に _update_within_current_state 経由で
# confirmed が更新されると誤色が焼き付く。
# WARMUP 中は STABLE state を維持したまま confirmed を凍結する。
# 0.4 秒 @ 30fps。
STABLE_WARMUP_FRAMES: int = 12

# 推論優先 (NON-STABLE) state の集合
NON_STABLE_STATES: frozenset["BoardState"] = frozenset()  # 後で初期化

# GRAVITY_SETTLE 状態定数 (feat/gravity-settle-2026-06-05)
# -------------------------------------------------------
# 連鎖終了直後の重力settle/着地中 window 用定数。
# GravitySettleDetector はこれらを参照して STABLE 復帰タイミングを制御する。
# -------------------------------------------------------

# ぷよ数変化が「ほぼ静止」と判定するための最小連続フレーム数。
# @30fps: 8f ≈ 0.27s。連鎖終了直後の board_log 分析で physics_fix が
# 遷移+0fに10-13件集中するため、それが収まるまで待つ設計。
GRAVITY_SETTLE_MIN_FRAMES: int = 8

# GRAVITY_SETTLE の最大保持時間 (秒)。タイムアウトで強制 STABLE 復帰。
# 最長連鎖でも settle は 0.5-1.0s 程度。1.5s は十分な安全マージン。
GRAVITY_SETTLE_MAX_SEC: float = 1.5

# 最低待機フレーム数 (physics_fix 解消確認用)。
# GRAVITY_SETTLE_MIN_FRAMES より小さい値を設定できる下限ガード。
# 連鎖エフェクト残光が消えるまでの最小保護時間。
GRAVITY_SETTLE_PHYSICS_CLEAR_MIN: int = 3

# settle 中に「ぷよ数変化が安定」と見なすフレーム間差分の上限。
# raw CNN ぷよ数が連続フレームで ±N 以内なら静止とみなす。
GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD: int = 2

# 設計C 事後復旧ゲート (fix/v70-zeropatch-redyellow, 2026-06-02):
# STABLE 中に confirmed != (CNN==HSV の合意値) が N フレーム継続したセルを
# confirmed に追従させる双方向ゲート。
#   方向1 (空→色): confirmed=EMPTY, CNN==HSV=有効色 → 色に復活 (幽霊ぷよ追加)
#   方向2 (色→空): confirmed=色,   CNN==HSV=EMPTY   → 空に修正 (幽霊ぷよ除去)
#   方向3 (色→色): confirmed=色A,  CNN==HSV=色B     → 色Bに訂正 (誤色修正)
# N フレーム = 8 frame (約 0.27s @ 30fps) で実質的な持続合意と判断。
STABLE_RECOVERY_MIN_FRAMES: int = 8

# 復旧ゲートの「合意値」として無効な色 = UNKNOWN(10) のみ。
# EMPTY(0) / OJAMA(9) は方向2の除去ターゲットなので含めない。
RECOVERY_EXCLUDED_COLORS: frozenset[int] = frozenset({10})


class BoardState(Enum):
    """1 プレイヤー側盤面の状態。"""

    MENU = "menu"            # 試合外 (タイトル / リザルト / リトライ)
    STABLE = "stable"        # 平常時、CNN 出力を盤面確定に使う
    TSUMO_FALL = "tsumo_fall"  # ツモ落下中
    CHAIN = "chain"          # 連鎖中 (消去 + 重力)
    OJAMA_FALL = "ojama_fall"  # おじゃま落下中
    EFFECT = "effect"        # 全消し演出 / 連鎖カットイン等
    # feat/gravity-settle-2026-06-05: 連鎖終了直後の重力 settle/着地中。
    # CHAIN 終了後から board が物理的に静止するまでの window。
    # NON_STABLE_STATES に含まれるため採点外・confirmed 凍結対象。
    # backwards compat: 既存コードは GRAVITY_SETTLE を認識しないが、
    # NON_STABLE_STATES に加えることで「採点外」として既存の is_action() が True を返し、
    # 挙動上は他の NON-STABLE state と同等に扱われる。
    GRAVITY_SETTLE = "gravity_settle"


# 推論優先 state (= 認識結果を盤面確定に使わない state)
NON_STABLE_STATES = frozenset({
    BoardState.TSUMO_FALL,
    BoardState.CHAIN,
    BoardState.OJAMA_FALL,
    BoardState.EFFECT,
    # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE は採点外 (confirmed 凍結)。
    # backwards compat: default OFF (enable_gravity_settle_state=False) なら
    # この state に遷移しないため挙動は完全不変。
    BoardState.GRAVITY_SETTLE,
})


# ============================
# データクラス
# ============================


@dataclass
class StateContext:
    """1 プレイヤー側 state machine の状態スナップショット。

    Attributes:
        state: 現在の BoardState。
        frame_idx: 入力 frame 番号 (連番)。
        time_sec: 現在 frame の時刻。
        confirmed_board: 直近 STABLE 確定盤面 (None なら未確定)。
        last_stable_idx: confirmed_board の frame_idx。
        pending_board: STABLE 候補だがまだ多数決を満たしていない盤面。
        pending_count: pending_board の連続観測 frame 数。
        next_queue: 直近のネクスト履歴 (ペア色)。新しいほど末尾。
        chain_count: CHAIN state 時の連鎖カウンタ (0 なら CHAIN 外)。
        ojama_pending: OJAMA_FALL state 時の落下予定個数。
    """

    state: BoardState = BoardState.MENU
    frame_idx: int = 0
    time_sec: float = 0.0
    confirmed_board: Board | None = None
    last_stable_idx: int = -1
    pending_board: Board | None = None
    pending_count: int = 0
    next_queue: list[tuple[int, int]] = field(default_factory=list)
    chain_count: int = 0
    ojama_pending: int = 0
    # F (cycle 56): NON-STABLE 中の cnn_board 履歴。 STABLE 復帰時に多数決を取り、
    # 瞬間的な背景誤認が confirmed_board に焼き付くのを防ぐ。 最大 N frame 保持。
    non_stable_cnn_history: list[Board] = field(default_factory=list)
    # B1 (M1 warmup guard): NON-STABLE → STABLE 遷移直後の confirmed 凍結残りフレーム数。
    # 0 = warmup なし (通常 STABLE)。0 より大きければ confirmed 更新をスキップ。
    # backwards compat: default=0 で既存動作と同一。
    stable_warmup_remaining: int = 0
    # 設計C 事後復旧ゲート (2026-06-02):
    # STABLE 中に confirmed==EMPTY かつ CNN==HSV が同一有効色で連続している
    # フレーム数を (row, col) → int で管理するカウンタ。
    # STABLE→NON-STABLE 遷移時にクリアする。
    # backwards compat: default=empty dict で既存動作と完全同一。
    stable_recovery_counters: "dict[tuple[int,int],int]" = field(
        default_factory=dict,
    )
    # 設計C: 復旧済みセルの集合。STABLE→NON-STABLE 遷移時にクリア。
    # 復旧したセルは non_stable_cnn_history へ復旧色を強制投入し
    # 次の NON-STABLE→STABLE 多数決で再崩壊しにくくする。
    recovery_cells: "set[tuple[int,int]]" = field(
        default_factory=set,
    )

    def is_stable(self) -> bool:
        """STABLE 確定中か (= 認識結果を盤面確定に使う)。"""
        return self.state == BoardState.STABLE

    def is_action(self) -> bool:
        """アクション中か (= NON-STABLE)。"""
        return self.state in NON_STABLE_STATES


# ============================
# Detector Protocol (B-2 で具体実装を差し込む)
# ============================


@dataclass(frozen=True)
class DetectorSignals:
    """1 frame 分の入力シグナル。各 detector はこの dataclass を読み取る。

    Attributes:
        time_sec: 現 frame の時刻。
        cnn_board: CNN 認識結果の生盤面 (drift detector 入力にも使う)。
        is_match_active: match_state.py の試合中判定。
        chain_event: VideoChainTracker から渡される連鎖イベント (確定時のみ)。
        chain_counter_visible: 画面上の ★n連鎖★ が検出されたか (B-3 で実装)。
        score_delta: 直前 STABLE からの score 差分 (B-4 で実装)。
        next_pair: 直近で確定した next ペア (B-2 で next_detector hookup)。
        slide_motion: NEXT ROI のスライド motion (= ツモが画面から消えて
            次のツモに繰り上がった signal)。Phase I R-7。default False で
            backward compat。
        placement_validated: 着地後の cnn_board と baseline の色 count delta が
            落下ペアと整合 (= 自己整合性チェック合格) しているか。
            Phase I R-1。default False で backward compat。
    """

    time_sec: float
    cnn_board: Board
    is_match_active: bool
    chain_event: object | None = None
    chain_counter_visible: bool = False
    score_delta: int = 0
    next_pair: tuple[int, int] | None = None
    slide_motion: bool = False
    placement_validated: bool = False
    # 設計C 事後復旧ゲート用 HSV-only 盤面 (2026-06-02)。
    # ImageReader の HSV 単独分類結果を渡し、CNN との独立二重合意チェックに使う。
    # None なら復旧ゲートの「CNN==HSV 一致」要件が満たせないため発火しない。
    # backwards compat: default None で既存挙動と完全同一。
    hsv_board: "Board | None" = None
    # 演出中フラグ (telop/match_end/all_clear/win_panel/chain_animation の OR)。
    # True なら EffectPhaseDetector が EFFECT state に遷移し、 CNN 出力を
    # 信用せず直前 STABLE 盤面を hold する (memory `feedback_chain_phase_physics_only`)。
    effect_visible: bool = False
    # 試合開始直後 window (cycle 71v, 2026-05-14). True の間は初回 STABLE 確定で
    # confirmed_board に空 Board() を強制し、 CNN の背景誤認による
    # 開始フィールド corruption を防ぐ。 ぷよぷよ eスポーツの物理ルール:
    # 「試合開始時のフィールドは必ず空」 を構造的に取り込んだ防御層。
    match_just_started: bool = False
    # フェーズ A 精緻化: 可視最上段付近の ROI にお邪魔が存在するか (一次判定)。
    # OjamaVisualDetector の内部カウンタ更新前に RecognitionPipeline 側で
    # _count_top_ojama() を呼び出してセットする。default False で既存挙動不変。
    # backwards compat: default False で既存動作と完全同一。
    ojama_top_positive: bool = False
    # 案P3: CHAIN_MAX_HOLD_SEC を超過して active_chain が強制クリアされたか。
    # True のとき ChainPhaseDetector は ojama_top_positive による STABLE 復帰保留を
    # 無視して強制 STABLE に遷移させる。これにより安全弁が本来機能する。
    # pipeline 側で active_chain が None にクリアされた瞬間の 1 frame だけ True になり、
    # 翌 frame 以降は False に戻る (新連鎖発火時 / STABLE 復帰確定時もリセット)。
    # backwards compat: default False で既存動作と完全同一。
    chain_max_hold_expired: bool = False


class StateTransitionDetector(Protocol):
    """state 遷移検出器の Protocol.

    各 detector は現 context + 入力 signals を見て、遷移先 state を返す。
    遷移しないなら None を返す。
    """

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        ...


# ============================
# State Machine 本体
# ============================


def _boards_equal(a: Board | None, b: Board | None) -> bool:
    """2 つの Board が完全一致か (None 同士は False)。"""
    if a is None or b is None:
        return False
    return a == b


def _vote_majority_board(
    history: list[Board], min_votes: int,
) -> Board:
    """history の各 cell について多数決を取り、 min_votes 以上同色なら採用、
    そうでなければ EMPTY を入れた Board を返す.

    F (cycle 56): STABLE 復帰時の「EMPTY → 色 遷移ガード」 用。
    NON-STABLE 中の cnn_board 履歴から「安定して観測されている色」 のみを抽出。

    Args:
        history: NON-STABLE 中の cnn_board 履歴 (= 最大 N frame)
        min_votes: cell を採用するための最低観測 frame 数

    Returns:
        多数決 board (= 観測不足 cell は EMPTY)
    """
    from src.board import COLOR_EMPTY

    result = Board()
    if not history:
        return result
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            counter: dict[int, int] = {}
            for b in history:
                v = b.get(r, c)
                counter[v] = counter.get(v, 0) + 1
            if not counter:
                continue
            max_v, max_n = max(counter.items(), key=lambda x: x[1])
            if max_n >= min_votes and max_v != COLOR_EMPTY:
                result.set(r, c, max_v)
            # else: EMPTY (= Board default、 観測不足)
    return result


def _merge_diff_only(
    baseline: Board | None, new_cnn: Board,
    *, allow_puyo_to_empty: bool = True,
    empty_to_color_guard: Board | None = None,
) -> Board:
    """baseline をベースに、CNN との差分 cell のみ new_cnn 値で上書き.

    Phase C-5 の「差分のみ反映」コア。baseline cell の情報を絶対に
    壊さないことで、CNN ぶれ (= 背景誤認、エフェクト残光等) が
    confirmed_board に乗るのを構造的に防ぐ。

    Args:
        baseline: 直前の確定盤面 (None なら初回 = CNN 全採用)
        new_cnn: 新たに観測された CNN 盤面
        allow_puyo_to_empty: True なら puyo→空 遷移を許容、False なら
            「置いた puyo は CNN の puyo→空 転換を ban」(= 連鎖発火以外で
            消えない、Phase C-6 の B 強化)
        empty_to_color_guard: F (cycle 56) STABLE 復帰ゲート用の多数決 board.
            None でない場合、 EMPTY → 色 遷移 cell について
            「guard 内で同色が観測されている」 場合のみ採用する。
            これにより、 NON-STABLE 中の瞬間的な背景誤認 (= 1-2 frame 限定)
            が confirmed_board に焼き付くのを防ぐ。 default None で従来挙動。

    Returns:
        merged: baseline + 物理整合性 filter 後の差分のみ更新された盤面
    """
    from src.board import COLOR_EMPTY

    if baseline is None:
        new = new_cnn.copy()
        # 浮きぷよ ban (A): baseline がなくても列ごとに重力整合
        _apply_gravity_filter(new)
        return new

    from src.board import COLOR_UNKNOWN

    merged = baseline.copy()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            base_v = baseline.get(r, c)
            cnn_v = new_cnn.get(r, c)
            if base_v == cnn_v:
                continue
            # D (2026-05-11): cnn が UNKNOWN を返した cell は baseline 維持.
            # telop 被覆や image_reader 不確実時に旧色を保持.
            if cnn_v == COLOR_UNKNOWN:
                continue
            # B: puyo→空 遷移を ban (= 確定 cell 保護)
            if (
                not allow_puyo_to_empty
                and base_v != COLOR_EMPTY
                and cnn_v == COLOR_EMPTY
            ):
                continue  # baseline の puyo を維持
            # F (cycle 56): EMPTY → 色 遷移ガード。 NON-STABLE 中の多数決で
            # 同色が確認できない cell は背景誤認の可能性高 → baseline 維持。
            if (
                base_v == COLOR_EMPTY
                and cnn_v != COLOR_EMPTY
                and empty_to_color_guard is not None
            ):
                guard_v = empty_to_color_guard.get(r, c)
                if guard_v != cnn_v:
                    continue  # baseline (= EMPTY) 維持
            merged.set(r, c, cnn_v)
    # A: 浮きぷよ ban (空 cell の上に puyo は物理的にあり得ない、空に戻す)
    _apply_gravity_filter(merged)
    return merged


def _apply_gravity_filter(board: Board) -> None:
    """浮きぷよ ban (Phase C-6 の A): 空 cell の上に puyo がある場合、
    その上の puyo を空に戻す。連鎖中の重力再配置中に CNN が誤検出した
    「浮きぷよ」を物理的にあり得ないと却下する。

    実装: 各列を下から走査、最下段 (row=12) から上に進み、空セルが
    現れた次の non-EMPTY cell があれば消去 (= 浮き判定)。

    UNKNOWN cell は「判定保留」 なので gap_found に立てない (= 上のぷよを
    erase しない)。 cycle 71v で一時的に UNKNOWN を gap 扱いにしたが、
    2P telop/UI で UNKNOWN になる cell 上のぷよが大量 erase される副作用が
    確認されたため撤回 (2026-05-14)。
    """
    from src.board import COLOR_EMPTY

    for c in range(BOARD_COLS):
        gap_found = False
        for r in range(BOARD_ROWS - 1, -1, -1):
            v = board.get(r, c)
            if v == COLOR_EMPTY:
                gap_found = True
            elif gap_found:
                # 空 cell の上に puyo → 浮き → 空に戻す
                board.set(r, c, COLOR_EMPTY)


class BoardStateMachine:
    """1 プレイヤー側の盤面状態を時系列で追跡する.

    Usage:
        sm = BoardStateMachine(detectors=[...])
        for frame_idx, signals in enumerate(stream):
            ctx = sm.update(frame_idx, signals)
            if ctx.is_stable():
                # CNN 確定盤面が ctx.confirmed_board に入っている
                ...
            else:
                # 推論モード、別途 ChainSimulator 等で盤面を生成
                ...
    """

    def __init__(
        self,
        detectors: list[StateTransitionDetector] | None = None,
        stable_frame_count: int = DEFAULT_STABLE_FRAME_COUNT,
        *,
        non_stable_history_size: int = DEFAULT_NON_STABLE_HISTORY_SIZE,
        empty_to_color_min_votes: int = DEFAULT_EMPTY_TO_COLOR_MIN_VOTES,
        enable_stable_resume_gate: bool = True,
        enable_warmup_guard: bool = False,
        warmup_frames: int = STABLE_WARMUP_FRAMES,
        enable_stable_recovery_gate: bool = False,
        recovery_min_frames: int = STABLE_RECOVERY_MIN_FRAMES,
    ) -> None:
        self._non_stable_history_size = int(non_stable_history_size)
        self._empty_to_color_min_votes = int(empty_to_color_min_votes)
        self._enable_stable_resume_gate = bool(enable_stable_resume_gate)
        # B1 (M1 warmup guard): STABLE 遷移直後の confirmed 凍結を有効化するか。
        # backwards compat: default False で既存動作と同一。
        self._enable_warmup_guard = bool(enable_warmup_guard)
        self._warmup_frames = max(0, int(warmup_frames))
        # 設計C 事後復旧ゲート (2026-06-02):
        # True で STABLE 中の confirmed==EMPTY かつ CNN==HSV 継続セルを復旧する。
        # default False = B1 禁忌隣接のため OFF。viz 検証後に判断する。
        self._enable_stable_recovery_gate = bool(enable_stable_recovery_gate)
        self._recovery_min_frames = max(1, int(recovery_min_frames))
        self._detectors: list[StateTransitionDetector] = (
            list(detectors) if detectors else []
        )
        self._stable_n = max(1, stable_frame_count)
        self._ctx = StateContext()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def context(self) -> StateContext:
        return self._ctx

    def add_detector(self, detector: StateTransitionDetector) -> None:
        """検出器を追加。優先順は登録順。"""
        self._detectors.append(detector)

    def reset(self, *, keep_match_state: bool = False) -> None:
        """state machine を初期化。drift 検出時の re-sync で使う想定。"""
        if keep_match_state:
            kept_state = self._ctx.state if self._ctx.state == BoardState.MENU \
                else BoardState.STABLE
            self._ctx = StateContext(state=kept_state)
        else:
            self._ctx = StateContext()

    def update(
        self, frame_idx: int, signals: DetectorSignals,
    ) -> StateContext:
        """新 frame を投入し、更新後の StateContext を返す."""
        self._ctx.frame_idx = frame_idx
        self._ctx.time_sec = signals.time_sec

        # 試合外なら全部 MENU に倒す (= 認識結果を保持しない)
        if not signals.is_match_active:
            self._ctx.state = BoardState.MENU
            self._ctx.pending_board = None
            self._ctx.pending_count = 0
            self._ctx.confirmed_board = None
            self._ctx.last_stable_idx = -1
            self._ctx.chain_count = 0
            self._ctx.ojama_pending = 0
            return self._ctx

        # 検出器を順に適用、最初に発火したものを採用
        new_state: BoardState | None = None
        for det in self._detectors:
            res = det.detect(self._ctx, signals)
            if res is not None:
                new_state = res
                break

        if new_state is not None:
            self._apply_transition(new_state, signals)
        else:
            self._update_within_current_state(signals)

        # next_queue 更新 (next_pair が新規確定なら追加)
        if signals.next_pair is not None:
            if (
                not self._ctx.next_queue
                or self._ctx.next_queue[-1] != signals.next_pair
            ):
                self._ctx.next_queue.append(signals.next_pair)
                # 履歴は最新 8 件まで保持 (ダブルネクスト + α 余裕)
                if len(self._ctx.next_queue) > 8:
                    self._ctx.next_queue.pop(0)

        return self._ctx

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _apply_transition(
        self, new_state: BoardState, signals: DetectorSignals,
    ) -> None:
        """detector が指定した state へ強制遷移."""
        if new_state == self._ctx.state:
            self._update_within_current_state(signals)
            return
        # state 切り替えで pending をリセット
        self._ctx.pending_board = None
        self._ctx.pending_count = 0
        # NON-STABLE → STABLE 復帰時: 差分のみ反映で baseline 維持
        # (Phase C-5: 全 cell コピーは CNN ぶれを取り込むため廃止)
        if (
            self._ctx.state in NON_STABLE_STATES
            and new_state == BoardState.STABLE
        ):
            # F (cycle 56): NON-STABLE 中の cnn_board 履歴から多数決 board を生成。
            # EMPTY → 色 遷移ガードに渡し、 瞬間的な背景誤認の焼き付きを防ぐ。
            empty_guard: Board | None = None
            if (
                self._enable_stable_resume_gate
                and self._ctx.non_stable_cnn_history
            ):
                empty_guard = _vote_majority_board(
                    self._ctx.non_stable_cnn_history,
                    min_votes=self._empty_to_color_min_votes,
                )
            self._ctx.confirmed_board = _merge_diff_only(
                self._ctx.confirmed_board, signals.cnn_board,
                empty_to_color_guard=empty_guard,
            )
            self._ctx.last_stable_idx = self._ctx.frame_idx
            self._ctx.pending_board = self._ctx.confirmed_board.copy()
            self._ctx.pending_count = 1
            # B1 (M1 warmup guard): STABLE 復帰直後は N frame 間 confirmed 凍結。
            # 遷移直後の CNN 出力が不安定 (エフェクト残光・背景誤認) な期間に
            # _update_within_current_state が confirmed を書き換えるのを防ぐ。
            if self._enable_warmup_guard:
                self._ctx.stable_warmup_remaining = self._warmup_frames
        self._ctx.state = new_state
        # F: state 切替で NON-STABLE history をリセット
        self._ctx.non_stable_cnn_history = []
        if new_state != BoardState.CHAIN:
            self._ctx.chain_count = 0
        if new_state != BoardState.OJAMA_FALL:
            self._ctx.ojama_pending = 0
        # 設計C: STABLE → NON-STABLE 遷移時に復旧カウンタ・復旧済みセル集合をクリア。
        # 次の STABLE 期間は新規状態から積み上げ直す。
        if new_state in NON_STABLE_STATES:
            self._ctx.stable_recovery_counters.clear()
            self._ctx.recovery_cells.clear()

    def _update_within_current_state(
        self, signals: DetectorSignals,
    ) -> None:
        """現 state を維持したまま内部メトリクスのみ更新。"""
        if self._ctx.state in NON_STABLE_STATES:
            # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE 中は
            # non_stable_cnn_history に蓄積しない (連鎖後エフェクト残光・
            # 落下中ぷよの混入による F ガード汚染を防ぐ)。
            # GRAVITY_SETTLE → STABLE 遷移時の _merge_diff_only には
            # empty_guard として None が渡されるため F ガードは発火しない
            # (保守的: 連鎖後は全 cell を新規 STABLE で直接評価)。
            if self._ctx.state == BoardState.GRAVITY_SETTLE:
                # アクション中: 認識結果は盤面確定に使わない
                return
            # F (cycle 56): NON-STABLE 中の cnn_board 履歴を保持。
            # STABLE 復帰時の多数決ガードに使う。 最大 N frame 保持。
            self._ctx.non_stable_cnn_history.append(
                signals.cnn_board.copy(),
            )
            if (
                len(self._ctx.non_stable_cnn_history)
                > self._non_stable_history_size
            ):
                self._ctx.non_stable_cnn_history.pop(0)
            # アクション中: 認識結果は盤面確定に使わない
            return

        # B1 (M1 warmup guard): STABLE 遷移直後の warmup 期間中は confirmed 更新をスキップ。
        # エフェクト残光・背景誤認の CNN 出力が confirmed に焼き付くのを防ぐ。
        # last_stable_idx は更新し「STABLE 中」であることだけ伝える。
        if self._ctx.state == BoardState.STABLE and self._ctx.stable_warmup_remaining > 0:
            self._ctx.stable_warmup_remaining -= 1
            self._ctx.last_stable_idx = self._ctx.frame_idx
            return

        # MENU (試合復帰直後) または STABLE: 連続多数決で confirmed を更新。
        # MENU から N 連続一致で STABLE へ自動遷移する。
        cnn_board = signals.cnn_board
        if _boards_equal(self._ctx.pending_board, cnn_board):
            self._ctx.pending_count += 1
        else:
            self._ctx.pending_board = cnn_board.copy()
            self._ctx.pending_count = 1

        if self._ctx.pending_count >= self._stable_n:
            # Phase C-7 (E-1): CNN を盤面更新ソースから完全排除
            # 初回 STABLE 確定 (= 試合開始時) は連続 N frame 観測の
            # pending_board を採用 (= 試合開始時の初期盤面確定)。
            # その後の連続多数決による confirmed 更新は停止し、盤面確定は
            # state 遷移時 (TSUMO/CHAIN/OJAMA → STABLE) の物理推論のみで行う。
            if self._ctx.confirmed_board is None:
                # cycle 71v (2026-05-14): 試合開始直後 window では物理ルール
                # 「試合開始フィールドは空」 を CNN 多数決より優先。 v51/v70 の
                # 2 試合目開始時の背景誤認が confirmed_board に乗るのを防ぐ。
                if signals.match_just_started:
                    self._ctx.confirmed_board = Board()  # all empty
                elif self._ctx.pending_board is not None:
                    # cycle 71v: 初回 STABLE 確定パスでも gravity filter を適用.
                    # 旧実装は pending_board.copy() で gravity filter が走らず、
                    # CNN の背景誤認が浮きぷよとして confirmed に残留した。
                    b = self._ctx.pending_board.copy()
                    _apply_gravity_filter(b)
                    self._ctx.confirmed_board = b
            # 既に confirmed あれば維持 (= CNN 多数決経路で更新しない)
            self._ctx.last_stable_idx = self._ctx.frame_idx
            self._ctx.state = BoardState.STABLE

        # 設計C 事後復旧ゲート: STABLE 確定後に実行。
        # warmup 期間中は発火させない (背景誤認残光期間を避けるため)。
        if (
            self._enable_stable_recovery_gate
            and self._ctx.state == BoardState.STABLE
            and self._ctx.confirmed_board is not None
            and self._ctx.stable_warmup_remaining == 0
        ):
            _apply_stable_recovery_gate(
                self._ctx, signals, self._recovery_min_frames,
            )


# ============================
# 設計C 事後復旧ゲート ヘルパー
# ============================


def _check_recovery_column(
    confirmed: "Board", col: int, candidates: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """列の重力整合チェック: 下から連続するブロックのみ復旧候補として残す.

    浮きぷよ防止 (安全弁C): 復旧候補セルの下に空 confirmed があれば浮きぷよに
    なるため除外する。列を下段から走査し、confirmed が空でない連続区間のみ許可。

    Args:
        confirmed: 現在の confirmed_board。
        col: 対象列番号 (0-5)。
        candidates: [(row, col, recovery_color), ...] — 復旧候補リスト。

    Returns:
        重力整合チェック通過後の復旧候補リスト。
    """
    # 対象列の既存 confirmed 色マップ (row → color)
    col_confirmed: dict[int, int] = {}
    for r in range(BOARD_ROWS):
        col_confirmed[r] = int(confirmed.get(r, col))

    # 候補を row の降順 (最下段優先) にソート
    col_candidates = sorted(
        [(r, c_col, color) for (r, c_col, color) in candidates if c_col == col],
        key=lambda x: -x[0],
    )
    if not col_candidates:
        return []

    # 最下段から連続して confirmed が空か候補セル (= 仮に復旧した場合) か確認
    # 合格した候補 (下から連続ブロック) のみを返す
    result: list[tuple[int, int, int]] = []
    # まず候補 row の集合を得る
    cand_rows = {r for (r, _, _) in col_candidates}

    for r_desc, c_col, color in col_candidates:
        # このセルの下 (r+1..12) に、確定 EMPTY かつ候補でもないセルがあれば浮き
        floating = False
        for below in range(r_desc + 1, BOARD_ROWS):
            below_v = col_confirmed[below]
            if below_v == COLOR_EMPTY and below not in cand_rows:
                floating = True
                break
        if not floating:
            result.append((r_desc, c_col, color))
    return result


def _collect_recovery_candidates(
    ctx: "StateContext",
    cnn_board: "Board",
    hsv_board: "Board",
    min_frames: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """各セルの合意値チェックとカウンタ更新を行い、候補を方向別に返す.

    双方向の発火候補を収集する:
        - add_candidates: 方向1 (空→色) — 重力整合チェックが必要
        - fix_candidates: 方向2/3 (色→空/色→別色) — 重力整合チェック不要

    安全弁: UNKNOWN(10) は合意値として無効 (RECOVERY_EXCLUDED_COLORS)。

    Args:
        ctx: StateContext (stable_recovery_counters を in-place 更新)。
        cnn_board: CNN 認識盤面。
        hsv_board: HSV 認識盤面。
        min_frames: 発火に必要な連続フレーム数。

    Returns:
        (add_candidates, fix_candidates) のタプル。
        各要素は [(row, col, target_color), ...] 形式。
    """
    recovery_counters = ctx.stable_recovery_counters
    confirmed = ctx.confirmed_board
    assert confirmed is not None

    add_candidates: list[tuple[int, int, int]] = []
    fix_candidates: list[tuple[int, int, int]] = []

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            confirmed_v = int(confirmed.get(r, c))
            cnn_v = int(cnn_board.get(r, c))
            hsv_v = int(hsv_board.get(r, c))

            # UNKNOWN は合意値として無効 → カウンタリセット
            if cnn_v in RECOVERY_EXCLUDED_COLORS or hsv_v in RECOVERY_EXCLUDED_COLORS:
                recovery_counters.pop((r, c), None)
                continue
            # CNN≠HSV → 独立二重合意なし → カウンタリセット
            if cnn_v != hsv_v:
                recovery_counters.pop((r, c), None)
                continue
            # confirmed == 合意値 → 差分なし → カウンタリセット
            if confirmed_v == cnn_v:
                recovery_counters.pop((r, c), None)
                continue

            # CNN==HSV (=合意値) かつ confirmed != 合意値 → カウント継続
            prev_count = recovery_counters.get((r, c), 0)
            new_count = prev_count + 1
            recovery_counters[(r, c)] = new_count
            if new_count < min_frames:
                continue

            # 発火: 方向別に振り分け
            if confirmed_v == COLOR_EMPTY:
                # 方向1: 空→色 (重力整合チェック必要)
                add_candidates.append((r, c, cnn_v))
            else:
                # 方向2: 色→空 / 方向3: 色→別色 (重力整合チェック不要)
                fix_candidates.append((r, c, cnn_v))

    return add_candidates, fix_candidates


def _apply_stable_recovery_gate(
    ctx: "StateContext",
    signals: "DetectorSignals",
    min_frames: int,
) -> None:
    """設計C 事後復旧ゲート本体 (in-place で confirmed_board を更新).

    双方向発火条件 (STABLE state、hsv_board!=None、warmup 外):
        方向1 (空→色): confirmed=EMPTY, CNN==HSV=有効色 が min_frames 連続
        方向2 (色→空): confirmed=色,   CNN==HSV=EMPTY  が min_frames 連続
        方向3 (色→色): confirmed=色A,  CNN==HSV=色B    が min_frames 連続

    安全弁A: hsv_board が None (= CNN 単独) のとき発火しない。
    安全弁B: UNKNOWN(10) は合意値として無効 (RECOVERY_EXCLUDED_COLORS)。
    安全弁C: 方向1のみ列単位の重力整合チェック (_check_recovery_column)。
             方向2/3 は除去・上書き方向なので適用しない。

    振動抑制: 復旧済みセルを recovery_cells に記録し、
    non_stable_cnn_history に復旧後盤面を強制投入する。

    Args:
        ctx: StateContext (in-place 更新)。
        signals: 現フレームのシグナル (cnn_board, hsv_board を参照)。
        min_frames: 発火に必要な連続フレーム数。
    """
    if ctx.confirmed_board is None:
        return
    # 安全弁A: hsv_board が None → 発火しない
    hsv_board = signals.hsv_board
    if hsv_board is None:
        return

    # パス1: 候補収集 + カウンタ更新
    add_candidates, fix_candidates = _collect_recovery_candidates(
        ctx, signals.cnn_board, hsv_board, min_frames,
    )

    if not add_candidates and not fix_candidates:
        return

    # パス2 (方向1のみ): 列ごとに重力整合チェック (安全弁C)
    passed_add: list[tuple[int, int, int]] = []
    cols_with_add = {c for (_, c, _) in add_candidates}
    for col in cols_with_add:
        passed_add.extend(
            _check_recovery_column(ctx.confirmed_board, col, add_candidates),
        )

    # パス3: confirmed 更新 (方向1/2/3 合算)
    all_passed = passed_add + fix_candidates
    if not all_passed:
        return

    for r, c, target in all_passed:
        ctx.confirmed_board.set(r, c, target)
        ctx.recovery_cells.add((r, c))
        # カウンタリセット (発火済み)
        ctx.stable_recovery_counters.pop((r, c), None)

    # 方向1に対しのみ重力整合最終確認 (方向2/3は浮きぷよを生じさせない)
    if passed_add:
        _apply_gravity_filter(ctx.confirmed_board)

    # 振動抑制: non_stable_cnn_history に復旧後盤面を強制投入。
    # 次の NON-STABLE→STABLE 遷移の F ガード多数決で復旧色が票を持ち再崩壊防止。
    reinforce = ctx.confirmed_board.copy()
    ctx.non_stable_cnn_history.insert(0, reinforce)
    if len(ctx.non_stable_cnn_history) > DEFAULT_NON_STABLE_HISTORY_SIZE:
        ctx.non_stable_cnn_history.pop(-1)


# ============================
# 補助 detector (テスト + デバッグ用)
# ============================


class NullDetector:
    """常に None を返す detector (= state machine 単体テスト用)."""

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        return None


__all__ = [
    "BOARD_COLS",
    "BOARD_ROWS",
    "BoardState",
    "BoardStateMachine",
    "DEFAULT_STABLE_FRAME_COUNT",
    "DetectorSignals",
    "NON_STABLE_STATES",
    "NullDetector",
    "_apply_gravity_filter",
    "_apply_stable_recovery_gate",
    "_check_recovery_column",
    "_collect_recovery_candidates",
    "_merge_diff_only",
    "_vote_majority_board",
    "DEFAULT_NON_STABLE_HISTORY_SIZE",
    "DEFAULT_EMPTY_TO_COLOR_MIN_VOTES",
    "STABLE_WARMUP_FRAMES",
    "STABLE_RECOVERY_MIN_FRAMES",
    "RECOVERY_EXCLUDED_COLORS",
    "StateContext",
    "StateTransitionDetector",
    # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE 関連定数
    "GRAVITY_SETTLE_MIN_FRAMES",
    "GRAVITY_SETTLE_MAX_SEC",
    "GRAVITY_SETTLE_PHYSICS_CLEAR_MIN",
    "GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD",
]
