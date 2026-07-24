"""OjamaVisualDetector — 視覚ベースのお邪魔降下検知 (フェーズ A 精緻化).

連鎖終了直後のお邪魔降下を「可視最上段付近の ROI にお邪魔色が出現」する
視覚情報で確実に捉え、 infer_placement 誤配置 (幽霊) の根絶を目指す。

設計方針:
  - 一次判定: cnn_board の対象 ROI (HIDDEN_ROWS 〜 HIDDEN_ROWS+OJAMA_ROI_HEIGHT 行)
    に COLOR_OJAMA(9) が存在、 または hsv_board が非 None ならそこにも存在。
  - 発火条件: 一次 positive かつ (前フレームより増加 or 前が 0) かつ
    OJAMA_CONSEC_THRESH フレーム連続観測 → BoardState.OJAMA_FALL へ遷移。
  - 完了判定: OJAMA_FALL 中に ROI お邪魔 count==0 で即 STABLE 復帰。
    enable_ojama_settle_detection=True 時はさらに OJAMA_SETTLE_CONSEC フレーム
    不変でも STABLE 復帰 (お邪魔が静止した = 落下完了) 判定を行う。
  - stateless Protocol 準拠: 内部 state は dataclass field で管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.board import BOARD_COLS, COLOR_OJAMA, HIDDEN_ROWS, Board
from src.board_state_machine import (
    BoardState,
    DetectorSignals,
    GRAVITY_SETTLE_MAX_SEC,
    GRAVITY_SETTLE_MIN_FRAMES,
    GRAVITY_SETTLE_PHYSICS_CLEAR_MIN,
    GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD,
    StateContext,
)

# ============================
# 定数
# ============================

# お邪魔検知 ROI: 可視最上段 (HIDDEN_ROWS 行目) から何行を対象とするか。
# ぷよぷよ eスポーツではお邪魔は最上段から降ってくるため 2 行で十分。
OJAMA_ROI_HEIGHT: int = 2

# 発火に必要な連続観測フレーム数。1 フレームの CNN ぶれを除去する。
OJAMA_CONSEC_THRESH: int = 2

# settle 判定: OJAMA_FALL 中に ROI お邪魔 count が N フレーム不変なら
# お邪魔が静止完了と判断して STABLE 復帰する。
OJAMA_SETTLE_CONSEC: int = 3

# 案B (第2の根本原因対処, 2026-07-24):
# OJAMA_FALL 退出が早すぎ (滞在中央値 0.0167s = 1 frame) で着弾前に状態が抜け、
# empty_to_color 3 票ゲートが成立せず本物おじゃまが却下される問題への対処。
# GravitySettleDetector (src/state_detectors.py) と同じ「全盤面ぷよ数の静止」
# 判定方式を OJAMA_FALL 退出条件にも適用する。
# 定数は GravitySettle 側の値を初期値として流用するが、 今後 ojama 固有の
# チューニングが chain (GRAVITY_SETTLE) 側に波及しないよう独立命名する。
OJAMA_FALL_SETTLE_MIN_FRAMES: int = GRAVITY_SETTLE_PHYSICS_CLEAR_MIN
OJAMA_FALL_SETTLE_STABLE_FRAMES: int = GRAVITY_SETTLE_MIN_FRAMES
OJAMA_FALL_SETTLE_DIFF_THRESHOLD: int = GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD
OJAMA_FALL_MAX_SEC: float = GRAVITY_SETTLE_MAX_SEC

# ROI の開始行 (= HIDDEN_ROWS = 1 = 可視最上段)
_ROI_ROW_START: int = HIDDEN_ROWS
# ROI の終了行 (exclusive)
_ROI_ROW_END: int = HIDDEN_ROWS + OJAMA_ROI_HEIGHT


# ============================
# ヘルパー (static)
# ============================


def _count_top_ojama(
    cnn_board: Board,
    hsv_board: "Board | None" = None,
) -> int:
    """対象 ROI 内の COLOR_OJAMA セル数を返す (一次判定).

    cnn_board に OJAMA が存在するか、 または hsv_board 非 None かつそちらにも
    OJAMA が存在する場合に count を返す。
    両方 0 なら 0 を返し「お邪魔なし」 と判断する。

    Args:
        cnn_board: CNN 認識結果の盤面。
        hsv_board: HSV-only 認識盤面 (None なら cnn_board のみで判定)。

    Returns:
        ROI 内の OJAMA セル数 (0 以上の整数)。
    """
    count_cnn: int = 0
    for r in range(_ROI_ROW_START, _ROI_ROW_END):
        for c in range(BOARD_COLS):
            if int(cnn_board.get(r, c)) == COLOR_OJAMA:
                count_cnn += 1

    if hsv_board is None:
        return count_cnn

    count_hsv: int = 0
    for r in range(_ROI_ROW_START, _ROI_ROW_END):
        for c in range(BOARD_COLS):
            if int(hsv_board.get(r, c)) == COLOR_OJAMA:
                count_hsv += 1

    # CNN または HSV 一方でも検知した場合はその最大値を採用 (感度優先)
    return max(count_cnn, count_hsv)


# ============================
# OjamaVisualDetector
# ============================


@dataclass
class OjamaVisualDetector:
    """視覚ベースのお邪魔降下検知 detector.

    OjamaPhaseDetector (score 差分ベース) より前に登録し、
    連鎖終了直後のお邪魔降下を視覚で即座に捉える。

    フラグ (RecognitionPipeline 側で注入):
        enable_ojama_visual_chain_exit: True で CHAIN 中でも発火を許可。
        enable_ojama_settle_detection: True で settle (お邪魔静止) 判定を有効化。
        enable_ojama_fall_board_settle: 案B (2026-07-24)。True で OJAMA_FALL
            退出条件を「全盤面ぷよ数が静止するまで待つ」方式 (GravitySettle と
            同型) に切替える。default False = 既存挙動と完全 bit-identical。

    いずれも False のままなら既存挙動に近い動作となる (= STABLE/TSUMO_FALL 時の
    新規お邪魔出現のみ OJAMA_FALL に遷移)。
    """

    # 外部フラグ: RecognitionPipeline 側から __init__ 後に代入する。
    enable_ojama_visual_chain_exit: bool = False
    enable_ojama_settle_detection: bool = False
    # 案B (2026-07-24): OJAMA_FALL 退出を全盤面 settle 判定に置き換えるフラグ。
    # default False で `_detect_ojama_fall_exit` (既存ロジック) を完全維持する。
    enable_ojama_fall_board_settle: bool = False

    # 内部 state (dataclass field、 init=False)
    _consec_count: int = field(default=0, init=False, repr=False)
    _prev_top_ojama_count: int = field(default=0, init=False, repr=False)
    _last_frame_idx: int = field(default=-1, init=False, repr=False)
    _settle_count: int = field(default=0, init=False, repr=False)
    _prev_settle_count: int = field(default=0, init=False, repr=False)
    # 案B: 全盤面 settle 判定用の内部 state (GravitySettleDetector と同型)。
    _settle_start_frame: int = field(default=-1, init=False, repr=False)
    _settle_start_time: float = field(default=0.0, init=False, repr=False)
    _board_stable_consec: int = field(default=0, init=False, repr=False)
    _prev_board_puyo_count: int = field(default=-1, init=False, repr=False)

    def detect(
        self,
        ctx: StateContext,
        signals: DetectorSignals,
    ) -> BoardState | None:
        """state 遷移を返す (None = 遷移なし).

        OJAMA_FALL 中の完了判定と、 新規お邪魔出現による OJAMA_FALL 発火を担う。
        """
        cur_count = _count_top_ojama(
            signals.cnn_board, signals.hsv_board,
        )

        if ctx.state == BoardState.OJAMA_FALL:
            # 案B (2026-07-24): 全盤面 settle 判定に委譲 (default False で既存不変)。
            if self.enable_ojama_fall_board_settle:
                return self._detect_ojama_fall_exit_board_settle(ctx, signals)
            return self._detect_ojama_fall_exit(cur_count)

        return self._detect_ojama_fall_entry(ctx, cur_count)

    def _detect_ojama_fall_exit(
        self, cur_count: int,
    ) -> BoardState | None:
        """OJAMA_FALL 中の完了判定 (STABLE 復帰).

        ROI が空になれば即 STABLE 復帰。
        settle 判定 ON 時は cur_count が OJAMA_SETTLE_CONSEC フレーム不変でも復帰。
        """
        # お邪魔消滅: 即 STABLE 復帰
        if cur_count == 0:
            self._reset_internal_state()
            return BoardState.STABLE

        # settle 判定: count が N フレーム不変 → 落下完了
        if self.enable_ojama_settle_detection:
            if cur_count == self._prev_settle_count:
                self._settle_count += 1
            else:
                self._settle_count = 1
            self._prev_settle_count = cur_count
            if self._settle_count >= OJAMA_SETTLE_CONSEC:
                self._reset_internal_state()
                return BoardState.STABLE

        # OJAMA_FALL 継続中: 遷移なし
        return None

    def _detect_ojama_fall_exit_board_settle(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        """案B: 全盤面ぷよ数の静止を待つ OJAMA_FALL 退出判定.

        GravitySettleDetector (src/state_detectors.py) と同型のロジック。
        ROI (可視最上段 2 行) でなく盤面全体のぷよ数 (お邪魔含む) を見ることで、
        おじゃまが着弾しきる前に退出してしまう問題 (真因: 滞在中央値 1 frame) を防ぐ。

        条件 (いずれか):
          - タイムアウト: OJAMA_FALL_MAX_SEC 秒経過で安全弁として強制 STABLE。
          - OJAMA_FALL_SETTLE_MIN_FRAMES フレーム以上経過 かつ
            ぷよ数差分 < OJAMA_FALL_SETTLE_DIFF_THRESHOLD の状態が
            OJAMA_FALL_SETTLE_STABLE_FRAMES フレーム連続。
        """
        cur_board_count = signals.cnn_board.count_puyos()

        # OJAMA_FALL に初めて入ったフレームを記録
        if self._settle_start_frame < 0:
            self._settle_start_time = signals.time_sec
            self._settle_start_frame = ctx.frame_idx
            self._board_stable_consec = 0
            self._prev_board_puyo_count = cur_board_count
            return None  # 最初のフレームは必ず継続

        # タイムアウト: 安全弁として強制 STABLE 復帰
        elapsed = signals.time_sec - self._settle_start_time
        if elapsed >= OJAMA_FALL_MAX_SEC:
            # バグ修正 (2026-07-24): reset で _prev_top_ojama_count を無条件 0
            # にすると、 ROI にまだ残っているお邪魔 (着弾済み・不動) を次
            # フレームの _detect_ojama_fall_entry が「新規出現」と誤認し
            # 即座に OJAMA_FALL へ再突入する (振動ループ、 TSUMO_FALL 検出を
            # 最大 +7秒 遅延させる真因)。 退出時点の実カウントを保持する。
            exit_top_count = _count_top_ojama(signals.cnn_board, signals.hsv_board)
            self._reset_internal_state(keep_top_ojama_count=exit_top_count)
            return BoardState.STABLE

        # 最低待機フレーム数未達: 継続 (カウンタのみ更新)
        frames_in_settle = ctx.frame_idx - self._settle_start_frame
        if frames_in_settle < OJAMA_FALL_SETTLE_MIN_FRAMES:
            self._prev_board_puyo_count = cur_board_count
            self._board_stable_consec = 0
            return None

        # ぷよ数変化の安定性チェック
        diff = abs(cur_board_count - self._prev_board_puyo_count)
        self._prev_board_puyo_count = cur_board_count
        if diff < OJAMA_FALL_SETTLE_DIFF_THRESHOLD:
            self._board_stable_consec += 1
        else:
            self._board_stable_consec = 0

        if self._board_stable_consec >= OJAMA_FALL_SETTLE_STABLE_FRAMES:
            # バグ修正 (2026-07-24): タイムアウト分岐と同様、 settle 退出時も
            # ROI 実カウントを保持して次フレームの誤再突入を防ぐ。
            exit_top_count = _count_top_ojama(signals.cnn_board, signals.hsv_board)
            self._reset_internal_state(keep_top_ojama_count=exit_top_count)
            return BoardState.STABLE

        return None  # 継続

    def _detect_ojama_fall_entry(
        self, ctx: StateContext, cur_count: int,
    ) -> BoardState | None:
        """OJAMA_FALL 発火判定 (STABLE/TSUMO_FALL/CHAIN → OJAMA_FALL).

        CHAIN 中からの発火は enable_ojama_visual_chain_exit が True の時のみ。

        発火ロジック:
          - 「開始トリガー」: cur_count > 0 かつ (prev_count == 0 or cur_count > prev_count)
            = お邪魔が新規出現または増加したフレームでカウント開始。
          - 「継続」: 開始後 cur_count > 0 の間はカウントを継続する。
          - OJAMA_CONSEC_THRESH フレーム到達で OJAMA_FALL 発火。
          この設計により「出現 → 安定」の 2 フレームで確実に検知できる。

        防御コード (案B, 2026-07-24): CHAIN 等に途中横取りされて OJAMA_FALL から
        state が抜けると、 このメソッドが (exit メソッドではなく) 毎 frame
        呼ばれ続ける。 その間に settle 用内部 state をここで都度リセットしておかないと、
        次回 OJAMA_FALL 再突入時に前回の残骸 (_settle_start_frame 等) が残り、
        即 timeout 誤判定を起こす。
        """
        self._settle_start_frame = -1
        self._settle_start_time = 0.0
        self._board_stable_consec = 0
        self._prev_board_puyo_count = -1

        # CHAIN 中かつフラグ OFF → 発火しない
        if (
            ctx.state == BoardState.CHAIN
            and not self.enable_ojama_visual_chain_exit
        ):
            self._reset_internal_state()
            return None

        # MENU/EFFECT: 発火しない
        if ctx.state in (BoardState.MENU, BoardState.EFFECT):
            self._reset_internal_state()
            return None

        prev_count = self._prev_top_ojama_count
        self._prev_top_ojama_count = cur_count

        if cur_count == 0:
            # お邪魔なし: カウントリセット
            self._consec_count = 0
            return None

        # 開始トリガー: 前フレームが 0 またはお邪魔が増加した時にカウントを (再) 開始
        is_new_trigger = prev_count == 0 or cur_count > prev_count

        if self._consec_count == 0 and not is_new_trigger:
            # まだ開始トリガーが来ていない: カウントしない
            return None

        # 連続観測カウント (同 frame 二重カウント防止)
        if ctx.frame_idx != self._last_frame_idx:
            if is_new_trigger or self._consec_count > 0:
                self._consec_count += 1
            self._last_frame_idx = ctx.frame_idx

        if self._consec_count >= OJAMA_CONSEC_THRESH:
            self._consec_count = 0
            return BoardState.OJAMA_FALL

        return None

    def _reset_internal_state(
        self, keep_top_ojama_count: int | None = None,
    ) -> None:
        """内部カウンタを全リセットする.

        Args:
            keep_top_ojama_count: None (既定) なら従来通り
                `_prev_top_ojama_count` を 0 にリセットする (bit-identical)。
                int を渡すとその値を `_prev_top_ojama_count` に保持する。
                案B (`_detect_ojama_fall_exit_board_settle`) の退出時のみ、
                退出時点の ROI 実カウントを渡すことで、 まだ ROI に残る
                (着弾済み・不動の) お邪魔を次フレームが新規出現と誤認して
                OJAMA_FALL へ即再突入する振動バグを防ぐ。
        """
        self._consec_count = 0
        self._prev_top_ojama_count = (
            0 if keep_top_ojama_count is None else keep_top_ojama_count
        )
        self._last_frame_idx = -1
        self._settle_count = 0
        self._prev_settle_count = 0
        # 案B: 全盤面 settle 判定用の内部 state もリセット。
        self._settle_start_frame = -1
        self._settle_start_time = 0.0
        self._board_stable_consec = 0
        self._prev_board_puyo_count = -1
