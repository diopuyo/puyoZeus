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

    いずれも False のままなら既存挙動に近い動作となる (= STABLE/TSUMO_FALL 時の
    新規お邪魔出現のみ OJAMA_FALL に遷移)。
    """

    # 外部フラグ: RecognitionPipeline 側から __init__ 後に代入する。
    enable_ojama_visual_chain_exit: bool = False
    enable_ojama_settle_detection: bool = False

    # 内部 state (dataclass field、 init=False)
    _consec_count: int = field(default=0, init=False, repr=False)
    _prev_top_ojama_count: int = field(default=0, init=False, repr=False)
    _last_frame_idx: int = field(default=-1, init=False, repr=False)
    _settle_count: int = field(default=0, init=False, repr=False)
    _prev_settle_count: int = field(default=0, init=False, repr=False)

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
        """
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

    def _reset_internal_state(self) -> None:
        """内部カウンタを全リセットする."""
        self._consec_count = 0
        self._prev_top_ojama_count = 0
        self._last_frame_idx = -1
        self._settle_count = 0
        self._prev_settle_count = 0
