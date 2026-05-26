"""state 遷移検出器 (Phase B-2).

各検出器は `BoardStateMachine` の `StateTransitionDetector` Protocol を
満たす形で実装される。BoardStateMachine に登録された順で評価され、最初に
None 以外の `BoardState` を返した detector の値が採用される。

推奨優先順位 (登録順):
    1. ChainPhaseDetector  — 連鎖は最も明確なシグナル
    2. EffectPhaseDetector — 全消し演出は連鎖直後に発生
    3. OjamaPhaseDetector  — おじゃま落下は相手連鎖完了後
    4. TsumoPhaseDetector  — 上記すべてに該当しない puyo 増加は通常ツモ

各 detector は内部状態を持つ場合、frame 列を時系列で受け取る前提で
動作する (= 同じ frame_idx で複数回 detect を呼ばれてはいけない)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.board import Board
from src.board_state_machine import (
    BoardState,
    DetectorSignals,
    StateContext,
)


# ============================
# Chain phase detector
# ============================


@dataclass
class ChainPhaseDetector:
    """連鎖発火イベントから CHAIN state を判定する.

    入力:
        signals.chain_event: pipeline 側で保持された有効期限内 ChainEvent
                            (None なら連鎖中ではない)。
    出力:
        - signals.chain_event != None → CHAIN
        - signals.chain_event == None かつ 現 state == CHAIN → STABLE 復帰
        - それ以外 → None (state 維持)

    時刻範囲の制御は RecognitionPipeline 側で行う (CHAIN_HOLD_PER_STEP_SEC ×
    chain_count 秒間 event を保持し、過ぎたら None に戻す)。本 detector は
    シンプルに event 存在の有無のみを見る。
    """

    chain_sim: object | None = None  # cycle 49: optional ChainSimulator 注入

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        if signals.chain_event is not None:
            # cycle 49 (2026-05-20): 4 連結必須化 gate
            # 強化アナリスト retrospective_chain_missing / chain_no_puyo_loss が
            # 全動画で 12-30 件検出 = chain 誤判定が普遍的問題。
            # 前 STABLE 盤面に 4 連結 (= erasable) が無ければ chain 遷移を拒否。
            # ただし: ctx.confirmed_board が None or UNKNOWN だらけの認識不確実時は
            # 真の連鎖を見逃さないため遷移許容 (= fail-silent 防止)。
            if self.chain_sim is not None and ctx.confirmed_board is not None:
                from src.board import COLOR_UNKNOWN as _CU
                grid = ctx.confirmed_board._grid
                # UNKNOWN cell が 3 個以上ある場合は認識不確実 → gate skip
                unknown_count = int((grid == _CU).sum())
                if unknown_count < 3:
                    erasable = self.chain_sim.find_erasable_groups(
                        ctx.confirmed_board,
                    )
                    if not erasable:
                        # 4 連結なし = 偽 chain event を無視
                        return None  # state 維持
            return BoardState.CHAIN
        if ctx.state == BoardState.CHAIN:
            return BoardState.STABLE
        return None


# ============================
# Tsumo (ツモ落下) detector
# ============================


@dataclass
class TsumoPhaseDetector:
    """直近 STABLE 盤面との puyo 数差分でツモ落下を判定する.

    Logic:
        - 直近 STABLE 盤面より +1〜+max_increase 個増えた状態が連続
          consec_threshold frame 観測されたら TSUMO_FALL
          (1 frame だけの CNN ぶれは無視する設計、Phase B-7 で追加)
        - **着地検出 (Phase B-20)**: 現 state==TSUMO_FALL 中に CNN 盤面が
          landed_consec frame 連続で同一なら「着地完了 = ツモが静止した」
          と判断して STABLE 復帰
          (旧実装の「diff==0 で復帰」は連鎖発火後しか成立せず、通常着地で
           +2 puyo のまま TSUMO_FALL ロックインするバグだった)
        - 増加が 0 で現 state == TSUMO_FALL → STABLE (連鎖発火による消去)
        - 増加が max_increase より多い (= 連鎖や異常) → None (CHAIN detector
          に任せる)
        - **Phase I R-7 強化**: signals.slide_motion=True (= NEXT ROI で
          ツモのスライドを検出) なら、着地連続確認を待たずに STABLE 復帰。
          TSUMO_FALL → STABLE 遷移の検出漏れ (= 「9 秒問題」: 連続 2 手の
          うち 2 手目が STABLE 確定されないバグ) を補強する。
        - **Phase I R-1 強化**: signals.placement_validated=True (= cnn_board
          と baseline の色 count delta が落下ペアと整合) なら、着地連続
          確認 (landed_consec) を 1 frame 早めて STABLE 復帰の早期化を行う。

    注意: 直近 STABLE 盤面は `ctx.confirmed_board` を参照する。
    BoardStateMachine が CHAIN/OJAMA_FALL/EFFECT 中は confirmed_board を
    更新しない仕様なので、それらが終わるまで baseline は固定される。
    """

    min_increase: int = 1
    max_increase: int = 2
    consec_threshold: int = 2  # CNN ぶれ吸収用、連続観測 frame 数
    landed_consec: int = 2  # 着地確定: 同一盤面 N 連続で STABLE 復帰

    # 内部 state (dataclass field、init から除外)
    _consec_count: int = field(default=0, init=False, repr=False)
    _last_frame_idx: int = field(default=-1, init=False, repr=False)
    _landed_consec_count: int = field(default=0, init=False, repr=False)
    _last_cnn_board_for_landing: object = field(
        default=None, init=False, repr=False,
    )

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        baseline = ctx.confirmed_board
        if baseline is None:
            return None  # 初回 STABLE 確定前は判定不能
        baseline_count = baseline.count_puyos()
        cur_count = signals.cnn_board.count_puyos()
        diff = cur_count - baseline_count

        # ------------------------------
        # R-7: NEXT slide motion による STABLE 強制復帰 (TSUMO_FALL 中のみ)
        # ------------------------------
        # ツモが画面から消えて次のツモが繰り上がった signal は「手が置かれた」
        # 強い証拠。puyo count delta が +1〜+2 範囲内に落ち着いていれば
        # 連続観測 (landed_consec) を待たず即時 STABLE に遷移する。
        # diff>2 (= 連鎖継続中) は CHAIN detector に任せるためスキップ。
        if (
            ctx.state == BoardState.TSUMO_FALL
            and signals.slide_motion
            and self.min_increase <= diff <= self.max_increase
        ):
            self._consec_count = 0
            self._landed_consec_count = 0
            self._last_cnn_board_for_landing = None
            return BoardState.STABLE

        # 着地検出: TSUMO_FALL 中に CNN 盤面が連続同一 → 着地完了
        # diff の符号によらず判定する (puyo 数が +1 でも +2 でも安定すれば着地)
        if ctx.state == BoardState.TSUMO_FALL:
            prev = self._last_cnn_board_for_landing
            same = (
                prev is not None
                and signals.cnn_board == prev
            )
            if same:
                self._landed_consec_count += 1
            else:
                self._landed_consec_count = 1
            self._last_cnn_board_for_landing = signals.cnn_board.copy()
            # R-1: placement_validated なら早期復帰 (landed_consec を 1 緩和)
            effective_landed = (
                max(1, self.landed_consec - 1)
                if signals.placement_validated
                else self.landed_consec
            )
            # cycle 71v (2026-05-15): diff >= min_increase ガードを追加.
            # 旧実装は cnn_board が連続同一なら diff=0 でも STABLE 復帰し、
            # CNN がツモを観測してない frame でも誤発火 → infer_placement が
            # cnn=空のまま arbitrary column commit (= ゴースト) する原因だった。
            # 実際に新規 puyo cells が見えている時のみ STABLE 復帰.
            if (
                self._landed_consec_count >= effective_landed
                and diff >= self.min_increase
            ):
                # 着地確定: STABLE 復帰
                self._consec_count = 0
                self._landed_consec_count = 0
                return BoardState.STABLE
        else:
            # TSUMO_FALL 以外なら着地カウンタリセット
            self._landed_consec_count = 0
            self._last_cnn_board_for_landing = None

        if self.min_increase <= diff <= self.max_increase:
            # 同 frame で複数回呼ばれた場合は重複カウントしない
            if ctx.frame_idx != self._last_frame_idx:
                self._consec_count += 1
                self._last_frame_idx = ctx.frame_idx
            if self._consec_count >= self.consec_threshold:
                return BoardState.TSUMO_FALL
            return None  # 連続性未達、まだ TSUMO 判定しない
        # 増加が範囲外: 連続カウンタリセット
        self._consec_count = 0
        # cycle 71v (2026-05-15): 連鎖発火による puyo 減少のみで STABLE 復帰.
        # 旧実装は diff == 0 で STABLE 復帰していたが、 これは CNN が落下中ツモを
        # 観測できていない frame でも誤発火し、 結果 cnn_after=空で infer_placement が
        # 走って arbitrary column の placement (= ゴースト) を confirmed に commit する
        # 原因だった。 1-chain の典型は +2 placed → 4 erased = diff=-2 で 0 にはならない。
        # diff < 0 (= 実際に puyo 減少) を chain 完了 signal として使う。
        if diff < 0 and ctx.state == BoardState.TSUMO_FALL:
            return BoardState.STABLE
        return None


# ============================
# Ojama (おじゃま落下) detector — score OCR は B-4
# ============================


@dataclass
class OjamaPhaseDetector:
    """score 差分から OJAMA_FALL を判定する.

    入力:
        signals.score_delta: 直近 frame の **相手** の score 増分。
                             RecognitionPipeline 側で 1P signal には
                             2P の score_delta が渡される。
    Logic:
        - score_delta が threshold 以上 → 相手連鎖完了 → 自分側に
          おじゃまが降る → OJAMA_FALL
        - 現 state == OJAMA_FALL かつ score_delta < threshold → STABLE 復帰
          (= 落下が止まった後の平常時へ)

    threshold は OJAMA_RATE_STANDARD (=70 点 = おじゃま 1 個分) を採用。
    これより小さい score 変動はおじゃまに化けないので OJAMA_FALL 不要。
    """

    score_threshold: int = 70

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        if signals.score_delta >= self.score_threshold:
            return BoardState.OJAMA_FALL
        # OJAMA_FALL に居て新たな score 増加が無くなったら STABLE へ復帰。
        # ChainPhase/Tsumo と同じ責任分界: 「自分が発火させた state は
        # 自分で抜ける」ロジックで state machine がロックインしないように
        # する。本格的な「おじゃま落下完了」検出は B-3/B-4 統合で実装。
        if ctx.state == BoardState.OJAMA_FALL:
            return BoardState.STABLE
        return None


# ============================
# Effect (全消し演出 / カットイン) detector — skeleton
# ============================


@dataclass
class EffectPhaseDetector:
    """演出フレーム検出 (全消し / カットイン / テロップ / 試合終了演出).

    2026-05-10 実装: signals.effect_visible (telop/match_end/all_clear/
    win_panel/chain_animation の OR) で EFFECT state に遷移。
    演出中は CNN 出力を信用せず直前 STABLE 盤面を hold する
    (memory `feedback_chain_phase_physics_only`)。
    """

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        if signals.effect_visible:
            return BoardState.EFFECT
        return None


__all__ = [
    "ChainPhaseDetector",
    "EffectPhaseDetector",
    "OjamaPhaseDetector",
    "TsumoPhaseDetector",
]
