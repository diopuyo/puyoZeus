"""
得点ベース予告お邪魔ぷよ推論モジュール

原理:
    視覚で予告お邪魔ぷよ (1P/2P 上部のサイン) を読み取らずとも、
    相手側の連鎖発火イベントから公式得点式で予告個数を計算できる。

        相手側で連鎖発火 → calculate_chain_score() → score_to_ojama() → 予告個数

    本モジュールは src/scoring.py を改造せずに、その関数を組み合わせて
    「試合全体の連鎖イベント時系列」から「両側の予告お邪魔の時系列」を
    導出するためのラッパー。

設計上の注意:
    - score_to_ojama の prev_leftover は「送り手側の前回得点端数」を意味する。
      つまり 1P が連鎖発火 → 1P 側の leftover を消費 / 更新し、
      その結果を 2P に「予告」として送る。
    - leftover はあくまで送り手側の状態として連続管理する。
    - 全消し持ち越しボーナス (ALL_CLEAR_BONUS=2100) は src/chain_detector.py
      と同じく「次回発火時に加算 → そのときの is_all_clear で再フラグ更新」
      とする。フラグも送り手側で別個に管理する。
    - マージンタイムは fired_at - match_start で elapsed を作り、
      compute_effective_rate / score_to_ojama に渡す。

使い方:
    from src.ojama_score_inferrer import OjamaScoreInferrer
    from src.chain import ChainSimulator

    sim = ChainSimulator()
    inferrer = OjamaScoreInferrer()

    # 単発推論
    result = sim.simulate(board)
    pred, lo1, lo2 = inferrer.infer_from_chain_event(
        result, fired_by="1P", match_elapsed_sec=80.0,
        prev_leftover_1p=0, prev_leftover_2p=0,
    )

    # タイムライン推論
    events = [(60.0, "1P", sim.simulate(board1), False),
              (75.0, "2P", sim.simulate(board2), False)]
    preds = inferrer.infer_timeline(events, match_start_sec=0.0)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.chain import ChainResult
from src.scoring import (
    ALL_CLEAR_BONUS,
    OJAMA_RATE_STANDARD,
    calculate_chain_score,
    calculate_step_score,
    score_to_ojama,
)

# プレイヤー側を表す文字列定数
SIDE_1P: str = "1P"
SIDE_2P: str = "2P"
VALID_SIDES: frozenset[str] = frozenset({SIDE_1P, SIDE_2P})


def _opponent(side: str) -> str:
    """発火側の反対側を返す。"""
    if side == SIDE_1P:
        return SIDE_2P
    if side == SIDE_2P:
        return SIDE_1P
    raise ValueError(f"不正な side: {side!r} (期待: '1P' or '2P')")


@dataclass(frozen=True)
class OjamaPrediction:
    """
    1 件の連鎖発火イベントから推定した予告お邪魔ぷよ予測。

    Attributes:
        side: 予告お邪魔を「受ける」側 ("1P" or "2P")。
        pending: 予告お邪魔ぷよ数 (今回追加分)。
        chain_length: 発火連鎖数。
        fired_at_sec: 発火時刻 [秒]。
        fired_by_side: 発火した側 ("1P" or "2P")。
        total_score: 加算済みのスコア (全消し持越し含む)。
        base_score: 持越し抜きの素点。
        all_clear_bonus_applied: 今回加算された全消し持越しボーナス。
        is_all_clear: 連鎖後盤面が全消しか (送り手側次回への持越しフラグ)。
        effective_rate: マージンタイム適用後の有効レート。
    """
    side: str
    pending: int
    chain_length: int
    fired_at_sec: float
    fired_by_side: str
    total_score: int
    base_score: int
    all_clear_bonus_applied: int
    is_all_clear: bool
    effective_rate: int


class OjamaScoreInferrer:
    """
    連鎖イベントから予告お邪魔ぷよ数を推論するクラス。

    内部状態として両側の leftover と全消し持越しフラグを保持する。
    無状態に使いたい場合は infer_from_chain_event に明示的に
    prev_leftover_1p / prev_leftover_2p を渡すこと。
    """

    def __init__(
        self,
        rate: int = OJAMA_RATE_STANDARD,
        margin_start_sec: float = 96.0,
    ) -> None:
        """
        Args:
            rate: おじゃま基準レート (通常 70)。
            margin_start_sec: マージンタイム開始秒 (現状は記録のみ。
                実際の減衰は src.scoring.compute_effective_rate が
                MARGIN_TIME_START_SEC を見るため、ここでは互換用)。
        """
        self._rate = rate
        self._margin_start_sec = margin_start_sec
        # タイムライン計算用の内部状態
        self._leftover_1p: int = 0
        self._leftover_2p: int = 0
        self._ac_pending_1p: bool = False
        self._ac_pending_2p: bool = False

    # ============================
    # 公開: 単発推論
    # ============================

    def infer_from_chain_event(
        self,
        chain_result: ChainResult,
        fired_by: str,
        match_elapsed_sec: float,
        prev_leftover_1p: int,
        prev_leftover_2p: int,
        all_clear_pending_1p: bool = False,
        all_clear_pending_2p: bool = False,
    ) -> tuple[OjamaPrediction, int, int]:
        """
        1 件の連鎖発火イベントから相手側への予告お邪魔ぷよ数を計算する。

        Args:
            chain_result: ChainSimulator().simulate() の結果。
            fired_by: 連鎖を発火した側 ("1P" or "2P")。
            match_elapsed_sec: 試合開始からの経過秒 (マージンタイム計算用)。
            prev_leftover_1p: 1P 側のこれまでの leftover。
            prev_leftover_2p: 2P 側のこれまでの leftover。
            all_clear_pending_1p: 1P 側に全消し持越しが残っているか。
            all_clear_pending_2p: 2P 側に全消し持越しが残っているか。

        Returns:
            (OjamaPrediction, 新 leftover_1p, 新 leftover_2p)
            送り手側の leftover のみが更新される。受け手側はそのまま。

        Raises:
            ValueError: fired_by が "1P"/"2P" 以外の場合。
        """
        if fired_by not in VALID_SIDES:
            raise ValueError(
                f"fired_by は '1P' or '2P' を指定してください: {fired_by!r}"
            )

        # 素点計算 (全消しボーナスは含まない)
        scored = calculate_chain_score(chain_result)

        # 発火側の全消し持越しを今回に加算
        if fired_by == SIDE_1P:
            ac_bonus = ALL_CLEAR_BONUS if all_clear_pending_1p else 0
            sender_leftover = prev_leftover_1p
        else:
            ac_bonus = ALL_CLEAR_BONUS if all_clear_pending_2p else 0
            sender_leftover = prev_leftover_2p
        effective_score = scored.total_score + ac_bonus

        elapsed = max(0.0, float(match_elapsed_sec))
        ojama_r = score_to_ojama(
            effective_score,
            prev_leftover=sender_leftover,
            elapsed_sec=elapsed,
            rate_base=self._rate,
        )

        # 送り手側のみ leftover を更新する
        new_leftover_1p = prev_leftover_1p
        new_leftover_2p = prev_leftover_2p
        if fired_by == SIDE_1P:
            new_leftover_1p = ojama_r.leftover_score
        else:
            new_leftover_2p = ojama_r.leftover_score

        prediction = OjamaPrediction(
            side=_opponent(fired_by),
            pending=ojama_r.ojama_count,
            chain_length=chain_result.chain_count,
            fired_at_sec=elapsed,
            fired_by_side=fired_by,
            total_score=effective_score,
            base_score=scored.total_score,
            all_clear_bonus_applied=ac_bonus,
            is_all_clear=scored.is_all_clear,
            effective_rate=ojama_r.effective_rate,
        )
        return prediction, new_leftover_1p, new_leftover_2p

    # ============================
    # 公開: ステップ別 ojama 内訳 (Phase R)
    # ============================

    def infer_per_step_breakdown(
        self,
        chain_result: ChainResult,
        fired_by: str,
        match_elapsed_sec: float,
        prev_leftover_sender: int = 0,
    ) -> list[dict]:
        """連鎖の各 step ごとに「N × M」の内訳と累積 ojama を返す。

        ユーザ仕様:
            - 画面表示の "N x M" は N=erased_count×10、M=bonus_multiplier
            - step_score = N × M
            - 累積 score → score_to_ojama で各 step 終了時点の ojama 個数

        Returns:
            各 step の dict リスト:
                step_idx (1-indexed), erased_count, n_display (=erased*10),
                m_display (=bonus_multiplier), step_score,
                cumulative_score, cumulative_ojama, leftover, effective_rate
        """
        breakdown: list[dict] = []
        cumulative_score = 0
        leftover = int(prev_leftover_sender)
        elapsed = max(0.0, float(match_elapsed_sec))
        for step in chain_result.steps:
            ssr = calculate_step_score(step)
            cumulative_score += ssr.score
            ojama_r = score_to_ojama(
                cumulative_score,
                prev_leftover=int(prev_leftover_sender),
                elapsed_sec=elapsed,
                rate_base=self._rate,
            )
            breakdown.append({
                "step_idx": ssr.chain_idx,
                "erased_count": ssr.erased_count,
                "n_display": ssr.erased_count * 10,
                "m_display": ssr.bonus_multiplier,
                "step_score": ssr.score,
                "cumulative_score": cumulative_score,
                "cumulative_ojama": ojama_r.ojama_count,
                "leftover": ojama_r.leftover_score,
                "effective_rate": ojama_r.effective_rate,
            })
        return breakdown

    # ============================
    # 公開: タイムライン推論
    # ============================

    def infer_timeline(
        self,
        chain_events: list[tuple[float, str, ChainResult, bool]],
        match_start_sec: float = 0.0,
    ) -> list[OjamaPrediction]:
        """
        試合全体の連鎖イベント時系列から予告お邪魔の時系列を生成する。

        Args:
            chain_events: (fired_at_sec, fired_by, chain_result, all_clear)
                のリスト。fired_at_sec は単調増加前提。
                all_clear はそのイベントの「次回への持越しフラグ」を上書きで
                指定する場合に True を渡すが、通常は False を渡し、
                ChainResult から自動算出された値を採用する。
            match_start_sec: 試合開始秒 (マージンタイム計算用)。

        Returns:
            list[OjamaPrediction]: 各イベントに対応する予測。

        Note:
            内部状態 (leftover, all_clear_pending) はメソッド冒頭でリセット
            されるため、複数試合を続けて呼び出す場合は試合ごとに本メソッドを
            呼ぶこと。
        """
        # 試合冒頭で内部状態をリセット
        self._leftover_1p = 0
        self._leftover_2p = 0
        self._ac_pending_1p = False
        self._ac_pending_2p = False

        predictions: list[OjamaPrediction] = []
        for fired_at, fired_by, chain_result, all_clear_override in chain_events:
            elapsed = max(0.0, float(fired_at) - float(match_start_sec))
            pred, lo1, lo2 = self.infer_from_chain_event(
                chain_result=chain_result,
                fired_by=fired_by,
                match_elapsed_sec=elapsed,
                prev_leftover_1p=self._leftover_1p,
                prev_leftover_2p=self._leftover_2p,
                all_clear_pending_1p=self._ac_pending_1p,
                all_clear_pending_2p=self._ac_pending_2p,
            )
            self._leftover_1p = lo1
            self._leftover_2p = lo2

            # 送り手側の全消し持越しフラグを更新
            # all_clear_override が True の場合は強制的に True にする
            new_ac = bool(pred.is_all_clear or all_clear_override)
            if fired_by == SIDE_1P:
                self._ac_pending_1p = new_ac
            else:
                self._ac_pending_2p = new_ac

            predictions.append(pred)

        return predictions

    # ============================
    # 公開: score 差分ベース推論 (落下ボーナス・全消し込みの実観測 score)
    # ============================

    def infer_from_score_delta(
        self,
        score_before: int,
        score_after: int,
        fired_by: str,
        match_elapsed_sec: float,
        prev_leftover_sender: int = 0,
        chain_length: int = 0,
        is_all_clear: bool = False,
        all_clear_bonus_applied: int = 0,
    ) -> tuple[OjamaPrediction, int]:
        """
        連鎖前後の生 score 差分から ojama を推論する。

        OCR で読んだ実観測値の差分には既に「落下ボーナス・全消し持越し
        ボーナス」が含まれているため、本メソッドでは追加計算しない。

        Args:
            score_before: 連鎖発火前の score。
            score_after: 連鎖発火後 (盤面静止後) の score。
            fired_by: "1P" or "2P"。
            match_elapsed_sec: 試合開始からの経過秒。
            prev_leftover_sender: 送り手側の前回 leftover。
            chain_length: 既知なら連鎖数を渡す (記録用、推論には使わない)。
            is_all_clear: 連鎖後の全消し有無 (記録用)。
            all_clear_bonus_applied: 今回 score 差分に含まれた持越しボーナス分
                (差分から base_score を分離するための情報、推論には使わない)。

        Returns:
            (OjamaPrediction, 送り手側の新 leftover)
        """
        if fired_by not in VALID_SIDES:
            raise ValueError(
                f"fired_by は '1P' or '2P' を指定してください: {fired_by!r}"
            )
        delta = max(0, int(score_after) - int(score_before))
        elapsed = max(0.0, float(match_elapsed_sec))
        ojama_r = score_to_ojama(
            delta,
            prev_leftover=int(prev_leftover_sender),
            elapsed_sec=elapsed,
            rate_base=self._rate,
        )
        base = max(0, delta - int(all_clear_bonus_applied))
        prediction = OjamaPrediction(
            side=_opponent(fired_by),
            pending=ojama_r.ojama_count,
            chain_length=int(chain_length),
            fired_at_sec=elapsed,
            fired_by_side=fired_by,
            total_score=delta,
            base_score=base,
            all_clear_bonus_applied=int(all_clear_bonus_applied),
            is_all_clear=bool(is_all_clear),
            effective_rate=ojama_r.effective_rate,
        )
        return prediction, ojama_r.leftover_score

    def infer_timeline_from_score_series(
        self,
        score_series: list[tuple[float, int, int]],
        match_start_sec: float = 0.0,
        min_chain_score: int = 40,
    ) -> list[OjamaPrediction]:
        """
        連続 score 観測列から、score の急増を連鎖イベントとして抽出し、
        両側それぞれの差分から ojama 推論を時系列出力する。

        Args:
            score_series: (t_sec, score_1p, score_2p) のリスト。t は単調増加
                前提。OCR 失敗等で score が読めなかった点は事前に除外する。
            match_start_sec: 試合開始秒 (マージンタイム計算用)。
            min_chain_score: この未満の差分は連鎖イベントとみなさず無視する。
                数桁の OCR ノイズ吸収用。

        Returns:
            list[OjamaPrediction]: 各連鎖イベントの予測。
            内部状態 (leftover) はサイドごとに連続管理する。
        """
        self.reset()
        predictions: list[OjamaPrediction] = []
        if len(score_series) < 2:
            return predictions
        _, prev_1p, prev_2p = score_series[0]
        for cur_t, cur_1p, cur_2p in score_series[1:]:
            elapsed = max(0.0, float(cur_t) - float(match_start_sec))
            d1 = int(cur_1p) - int(prev_1p)
            d2 = int(cur_2p) - int(prev_2p)
            if d1 >= int(min_chain_score):
                pred, lo = self.infer_from_score_delta(
                    score_before=prev_1p, score_after=cur_1p,
                    fired_by=SIDE_1P, match_elapsed_sec=elapsed,
                    prev_leftover_sender=self._leftover_1p,
                )
                self._leftover_1p = lo
                predictions.append(pred)
            if d2 >= int(min_chain_score):
                pred, lo = self.infer_from_score_delta(
                    score_before=prev_2p, score_after=cur_2p,
                    fired_by=SIDE_2P, match_elapsed_sec=elapsed,
                    prev_leftover_sender=self._leftover_2p,
                )
                self._leftover_2p = lo
                predictions.append(pred)
            prev_1p, prev_2p = cur_1p, cur_2p
        return predictions

    # ============================
    # 状態アクセサ
    # ============================

    @property
    def leftover_1p(self) -> int:
        """1P 側の現在の leftover (タイムライン推論後に有効)。"""
        return self._leftover_1p

    @property
    def leftover_2p(self) -> int:
        """2P 側の現在の leftover (タイムライン推論後に有効)。"""
        return self._leftover_2p

    @property
    def all_clear_pending_1p(self) -> bool:
        """1P 側の全消し持越しフラグ。"""
        return self._ac_pending_1p

    @property
    def all_clear_pending_2p(self) -> bool:
        """2P 側の全消し持越しフラグ。"""
        return self._ac_pending_2p

    def reset(self) -> None:
        """内部状態 (leftover/全消し持越し) を初期化する。"""
        self._leftover_1p = 0
        self._leftover_2p = 0
        self._ac_pending_1p = False
        self._ac_pending_2p = False
