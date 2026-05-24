"""
統合エンドポイントモジュール

フレーム or 盤面ペアを入力として、
  画像認識 → 連鎖シミュレーション → 8指標計算 → 総合スコア算出
までを一気通貫で実行する。

外部モジュール (CLI / overlay / 配信) はこの analyzer を起点とする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.board import Board
from src.chain import ChainResult, ChainSimulator
from src.image_reader import ImageReader
from src.indicators import IndicatorCalculator, IndicatorSet
from src.scorer import ScoreResult, Scorer


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class PlayerAnalysis:
    """
    1プレイヤー分の分析結果。

    Attributes:
        board: 盤面データ。
        chain_result: 連鎖シミュレーション結果。
        indicators: 8指標セット。
    """
    board: Board
    chain_result: ChainResult
    indicators: IndicatorSet

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存可能な辞書に変換する (board のみ深シリアライズ)。"""
        return {
            "board": self.board.to_dict(),
            "chain_count": self.chain_result.chain_count,
            "total_erased": self.chain_result.total_erased,
            "indicators": self.indicators.to_dict(),
        }


@dataclass(frozen=True)
class AnalysisResult:
    """
    1フレーム分の完全な分析結果。

    Attributes:
        player1: 1P の分析結果。
        player2: 2P の分析結果。
        score: 総合スコア (-100〜+100)。
        timestamp: フレームのタイムスタンプ (秒)。None で省略可。
    """
    player1: PlayerAnalysis
    player2: PlayerAnalysis
    score: ScoreResult
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存可能な辞書に変換する。"""
        d: dict[str, Any] = {
            "player1": self.player1.to_dict(),
            "player2": self.player2.to_dict(),
            "score": self.score.to_dict(),
        }
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        return d


# ============================
# Analyzer
# ============================


class Analyzer:
    """
    フレーム/盤面から完全な分析結果を生成する統合クラス。

    各下位モジュール (ImageReader, ChainSimulator, IndicatorCalculator, Scorer)
    は差し替え可能。デフォルト構成で ready-to-use。

    Usage:
        analyzer = Analyzer()
        result = analyzer.analyze_frame(frame_bgr)
        print(result.score.total_score)
    """

    def __init__(
        self,
        image_reader: ImageReader | None = None,
        simulator: ChainSimulator | None = None,
        calculator: IndicatorCalculator | None = None,
        scorer: Scorer | None = None,
    ) -> None:
        """
        Args:
            image_reader: 画像→盤面変換器 (None ならデフォルト)。
            simulator: 連鎖シミュレータ (None ならデフォルト)。
            calculator: 指標計算器 (None ならデフォルト)。
            scorer: スコアラー (None ならデフォルト)。
        """
        self._reader = image_reader or ImageReader()
        self._simulator = simulator or ChainSimulator()
        self._calculator = calculator or IndicatorCalculator(
            simulator=self._simulator,
        )
        self._scorer = scorer or Scorer()

    # ============================
    # 公開メソッド
    # ============================

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> AnalysisResult:
        """
        BGR フレーム画像を解析して完全な分析結果を返す。

        Args:
            frame: BGR 形式のフレーム画像 (H×W×3)。
            timestamp: フレームのタイムスタンプ (秒、任意)。

        Returns:
            AnalysisResult: 両プレイヤーの分析結果と総合スコア。
        """
        board_1p, board_2p = self._reader.read_both_boards(frame)
        return self.analyze_boards(board_1p, board_2p, timestamp=timestamp)

    def analyze_boards(
        self,
        board_1p: Board,
        board_2p: Board,
        timestamp: float | None = None,
        next_pair_1p: tuple[int, int] | None = None,
        dnext_pair_1p: tuple[int, int] | None = None,
        next_pair_2p: tuple[int, int] | None = None,
        dnext_pair_2p: tuple[int, int] | None = None,
        incoming_ojama_1p: int = 0,
        incoming_ojama_2p: int = 0,
    ) -> AnalysisResult:
        """
        盤面ペアを解析する (画像認識を省略したい場合に使用)。

        2026-04-27: opponent_board / incoming_ojama / next_pair を Phase J 指標
        (opponent_chain_threat / incoming_ojama_pressure / next_acceptance) に
        正しく供給するため引数を拡張。これらが None / 0 の場合は中立値となり、
        強い負係数の重み (-1.02 / -0.52) が事実上無効化されるため運用時は必ず指定。

        Args:
            board_1p: 1P の盤面。
            board_2p: 2P の盤面。
            timestamp: フレームタイムスタンプ (任意)。
            next_pair_1p / dnext_pair_1p: 1P の次/ダブルネクスト (色のペア)。
            next_pair_2p / dnext_pair_2p: 2P の同上。
            incoming_ojama_1p: 1P が受けている予告お邪魔個数。
            incoming_ojama_2p: 2P が受けている予告お邪魔個数。
        """
        p1 = self._analyze_player(
            board_1p,
            opponent_board=board_2p,
            next_pair=next_pair_1p,
            dnext_pair=dnext_pair_1p,
            incoming_ojama=incoming_ojama_1p,
        )
        p2 = self._analyze_player(
            board_2p,
            opponent_board=board_1p,
            next_pair=next_pair_2p,
            dnext_pair=dnext_pair_2p,
            incoming_ojama=incoming_ojama_2p,
        )
        score = self._scorer.score(p1.indicators, p2.indicators)

        return AnalysisResult(
            player1=p1,
            player2=p2,
            score=score,
            timestamp=timestamp,
        )

    def analyze_player(
        self,
        board: Board,
        opponent_board: Board | None = None,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
        incoming_ojama: int = 0,
    ) -> PlayerAnalysis:
        """
        1プレイヤー分の盤面のみを解析する (片側の評価に使用)。

        Args:
            board: 評価対象の盤面。
            opponent_board: 相手フィールド (凝視 opponent_chain_threat 用)。
            next_pair / dnext_pair: 次/ダブルネクスト。
            incoming_ojama: 受けている予告お邪魔個数。
        """
        return self._analyze_player(
            board,
            opponent_board=opponent_board,
            next_pair=next_pair,
            dnext_pair=dnext_pair,
            incoming_ojama=incoming_ojama,
        )

    # ============================
    # 内部メソッド
    # ============================

    def _analyze_player(
        self,
        board: Board,
        opponent_board: Board | None = None,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
        incoming_ojama: int = 0,
    ) -> PlayerAnalysis:
        """盤面1つから連鎖結果と指標を算出する。"""
        chain_result = self._simulator.simulate(board)
        indicators = self._calculator.compute_all(
            board,
            next_pair=next_pair,
            dnext_pair=dnext_pair,
            incoming_ojama=incoming_ojama,
            opponent_board=opponent_board,
        )
        return PlayerAnalysis(
            board=board,
            chain_result=chain_result,
            indicators=indicators,
        )
