"""幽霊連鎖ルール 本番ON配線の回帰防止テスト (2026-08-10)。

`src/chain.py` の `ChainSimulator(exclude_hidden_row_from_pop=True)` 自体の
正しさは `tests/test_chain.py::TestGhostChainRule` が担保する。

本ファイルはルールの正しさでなく **配線** を担保する:
    - `src/production_config.py` の `GHOST_CHAIN_RULE_ENABLED` が単一の
      情報源として True になっていること。
    - 本番構築箇所 (indicators_v2 / chain_detector / inference_board /
      recognition_pipeline / recognition_evaluator / self_supervised /
      exchange_virtual_board) が実際にこのフラグを受け取っていること。

過去に `--early-fire-reaction` を配線し忘れて機能が無いのと同じ状態に
なった事故 (memory feedback_recognition_regression_prevention) があるため、
「フラグは実装したが呼び出し側が使っていない (dead code 化)」を機械的に
検出する目的で追加する。
"""

from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src import production_config


def _grid_with_ghost_group() -> list[list[int]]:
    """13段目(row0)+可視3段の縦4連結を持つグリッド (ONなら消えない・OFFなら消える)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for row in range(4):
        grid[row][0] = COLOR_RED
    return grid


def _board_with_ghost_group() -> Board:
    return Board.from_list(_grid_with_ghost_group())


# ============================
# 1. production_config: 単一情報源そのもの
# ============================


class TestProductionConfigFlag:
    def test_ghost_chain_rule_enabled_is_true(self) -> None:
        assert production_config.GHOST_CHAIN_RULE_ENABLED is True

    def test_adopted_record_exists_with_reason(self) -> None:
        assert len(production_config.CHAIN_SIM_ADOPTED) >= 1
        flag = production_config.CHAIN_SIM_ADOPTED[0]
        assert "exclude_hidden_row_from_pop" in flag.flag
        assert flag.reason  # 根拠が空でないこと

    def test_describe_includes_chain_sim_section(self) -> None:
        text = production_config.describe()
        assert "連鎖シミュレーション" in text
        assert "exclude_hidden_row_from_pop" in text


# ============================
# 2. indicators_v2: 共有 simulator (最重要、~30 関数が依存)
# ============================


class TestIndicatorsV2Wiring:
    def test_shared_simulator_has_ghost_rule_on(self) -> None:
        from src import indicators_v2 as iv

        assert iv._SHARED_SIMULATOR._exclude_hidden_row_from_pop is True

    def test_shared_simulator_behavior_respects_ghost_rule(self) -> None:
        """`iv._SHARED_SIMULATOR` 経由の実 simulate() が実際にルールを反映すること。

        全指標関数が共有する simulator の挙動を直接確認する
        (属性フラグだけでなく実際の消去判定結果で担保する)。
        """
        from src import indicators_v2 as iv

        board = _board_with_ghost_group()
        result = iv._SHARED_SIMULATOR.simulate(board)
        # ON なら 13段目(row0)+可視3段の縦4連結は消えない。
        assert result.chain_count == 0
        assert result.final_board.count_puyos() == 4


# ============================
# 3. chain_detector.VideoChainTracker
# ============================


class TestChainDetectorWiring:
    def test_video_chain_tracker_simulator_on(self) -> None:
        from src.chain_detector import VideoChainTracker

        tracker = VideoChainTracker()
        assert tracker._simulator._exclude_hidden_row_from_pop is True


# ============================
# 4. inference_board.InferenceBoardGenerator
# ============================


class TestInferenceBoardWiring:
    def test_generator_simulator_on(self) -> None:
        from src.inference_board import InferenceBoardGenerator

        gen = InferenceBoardGenerator()
        assert gen._sim._exclude_hidden_row_from_pop is True


# ============================
# 5. recognition_evaluator.RecognitionEvaluator
# ============================


class TestRecognitionEvaluatorWiring:
    def test_evaluator_simulator_on(self) -> None:
        from src.recognition_evaluator import RecognitionEvaluator

        ev = RecognitionEvaluator()
        assert ev._chain_sim._exclude_hidden_row_from_pop is True


# ============================
# 6. self_supervised (Phase I)
# ============================


class TestSelfSupervisedWiring:
    def test_chain_validator_simulator_on(self) -> None:
        from src.self_supervised.chain_validator import ChainValidator

        validator = ChainValidator()
        assert validator._sim._exclude_hidden_row_from_pop is True

    def test_chain_fine_tuner_simulator_on(self, tmp_path) -> None:
        from src.self_supervised.chain_fine_tuner import ChainFineTuner

        tuner = ChainFineTuner(calibration_path=tmp_path / "cal.json")
        assert tuner._simulator._exclude_hidden_row_from_pop is True


# ============================
# 7. exchange_virtual_board (#24 打ち合い、実盤面指標)
# ============================


class TestExchangeVirtualBoardWiring:
    def test_reconstruct_virtual_board_pair_respects_ghost_rule(self) -> None:
        from src.exchange_virtual_board import reconstruct_virtual_board_pair

        attacker_board = _board_with_ghost_group()
        opponent_board = Board()
        pair = reconstruct_virtual_board_pair(
            before_board=attacker_board,
            opponent_board=opponent_board,
            net_ojama_after_pred=0.0,
        )
        # simulator 省略時のデフォルトが ON なら、幽霊4連結は消えず
        # attacker_board_after にそのまま残るはず。
        assert pair.attacker_board_after.count_puyos() == 4


# ============================
# 8. recognition_pipeline (RecognitionPipeline / ChainPhaseDetector 注入)
# ============================


class TestRecognitionPipelineWiring:
    """既存 tests/test_recognition_pipeline.py の `_make_pipe` ヘルパーを再利用し、
    実際に本番経路で構築される RecognitionPipeline インスタンスを検証する。
    """

    def test_pipeline_chain_sim_on(self) -> None:
        from tests.test_recognition_pipeline import _make_pipe

        pipe = _make_pipe(Board(), Board())
        assert pipe._chain_sim._exclude_hidden_row_from_pop is True

    def test_state_machine_chain_phase_detector_sim_on(self) -> None:
        from src.state_detectors import ChainPhaseDetector
        from tests.test_recognition_pipeline import _make_pipe

        pipe = _make_pipe(Board(), Board())
        chain_dets = [
            d for d in pipe._sm_1p._detectors
            if isinstance(d, ChainPhaseDetector)
        ]
        assert len(chain_dets) == 1
        assert chain_dets[0].chain_sim is not None
        assert chain_dets[0].chain_sim._exclude_hidden_row_from_pop is True
