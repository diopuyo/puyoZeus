"""厳密得点の採用は明示指定に限定し、既定の計算経路を守る。"""
from types import SimpleNamespace

import pytest

from src import indicators_v2 as iv
from src.board import Board
from src.chain import ChainSimulator
from src.puyo_core_bridge import ChainSimResult
from src.scoring import score_to_ojama

LEGACY_SCORE = 140
EXACT_SCORE = 700


@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("exact", [None, False, True])
def test_bridge_score_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch, known: bool, exact: bool | None,
) -> None:
    """Rust結果でも既定では既存計算関数へ渡し、既知・自由展開とも伝播する。"""
    result = ChainSimResult(1, 5, 0, LEGACY_SCORE, EXACT_SCORE, Board())
    calls = []

    def legacy_score(value: ChainSimResult) -> SimpleNamespace:
        calls.append(value)
        return SimpleNamespace(total_score=LEGACY_SCORE)

    monkeypatch.setattr(iv, "calculate_chain_score", legacy_score)
    sim = SimpleNamespace(simulate=lambda board: result)
    pair = (1, 1) if known else None
    options = {} if exact is None else {"use_exact_score": exact}
    actual = iv.near_future_fire_power(
        Board(), pair, pair, simulator=sim, k_levels=(1,),
        active_colors=(1,), beam_width=1, **options,
    )
    expected = EXACT_SCORE if exact else LEGACY_SCORE
    assert actual.values[1].raw == score_to_ojama(expected).ojama_count
    assert (not calls) if exact else (calls and all(v is result for v in calls))


def test_python_result_falls_back_to_legacy_score() -> None:
    """exact_scoreを持たない従来結果は明示指定しても同値になる。"""
    board = Board()
    board._grid[-1, :3] = 1
    options = dict(simulator=ChainSimulator(), k_levels=(1,), active_colors=(1,))
    assert iv.near_future_fire_power(board, use_exact_score=True, **options) == (
        iv.near_future_fire_power(board, **options)
    )


def test_training_init_does_not_replace_production_score() -> None:
    """学習ワーカーの初期化が同じプロセスの既定指標を変更しない。"""
    from scripts import train_exchange_event_models_v2_20260927 as train

    original = iv.calculate_chain_score
    train.init_worker()
    assert iv.calculate_chain_score is original
