"""A1の同一候補選択と、未来情報を使わない補完を検証する。"""
from types import SimpleNamespace as NS

from src.board import Board, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.exchange_completion_score import CompletionScoreFallback
from src.exchange_prefire_candidates import PrefireCandidates, statistics
from src.exchange_hidden_row_belief import majority_statistics
from src.exchange_hidden_row_probability import scenarios


def test_representative_is_joint_and_ties_prefer_longer() -> None:
    """最大段数や平均点を混ぜず、同率だけ長い候補を取る。"""
    options = [dict(weight=w, score=s, prefix=tuple(range(n)), board=[[s]])
               for w, s, n in ((3, 100, 1), (3, 500, 2), (1, 9000, 10))]
    result = majority_statistics(options, 0, coherent=True)
    assert (result['selected_score'], result['predicted_count'], result['board']) == (500, 2, [[500]])
    assert result['min_score'] == 100 and result['max_score'] == 9000
    assert result['mean_score'] == 10800 / 7
    assert statistics(options, 0)['predicted_count'] == 10


def fixture(stamp: float = 2.0) -> tuple:
    """段消去後の観測盤面と、予測不能な連鎖を作る。"""
    board = Board()
    board._grid[-1, :4] = 1
    chain = NS(chain_id=1, side='1P', trigger_sec=1., end_signal_sec=None,
               predicted_final_score=0, predicted_chain_count=0, predicted_final_board=None)
    side = NS(state=BoardState.GRAVITY_SETTLE, midchain_board=board,
              chain_event=NS(mechanism='formula_read', trigger_sec=stamp, chain_count=1, total_score=40))
    overlay = NS(_game=1, tracker=NS(latest_chain=lambda label: chain if label == '1P' else None),
                 _landing_projection=NS(simulator=ChainSimulator()))
    return chain, side, overlay


def test_remaining_uses_absolute_chain_bonus() -> None:
    """残りの消去を2段目として320点加算する。"""
    chain, side, overlay = fixture()
    CompletionScoreFallback().observe(overlay, NS(p1=side, p2=side), 2.)
    assert (chain.predicted_final_score, chain.predicted_chain_count) == (360, 2)
    assert not any(map(any, chain.predicted_final_board))


def test_future_formula_and_frozen_board_are_not_used() -> None:
    """未来通知とSTABLE凍結盤面からは補完しない。"""
    chain, side, overlay = fixture(3.)
    engine = CompletionScoreFallback()
    engine.observe(overlay, NS(p1=side, p2=side), 2.)
    assert chain.predicted_final_score == 0
    side.state = BoardState.CHAIN
    engine.observe(overlay, NS(p1=side, p2=side), 3.)
    assert chain.predicted_final_score == 0


def test_existing_prediction_is_preserved() -> None:
    """独立予測がある場合は候補選択の結果を上書きしない。"""
    chain, side, overlay = fixture()
    chain.predicted_final_score, chain.predicted_chain_count = 900, 3
    CompletionScoreFallback().observe(overlay, NS(p1=side, p2=side), 2.)
    assert (chain.predicted_final_score, chain.predicted_chain_count) == (900, 3)


def test_model_scenario_keeps_representative_completion() -> None:
    """期待得点の別候補へ戻さず、代表盤面と同じ得点でS3を評価する。"""
    chain = NS(provisional_score=900., predicted_final_score=500.)
    engine = NS(coherent=True, active=lambda c: dict(options=[dict(score=100., weight=.9)]))
    assert scenarios(engine, [chain]) == [((500.,), 1.)]


def test_remaining_zero_keeps_observed_total() -> None:
    """残り消去が無い観測盤面でも、観測済みの式をゼロに戻さない。"""
    chain, side, overlay = fixture()
    side.midchain_board = Board()
    CompletionScoreFallback().observe(overlay, NS(p1=side, p2=side), 2.)
    assert (chain.predicted_final_score, chain.predicted_chain_count) == (40, 1)


def test_unknown_board_cannot_be_completion() -> None:
    """不明セルを空扱いした完走を作らない。"""
    chain, side, overlay = fixture()
    side.midchain_board._grid[-1, 0] = COLOR_UNKNOWN
    CompletionScoreFallback().observe(overlay, NS(p1=side, p2=side), 2.)
    assert (chain.predicted_final_score, chain.predicted_chain_count) == (0, 0)


def test_formula_updates_only_own_fallback() -> None:
    """補完の続きは新しい観測累計と絶対段数で更新する。"""
    chain, side, overlay = fixture()
    engine = CompletionScoreFallback()
    engine.observe(overlay, NS(p1=side, p2=side), 2.)
    side.chain_event.chain_count, side.chain_event.total_score = 2, 360
    side.chain_event.trigger_sec = 3.
    engine.observe(overlay, NS(p1=side, p2=side), 3.)
    assert (chain.predicted_final_score, chain.predicted_chain_count) == (1000, 3)
    engine.reset()
    assert not engine.entries and len(engine.audit) == 2


def test_representative_send_keeps_original_rate_clock() -> None:
    """代表値化しても候補生成時のマージンタイム基準を変えない。"""
    engine = PrefireCandidates(ChainSimulator())
    engine.coherent = True
    chain = NS(side='1P', predicted_final_score=700)
    engine.active = lambda c: dict(elapsed=0., stats=dict(mean_send=10))
    assert engine.sends([chain], elapsed=200.) == [10., 0.]
