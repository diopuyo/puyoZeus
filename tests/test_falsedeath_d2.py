"""D2: 不安定入力と予測時間だけの上限を死亡証明へ渡さない。"""
from types import SimpleNamespace as NS
import pytest
from src.exchange_post_counter_bound import PostCounterDeathBound


@pytest.mark.parametrize('reason', ['board_unsettled', 'color_score_mismatch'])
def test_unsettled_rejected(reason: str) -> None:
    bound = PostCounterDeathBound(input_integrity=True)
    projection = NS(safety=NS(blocker=lambda *args: reason))
    assert bound.input_blocker(projection, NS(), 0, 1.) == reason


def test_motion_and_unverified_budget() -> None:
    bound = PostCounterDeathBound(input_integrity=True)
    projection = NS(safety=NS(blocker=lambda *args: None))
    chain = NS(end_signal_sec=1., post_end_drop_sec=None, end_confirmed=True)
    tracker = NS(latest_chain=lambda side: chain)
    assert bound.input_blocker(projection, tracker, 0, 2.) == 'unverified_response_upper_bound'
    chain.post_end_drop_sec = 1.5
    assert bound.input_blocker(projection, tracker, 0, 2.) is None
    bound.next_motion = (True, False)
    assert bound.input_blocker(projection, tracker, 0, 2.) == 'next_transition'
    assert not PostCounterDeathBound().input_integrity


@pytest.mark.parametrize('enabled', [False, True])
def test_motion_invalidates_cached_projection(enabled: bool) -> None:
    """色読みが変わる前のNEXT移動でも、ONなら投影キャッシュを再判定する。"""
    from src.board_state_machine import BoardState
    bound = PostCounterDeathBound(input_integrity=enabled)
    side = NS(state=BoardState.STABLE, confirmed_board=None, next_pair=(1, 2),
              dnext_pair=(3, 4), next_slide_motion=False)
    bound.observe(NS(p1=side, p2=side), 1)
    revision = bound.revision
    side.next_slide_motion = True
    bound.observe(NS(p1=side, p2=side), 1)
    assert bound.revision == revision + int(enabled)
