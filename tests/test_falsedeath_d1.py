"""D1: 不整合の拒否を旧死亡へ反転させない。"""
import pytest

from src.exchange_single_death_proof import resolve_negative_only, INPUT_INTEGRITY_REASONS


@pytest.mark.parametrize('reason', sorted(INPUT_INTEGRITY_REASONS))
def test_integrity_rejection(reason: str) -> None:
    result = dict(dead=False, reason=reason, nodes=0)
    assert resolve_negative_only(result)['dead']
    assert resolve_negative_only(result, True) == result


def test_proved_death_and_search_limit_unchanged() -> None:
    """探索結果自体の意味は変更しない。"""
    assert resolve_negative_only(dict(dead=True, reason='all_responses_dead'), True)['dead']
    assert resolve_negative_only(dict(dead=False, reason='node_limit'), True)['dead']
