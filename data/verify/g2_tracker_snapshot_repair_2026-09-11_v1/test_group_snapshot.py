"""実trackerのキャッシュ全体とグループ改変を検査する。原J完走ではない。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace as N
import pytest
import group_snapshot as G
import reproduce as R


def test_real_tracker_cache() -> None:
    join, tracker, evidence = R.load_join(), R.tracker(), {}
    with pytest.raises(join.SideJoinUninspectable, match='max_depth:PuyoGroup'):
        join._vrepr(tracker)
    original = join._vrepr_leaf
    with ExitStack() as stack:
        kind = G.group_type(N(_chain_tracker_2p=tracker))
        G.install(stack, join, kind, evidence)
        before = join._vrepr(tracker)
        assert join._vrepr(tracker) == before
        result = next(iter(tracker._simulator._cache.values()))
        group = result.steps[0].erased_groups[0]
        result.steps[0].erased_groups[0] = replace(group, color=group.color + 1)
        assert join._vrepr(tracker) != before
    assert join._vrepr_leaf is original and evidence['restored']
    assert join._MAX_DEPTH == 6


@pytest.mark.parametrize('field', ('color', 'cells', 'ojama_adjacent'))
def test_every_group_field_detected(field: str) -> None:
    from src.chain import PuyoGroup
    group = PuyoGroup(1, frozenset({(12, 0)}), 1, frozenset())
    changed = dict(color=2, cells=frozenset({(11, 0)}), size=2,
                   ojama_adjacent=frozenset({(12, 1)}))
    render = G.leaf(lambda _value: 'unhandled', PuyoGroup)
    assert render(group) != render(replace(group, **{field: changed[field]}))


def test_foreign_type_and_extra_fields() -> None:
    from src.chain import PuyoGroup
    render = G.leaf(lambda _value: 'unhandled', PuyoGroup)
    fake = type('PuyoGroup', (), {})()
    assert render(fake) == 'unhandled'
    real = PuyoGroup(1, frozenset(), 0, frozenset())
    object.__setattr__(real, 'extra', '未検査を許さない')
    with pytest.raises(ValueError, match='extra_fields'):
        render(real)


def test_size_and_bounds_rejected() -> None:
    from src.chain import PuyoGroup
    render = G.leaf(lambda _value: 'unhandled', PuyoGroup)
    with pytest.raises(ValueError, match='size_mismatch'):
        render(PuyoGroup(1, frozenset(), 1, frozenset()))
    with pytest.raises(ValueError, match='cell_range'):
        render(PuyoGroup(1, frozenset({(13, 0)}), 1, frozenset()))


def test_unknown_deep_type_still_rejected() -> None:
    from src.chain import PuyoGroup
    join = R.load_join()
    with ExitStack() as stack:
        G.install(stack, join, PuyoGroup, {})
        with pytest.raises(join.SideJoinUninspectable, match='max_depth'):
            join._vrepr(N(x=1), 6)


def test_cleanup_preserves_original_failure() -> None:
    from src.chain import PuyoGroup
    join, evidence, body = R.load_join(), {}, RuntimeError('body_failure')
    foreign = lambda _value: 'foreign'
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            G.install(stack, join, PuyoGroup, evidence)
            join._vrepr_leaf = foreign
            raise body
    assert caught.value is body and join._vrepr_leaf is foreign
    assert evidence['cleanup_error'] and not evidence['restored']


def test_subclass_and_deeper_holder_rejected() -> None:
    from src.chain import PuyoGroup
    class Other(PuyoGroup):
        pass
    join = R.load_join()
    with ExitStack() as stack:
        G.install(stack, join, PuyoGroup, {})
        with pytest.raises(join.SideJoinUninspectable, match='max_depth'):
            join._vrepr(Other(1, frozenset(), 0, frozenset()), 7)
        with pytest.raises(join.SideJoinUninspectable, match='max_depth:ChainStep'):
            join._vrepr(N(value=R.tracker()))


def test_double_install_rejected() -> None:
    from src.chain import PuyoGroup
    join = R.load_join()
    with ExitStack() as stack:
        G.install(stack, join, PuyoGroup, {})
        with pytest.raises(AssertionError, match='foreign_leaf|already_installed'):
            G.install(stack, join, PuyoGroup, {})
