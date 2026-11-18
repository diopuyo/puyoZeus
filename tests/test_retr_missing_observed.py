"""予測から作った連鎖の観測欠落と既存の観測値を確認する。"""
import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from tests.test_live_retr_visible import setup_case


@pytest.mark.parametrize('observed', [None, [], [{'count': 2, 't': 3.}]])
def test_extend_observed(monkeypatch: pytest.MonkeyPatch, observed: list | None) -> None:
    engine, value = setup_case(monkeypatch, busy=True)
    chain = dict(started=1., best=dict(scores=[40, 320, 640]))
    if observed is not None:
        chain['observed'] = observed
    engine.chains[0] = chain
    monkeypatch.setenv(retr.DEADLINE_EXTEND_ENV, '1')
    obs = observed or []
    expected = retr.extended_end(4., 1.+retr.duration(3),
        max((x['count'] for x in obs), default=0), 3,
        max((x['t'] for x in obs), default=None), 5., True)
    assert retr.extend_side_end(engine, value, 0, 5., 4.) == expected


@pytest.mark.parametrize('observed,expected', [(None, 0), ([], 0),
    ([{'count': 1, 't': 2.}, {'count': 2, 't': 3.}], 2)])
def test_compute_observed(monkeypatch: pytest.MonkeyPatch,
                          observed: list | None, expected: int) -> None:
    engine, value = setup_case(monkeypatch)
    chain = dict(started=1., best=dict(scores=[40, 320, 640]))
    if observed is not None:
        chain['observed'] = observed
    engine.chains[1] = chain
    retr.compute_retr(engine, value)
    assert engine.retr_diag[0]['n_obs_o'] == expected
