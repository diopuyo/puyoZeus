"""入れ子計時の排他性と無効化を確認する。"""
from src.b3_profile import profile, span


def test_exclusive_clock(monkeypatch) -> None:
    ticks = iter([0., 1., 3., 5.])
    monkeypatch.setattr('src.b3_profile.perf_counter', lambda: next(ticks))
    monkeypatch.setattr('src.b3_profile.thread_time', lambda: 0.)
    with profile() as costs:
        with span('feature'):
            with span('search'):
                pass
    assert costs['feature_ms'] == 3000.
    assert costs['search_ms'] == 2000.


def test_disabled_clock(monkeypatch) -> None:
    def fail() -> float:
        raise AssertionError('無効時の時計読取')
    monkeypatch.setattr('src.b3_profile.perf_counter', fail)
    with profile(False) as costs:
        with span('search'):
            pass
    assert costs == {}
