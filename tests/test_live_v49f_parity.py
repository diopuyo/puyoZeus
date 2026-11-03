"""PC1裁定の許容範囲から離散判定を漏らさない。"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('parity', Path(__file__).parents[1] / 'packaging/verify_live_v49f_windows.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_probability_tolerance_and_endpoints() -> None:
    assert not parity.differences({'v_post': .5}, {'v_post': .5 + 1e-10})
    assert parity.differences({'v_post': .5}, {'v_post': .5 + 2e-9})
    assert parity.differences({'v_post': 0.}, {'v_post': 1e-16})
    assert parity.differences({'v_post': 1.}, {'v_post': 1. - 1e-16})
    assert parity.differences({'v_post': .5}, {'v_post': float('nan')})


def test_discrete_and_causal_fields_are_exact() -> None:
    for key, a, b in [('active', True, False), ('death_sides', [0], [1]),
                      ('held', True, False), ('released', None, 'changed'),
                      ('reject', None, 'confirmed_short'), ('input_sec', 1., 1. + 1e-10),
                      ('route', 'rep', 'exact'), ('v_post', None, .5)]:
        assert parity.differences({key: a}, {key: b})
    assert not parity.differences({'pre_delta_full': .1}, {'pre_delta_full': .1 + 1e-16})
