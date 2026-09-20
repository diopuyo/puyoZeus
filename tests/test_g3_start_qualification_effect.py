"""effect許可の限定接続だけを検査し、他の検査を緩めていないことを反例で示す。"""
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import importlib.util
import sys

import pytest

from scripts import g3_start_qualification_effect as P

KEY = 'data/verify/g2_start_client_session_contract_2026-09-12_v1/start_qualification.py'


def loaded(monkeypatch: Any, name: str) -> Any:
    path = P.G.ROOT / KEY
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def trace(states: list[str]) -> list[dict]:
    """validate_traceが要求する最小構造を、状態だけ差し替えて作る。"""
    rows = []
    for index, state in enumerate(states):
        rows.append(dict(frame=index * 2, game=0, active=False, states=[state, 'stable'],
                         scores=[0, 0], gates=[False, False], unsettled=[False, False],
                         resets=[0, 0], pending=[0, 0], capped=[0, 0], leftover=[0, 0],
                         activity={'generated': [0, 0], 'offset_uncapped': [0, 0],
                                   'dropped_uncapped': [0, 0], 'clamp_loss': [0, 0],
                                   'finalized': [0, 0]}))
    return rows


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    stack.callback(setattr, owner, name, getattr(owner, name))
    setattr(owner, name, value)


def test_original_rejects_effect(monkeypatch: Any) -> None:
    """元の検査は effect を拒否する。反例が実在することを先に示す。"""
    module = loaded(monkeypatch, '_effect_original')
    rows = trace(['stable', 'effect'])
    with pytest.raises(ValueError, match='start_qualification:board_state'):
        module.validate_trace(rows, 0, 2)


def test_patched_accepts_effect(monkeypatch: Any) -> None:
    """差し替え後は effect を通し、scope終了で元へ戻る。"""
    module = loaded(monkeypatch, '_effect_patched')
    original = module.validate_trace
    rows = trace(['stable', 'effect'])
    with ExitStack() as stack:
        receipt = P.install(stack, replace)
        assert any(item['file'] == KEY for item in receipt['receipts'])
        module.validate_trace(rows, 0, 2)
    assert module.validate_trace is original
    with pytest.raises(ValueError, match='start_qualification:board_state'):
        module.validate_trace(rows, 0, 2)


def test_patched_still_rejects_unknown_state(monkeypatch: Any) -> None:
    """差し替えは effect の1語だけ。未知の状態名は依然として拒否する。"""
    module = loaded(monkeypatch, '_effect_unknown')
    rows = trace(['stable', 'bogus_state'])
    with ExitStack() as stack:
        P.install(stack, replace)
        with pytest.raises(ValueError, match='start_qualification:board_state'):
            module.validate_trace(rows, 0, 2)


def test_patched_keeps_other_checks(monkeypatch: Any) -> None:
    """他の検査を緩めていない。coverageとstate_typesは元のまま落ちる。"""
    module = loaded(monkeypatch, '_effect_other')
    with ExitStack() as stack:
        P.install(stack, replace)
        with pytest.raises(ValueError, match='start_qualification:coverage'):
            module.validate_trace(trace(['effect']), 0, 4)
        broken = trace(['effect'])
        broken[0]['game'] = 'not_int'
        with pytest.raises(ValueError, match='start_qualification:state_types'):
            module.validate_trace(broken, 0, 0)


def test_source_sha_guard(monkeypatch: Any) -> None:
    """原G2ファイルが変わったら黙って通さず、名前付きで落とす。"""
    module = loaded(monkeypatch, '_effect_sha')
    monkeypatch.setitem(P.TARGET_SHA, KEY, '0' * 64)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='effect_allowlist_source_sha'):
            P.install(stack, replace)
    assert module.validate_trace.__name__ == 'validate_trace'


def test_no_target_loaded_is_recorded_not_raised() -> None:
    """対象0件は事実として記録する。巻き戻し中に投げて原因の例外を隠さない。"""
    with ExitStack() as stack:
        receipt = P.install(stack, replace)
    assert receipt['replacements'] == 0 and receipt['no_target_loaded'] is True
    assert receipt['replaced_files'] == []
