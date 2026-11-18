"""確定済み予告の受け渡しと旗OFFの互換を確認する。"""
import json
import io
from types import SimpleNamespace

import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from tests.test_live_retr_visible import setup_case


@pytest.mark.parametrize('force', [None, '0', '1'])
def test_external_override(monkeypatch: pytest.MonkeyPatch, force: str | None) -> None:
    if force is None:
        monkeypatch.delenv('PUYO_AMA_FORCE_EXTERNAL', raising=False)
    else:
        monkeypatch.setenv('PUYO_AMA_FORCE_EXTERNAL', force)
    monkeypatch.setenv(retr.EXE_ENV, 'C:/job/ama.cmd')
    launches = []
    stream = io.StringIO('{"cummax": [70]}\n{"cummax": [140]}\n')

    def launch(command: list, **kwargs: object) -> SimpleNamespace:
        launches.append((command, kwargs))
        return SimpleNamespace(stdin=io.StringIO(), stdout=stream, wait=lambda timeout: None)

    monkeypatch.setattr(retr.subprocess, 'Popen', launch)
    ama = retr.Ama(SimpleNamespace(ama_reference_json_py=lambda *args: '{"cummax": [35]}'))
    assert ama._ask({}) == {'cummax': [70 if force == '1' else 35]}
    if force == '1':
        assert ama._ask({}) == {'cummax': [140]}
        assert len(launches) == 1
        assert launches[0][1]['env']['AMA_THREADS'] == '1'
        if retr.os.name == 'nt':
            assert launches[0][0][1:] == ['/d', '/c', 'C:/job/ama.cmd']
    else:
        assert not launches
    ama.close()


def test_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(retr.LANDING_ENV, raising=False)
    assert not retr.landing_enabled()


def test_profile_request_and_cache() -> None:
    requests = []

    def ask(line: str, threads: int) -> str:
        requests.append(json.loads(line))
        return json.dumps({'cummax': [len(requests)] * 16})

    ama = retr.Ama(SimpleNamespace(ama_reference_json_py=ask))
    board, pairs = [0] * 78, [[1, 2]] * 16
    assert ama.profile(board, pairs) == ama.profile(board, pairs, pending=0, rate=1)
    assert 'pending' not in requests[0]
    for pending, rate, cap in [(30, 70, 30), (60, 70, 30), (30, 40, 30), (30, 70, 6)]:
        ama.profile(board, pairs, pending, rate, cap)
        assert requests[-1]['pending'] == pending
        assert requests[-1]['rate'] == rate
        assert requests[-1]['drop_cap'] == cap
    assert len(requests) == 5


def test_confirmed_ledger_excludes_active_prediction() -> None:
    from src.phase_j._live_ex_frozen.src.b3_score_ledger import ScoreLedgerV2, _Batch
    core = ScoreLedgerV2()
    core.batches[0] = [_Batch(n=17, src_side=1, chain_id=1, confirmed=True, confirmed_t=1.),
                       _Batch(n=100, src_side=1, chain_id=2, confirmed=False)]
    engine = SimpleNamespace(state=SimpleNamespace(sl=SimpleNamespace(core=core)))
    assert core._pending(0) == 117
    assert retr.confirmed_pending(engine, 0) == 17


@pytest.mark.parametrize('enabled', [False, True])
def test_compute_passes_only_confirmed(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    engine, value = setup_case(monkeypatch)
    calls = []
    engine.retr_ama.profile = lambda *args, **kwargs: calls.append(kwargs) or [70] * 16
    engine.state.sl = SimpleNamespace(core=SimpleNamespace(
        _pending=lambda side, confirmed_only=False: [17, 35][side] if confirmed_only else 999))
    monkeypatch.setenv(retr.LANDING_ENV, str(int(enabled)))
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    extra, _ = retr.compute_retr(engine, value)
    assert calls == ([dict(pending=17, rate=70), dict(pending=35, rate=70)] if enabled else [{}, {}])
    assert engine.retr_diag[0].get('pending') == (17 if enabled else None)
    assert extra['retR_cvis_p1'] == extra['retR_c_p1'] == 1.
