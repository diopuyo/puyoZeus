"""外部Amaの候補共有、問い合わせの排他、既定経路の不変を検証する。"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import sys

import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from src.phase_j._live_ex_frozen.src.dl1k_engine import ExEngine
from src.phase_j.live_light_ex import clone_engine


def engine(ama: retr.Ama) -> SimpleNamespace:
    return SimpleNamespace(native=None, pool=ThreadPoolExecutor(max_workers=1),
                           knowledge=SimpleNamespace(lookup=None), retr_ama=ama, state={'x': [1]})


@pytest.mark.parametrize('flag', [None, '0'])
def test_off_clone_keeps_old_constraints(monkeypatch: pytest.MonkeyPatch, flag: str | None) -> None:
    if flag is None:
        monkeypatch.delenv('PUYO_AMA_FORCE_EXTERNAL', raising=False)
    else:
        monkeypatch.setenv('PUYO_AMA_FORCE_EXTERNAL', flag)
    ama = retr.Ama(SimpleNamespace(ama_reference_json_py=lambda *args: '{"cummax": [70]}'))
    source = engine(ama)
    clone = clone_engine(source)
    try:
        assert clone.retr_ama is not ama
        assert clone.retr_ama._ask({}) == ama._ask({}) == {'cummax': [70]}
        clone.retr_ama.cache['x'] = [1]
        assert not ama.cache
        ama.proc = object()
        with pytest.raises(RuntimeError, match='候補分離には配布構成のnative AMAが必要です'):
            clone_engine(source)
    finally:
        ama.proc = None
        ExEngine.close(clone)
        ExEngine.close(source)


def test_shared_process_serial_requests_and_cache_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_AMA_FORCE_EXTERNAL', '1')
    monkeypatch.setenv(retr.EXE_ENV, 'test-echo')
    launches = []
    popen = retr.subprocess.Popen
    code = ('import sys,json,time\nfor s in sys.stdin:\n r=json.loads(s); time.sleep(.002); '
            'print(json.dumps({"cummax":[r.get("pending",0)+r.get("rate",0)+r.get("drop_cap",0)]}),flush=True)')

    def launch(*args: object, **kwargs: object) -> object:
        proc = popen([sys.executable, '-u', '-c', code], stdin=retr.subprocess.PIPE,
                     stdout=retr.subprocess.PIPE, text=True, bufsize=1)
        launches.append(proc)
        return proc

    monkeypatch.setattr(retr.subprocess, 'Popen', launch)
    ama = retr.Ama(None)
    source = engine(ama)
    clone = clone_engine(source)
    try:
        assert clone.retr_ama is ama
        assert clone.retr_ama._finalizer is ama._finalizer
        requests = [dict(pending=i) for i in range(12)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(ama._ask, requests)) == [{'cummax': [i]} for i in range(12)]
        clone2 = clone_engine(source)
        assert clone2.retr_ama.proc is ama.proc is launches[0]
        ExEngine.close(clone2)
        ExEngine.close(clone)
        assert ama.proc.poll() is None
        for pending, rate, cap in [(30, 70, 30), (60, 70, 30), (30, 40, 30), (30, 70, 6)]:
            expected = [pending+rate+cap]
            assert ama.profile([0]*78, [[1, 2]]*16, pending, rate, cap) == expected
            assert clone.retr_ama.profile([0]*78, [[1, 2]]*16, pending, rate, cap) == expected
        assert len(launches) == 1 and ama.calls == ama.hits == 4
    finally:
        ExEngine.close(clone)
        ExEngine.close(source)
        ama.close()
    assert launches[0].poll() == 0
