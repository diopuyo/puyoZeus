"""段1 直接公開 (live_x_direct_publish、既定OFF) の OFF 同一性と ON の契約。

OFF: pump・判定ID・完了待ち・予測時間の例外門が従来の経路を通ること。
ON: 送受信の分離、X の初出での起床、判定IDへの X 完了時刻、例外門の記録化。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j import live_completion, live_v49f, live_v49f_async
from src.phase_j.live_x_direct import ENV_NAME, WakeSignal, XFirstTracker, direct_publish_enabled

RESULT_TIMEOUT_SEC = 30.
X_BASE = dict(predicted_scores=[[], []], chain_ids=[10., None])
X_BASE_NEXT = dict(predicted_scores=[[], []], chain_ids=[20., None])


def echo_worker(connection: Any, assets: tuple, output: Path) -> None:
    """spawn 子。packet に埋めた base/value をそのまま返す (FIFO・1件1結果は本物と同じ)。"""
    connection.send({'ready': True})
    while (packet := connection.recv()) is not None:
        connection.send(dict(id=packet['id'], game=packet['game'], t=packet['t'],
            base=packet['echo_base'], value=packet['echo_value'], origins=packet['origins'],
            completed_at=perf_counter(), origin_ms=0., compute_ms=0., wait_ms=0.))
    connection.close()


def value(scores: list) -> dict:
    return dict(predicted_scores=[scores, []], input_sec=0.)


SEQUENCE = (  # (base, value, origins, X の初出か)
    (None, None, [], False),
    (X_BASE, value([40, 700]), [], True),
    (None, value([40, 700]), [], False),
    (X_BASE_NEXT, value([100]), [], True),
    (None, value([120]), [dict(input_sec=0.)], False),     # 同じ発火の X の更新 (origins つき) は起こさない
    (None, None, [dict(input_sec=0.)], False),
)


def owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, direct: bool) -> live_v49f.LiveV49F:
    monkeypatch.setattr(live_v49f_async, 'worker', echo_worker)
    assets = tmp_path/'v49f'
    assets.mkdir()
    (assets/'model.json').write_text(json.dumps(dict(c=.46)), encoding='utf-8')
    monkeypatch.setattr(live_v49f, 'ASSETS', assets)
    monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', '1')
    if direct:
        monkeypatch.setenv(ENV_NAME, '1')
    else:
        monkeypatch.delenv(ENV_NAME, raising=False)
    return live_v49f.LiveV49F(tmp_path/'model', tmp_path/'table', tmp_path)


def feed(view: live_v49f.LiveV49F) -> list[dict]:
    for index, (base, result, origins, _) in enumerate(SEQUENCE):
        packet = dict(game=0, t=float(index), origins=origins, echo_base=base, echo_value=result)
        view.update(dict(frame=index, game_idx=0, _v49f=packet), 1)
    return [view.outputs.get(timeout=RESULT_TIMEOUT_SEC) for _ in SEQUENCE]


def comparable(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in ('completed_at', 'x_fresh')} for r in results]


def test_default_is_off_and_env_switches() -> None:
    from src.production_config import LIVE_X_DIRECT_PUBLISH_DEFAULT
    assert LIVE_X_DIRECT_PUBLISH_DEFAULT is False
    assert direct_publish_enabled({}) is False
    assert direct_publish_enabled({ENV_NAME: '1'}) is True
    assert direct_publish_enabled({ENV_NAME: '0'}) is False


def test_off_keeps_single_pump_and_on_matches_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ('off', 'on'):
        (tmp_path/name).mkdir()
    off = owner(monkeypatch, tmp_path/'off', False)
    try:
        assert off.direct is False and not hasattr(off, 'wake') and off.thread.name != 'v49f-send'
        off_results = feed(off)
    finally:
        off.close()
    assert all('x_fresh' not in r for r in off_results)
    assert [r['generation'] for r in off_results] == [1]*len(SEQUENCE)
    on = owner(monkeypatch, tmp_path/'on', True)
    try:
        on_results = feed(on)
        assert [r['x_fresh'] for r in on_results] == [s[3] for s in SEQUENCE]
        # 未読の起床は合流して高々1件。読んだ後の X 初出は再び起こす。
        assert on.wake.drain() == 1 and on.wake.drain() == 0
    finally:
        on.close()
    assert comparable(on_results) == comparable(off_results)
    status = json.loads((tmp_path/'on/v49f_display_status.json').read_text(encoding='utf-8'))
    fresh = sum(s[3] for s in SEQUENCE)
    # 起床の記録 (ON だけ): X 初出 2 件 → 起床要求 2 回、pipe への実書込みは未読と合流して 1 回。
    assert status == dict(submitted=len(SEQUENCE), received=len(SEQUENCE), failure=None,
                          pump_alive=False, direct_publish=True, x_fresh=fresh, x_fresh_fires=fresh, x_wakes=1)
    off_text = (tmp_path/'off/v49f_display_status.json').read_text(encoding='utf-8')
    assert set(json.loads(off_text)) == {'submitted', 'received', 'failure', 'pump_alive'}
    # OFF の記録は従来とバイト一致 (起床の欄を足していない)。
    assert off_text == json.dumps(dict(submitted=len(SEQUENCE), received=len(SEQUENCE), failure=None,
                                       pump_alive=False), indent=2)


def test_receive_sets_x_completion_only_on_first_x() -> None:
    view = live_v49f.LiveV49F.__new__(live_v49f.LiveV49F)
    from queue import Queue
    view.outputs, view.direct, view.failure, view.reason = Queue(), True, None, '入力待ち'
    view.generation, view.game, view.latest, view.x_completed_at = 1, 0, None, None
    for stamp, fresh in ((1., False), (2., True), (3., False)):
        view.outputs.put(dict(generation=1, game=0, value=dict(), completed_at=stamp, x_fresh=fresh))
    view.receive()
    assert view.latest['completed_at'] == 3. and view.x_completed_at == 2.
    view.update(dict(game_idx=1), 1)
    assert view.x_completed_at is None and view.latest is None


def test_tracker_first_x_per_trigger_and_identity_reset() -> None:
    tracker = XFirstTracker()
    flags = [tracker.observe(dict(base=b, value=v, origins=o), (1, 0)) for b, v, o, _ in SEQUENCE]
    assert flags == [s[3] for s in SEQUENCE]
    assert tracker.observe(dict(base=None, value=value([100]), origins=[]), (2, 0)) is False
    assert tracker.observe(dict(base=X_BASE_NEXT, value=value([100]), origins=[]), (2, 0)) is True
    image_route = dict(predicted_scores=[[], []], chain_ids=[30., None])   # 画像予測経路でも新しい発火なら起こす
    assert tracker.observe(dict(base=image_route, value=value([5]), origins=[dict(input_sec=0.)]), (2, 0)) is True


def test_tracker_counts_fresh_results_and_fires_without_resetting() -> None:
    """記録用の累計: 結果の数 (x_fresh) と (側, 発火) の数。入力世代・試合が変わっても戻さない。"""
    tracker = XFirstTracker()
    both_base = dict(predicted_scores=[[], []], chain_ids=[10., 11.])
    both = dict(predicted_scores=[[40], [50]], input_sec=0.)
    assert (tracker.fresh_results, tracker.fresh_fires) == (0, 0)
    assert tracker.observe(dict(base=both_base, value=both, origins=[]), (1, 0)) is True
    assert (tracker.fresh_results, tracker.fresh_fires) == (1, 2)           # 1 結果で両側 = 2 発火
    assert tracker.observe(dict(base=None, value=both, origins=[]), (1, 0)) is False
    assert tracker.observe(dict(base=None, value=None, origins=[]), (1, 0)) is False
    assert (tracker.fresh_results, tracker.fresh_fires) == (1, 2)           # 同じ発火・値なしは数えない
    assert tracker.observe(dict(base=both_base, value=both, origins=[]), (2, 0)) is True
    assert (tracker.fresh_results, tracker.fresh_fires) == (2, 4)           # 識別が変わっても累計は残る


def test_wake_signal_counts_only_actual_writes() -> None:
    wake = WakeSignal()
    try:
        assert wake.sent == 0
        for _ in range(3):
            wake.notify()
        assert wake.sent == 1                  # 未読と合流した通知は書かない・数えない
        assert wake.drain() == 1
        wake.notify()
        assert wake.sent == 2
    finally:
        wake.close()


def test_parity_record_check_needs_independent_counts_and_untouched_off_record() -> None:
    from scripts.verify_live_x_direct_parity import record_matches
    off_status = dict(submitted=1, received=1, failure=None, pump_alive=False)
    on_status = dict(off_status, direct_publish=True, x_fresh=3, x_fresh_fires=4, x_wakes=3)
    off, on = dict(status=off_status), dict(x_fresh=3, wakes=3, status=on_status)
    assert record_matches(off, on) is True
    assert record_matches(dict(), dict()) is True                           # 記録の欄が無い古い受領票は対象外
    for broken_on in (dict(on, x_fresh=4), dict(on, wakes=2), dict(on, status=dict(on_status, x_fresh_fires=2)),
                      dict(on, status=dict(on_status, direct_publish=False)), dict(on, status=None)):
        assert record_matches(off, broken_on) is False
    assert record_matches(dict(status=dict(off_status, x_fresh=0)), on) is False   # OFF の記録に欄が増えた
    assert record_matches(dict(status=None), on) is False


def judged(direct: bool | None, monkeypatch: pytest.MonkeyPatch) -> tuple:
    from src.phase_j.live_b3 import LiveB3
    monkeypatch.delenv('PUYO_B3_M3', raising=False)
    b3 = LiveB3.__new__(LiveB3)
    b3.completion_driven, b3.generation, b3.result_frame = True, 2, 7
    b3.display = SimpleNamespace(shown=(1., .55), source='null', raw=.5, game=0, contents=None)
    b3.ama = b3.time_gate = b3.v49f_baseline = None
    b3.v49f = None if direct is None else SimpleNamespace(
        direct=direct, x_completed_at=12.5, decorate=lambda row, *a, **k: row.update(v49f={}))
    row = dict(match_active=True)
    b3.decorate(row)
    return row['judgment_id']


def test_judgment_id_off_is_six_elements_and_on_adds_x(monkeypatch: pytest.MonkeyPatch) -> None:
    assert judged(None, monkeypatch) == (2, 7, 'null', .5, True, 0)
    assert judged(False, monkeypatch) == (2, 7, 'null', .5, True, 0)
    assert judged(True, monkeypatch) == (2, 7, 'null', .5, True, 0, 12.5)


def completion_bridge(v49f: Any, b3_done: bool, calls: list, context: bool = True) -> Any:
    alive = SimpleNamespace(is_alive=lambda: True)
    engine = SimpleNamespace(outputs=SimpleNamespace(_reader='b3'), process=alive, compute=alive)
    b3 = SimpleNamespace(completion_driven=True, async_failed=False, async_engine=engine,
                         receive_completed=lambda: b3_done, v49f=v49f)
    bridge = SimpleNamespace(b3=b3, queue=SimpleNamespace(_reader='recognition'), worker=alive,
                             on_b3_completed=lambda: calls.append(True))
    if context:
        bridge.b3_context = ('overlay', 'result', 0)
    return bridge


@pytest.mark.parametrize('v49f', [None, SimpleNamespace(direct=False)])
def test_completion_wait_off_is_unchanged(v49f: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen, calls = [], []
    monkeypatch.setattr(live_completion, 'wait', lambda handles, timeout: seen.append(handles) or ['b3'])
    assert not live_completion.receive_completion(completion_bridge(v49f, True, calls), .1)
    assert seen == [('recognition', 'b3')] and calls == [True]


@pytest.mark.parametrize('b3_ready,context,expected', [(False, True, 1), (True, True, 1), (False, False, 0)])
def test_completion_wake_publishes_once(b3_ready: bool, context: bool, expected: int,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    wake, calls = WakeSignal(), []
    try:
        wake.notify()
        wake.notify()
        ready = [wake.reader, 'b3'] if b3_ready else [wake.reader]
        monkeypatch.setattr(live_completion, 'wait', lambda handles, timeout: ready)
        bridge = completion_bridge(SimpleNamespace(wake=wake), b3_ready, calls, context)
        assert not live_completion.receive_completion(bridge, .1)
        assert len(calls) == expected and not wake.reader.poll()
    finally:
        wake.close()


def test_wake_signal_is_waitable() -> None:
    from multiprocessing.connection import wait
    wake = WakeSignal()
    try:
        assert wait([wake.reader], timeout=0.) == []
        for _ in range(3):
            wake.notify()
        assert wait([wake.reader], timeout=1.) == [wake.reader]
        assert wake.drain() == 1 and wake.drain() == 0
        wake.notify()
        assert wait([wake.reader], timeout=1.) == [wake.reader] and wake.drain() == 1
    finally:
        wake.close()


def origin_message(kind: str, elapsed: float) -> dict:
    return dict(id=0, kind=kind, t=1., order=0, data=dict(input_sec=1., sources=[], elapsed=elapsed, t_sec=1.))


def install_fake_origins(monkeypatch: pytest.MonkeyPatch) -> Any:
    """本物の record (予測時間の例外門) を呼ぶ最小の状態遷移。"""
    from src import prefire_origin_stream as stream

    def initialize(data: dict, policy: dict, native: Any) -> dict:
        first = stream.record('initial', data['input_sec'], data['elapsed'], dict(sent=1), [])
        return dict(initial=dict(low_spec_ms=first['low_spec_ms']), before=[first], after=[dict(first)])

    def on_formula(state: dict, data: dict, inp: dict, policy: dict, native: Any) -> None:
        for name in ('before', 'after'):
            state[name].append(stream.record('formula', data['t_sec'], data['elapsed'], dict(sent=2), []))

    monkeypatch.setattr(stream, 'initialize', initialize)
    monkeypatch.setattr(stream, 'on_formula', on_formula)
    monkeypatch.setattr(stream, 'snapshot', lambda state, data, *a: dict(input_sec=data['input_sec']))
    return stream


def test_budget_gate_off_raises_and_on_records(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_v49f_origins import OriginStream
    from src.prefire_worker import LOW_SPEC_RATIO
    stream = install_fake_origins(monkeypatch)
    over = 2*stream.MAX_LOW_SPEC_MS/LOW_SPEC_RATIO
    with pytest.raises(RuntimeError, match='低スペック予算超過'):
        OriginStream({}, None).observe(0, 1., [origin_message('initial', over)])
    recorded = OriginStream({}, None, record_budget=True)
    values = recorded.observe(0, 1., [origin_message('initial', over)])
    assert values == [dict(input_sec=1., budget_exceeded_ms=pytest.approx(2*stream.MAX_LOW_SPEC_MS))]
    assert stream.MAX_LOW_SPEC_MS == 500. and len(recorded.take_budget_exceeded()) == 1
    assert recorded.take_budget_exceeded() == []


def test_budget_gate_on_counts_only_new_records(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_v49f_origins import OriginStream
    from src.prefire_worker import LOW_SPEC_RATIO
    stream = install_fake_origins(monkeypatch)
    small, over = 1., 2*stream.MAX_LOW_SPEC_MS/LOW_SPEC_RATIO
    plain, recorded = OriginStream({}, None), OriginStream({}, None, record_budget=True)
    first = [plain.observe(0, 1., [origin_message('initial', small)]),
             recorded.observe(0, 1., [origin_message('initial', small)])]
    assert first[0] == first[1] == [dict(input_sec=1.)]
    assert recorded.observe(0, 1., [origin_message('formula', over)]) == [
        dict(input_sec=1., budget_exceeded_ms=pytest.approx(2*stream.MAX_LOW_SPEC_MS))]
    assert recorded.take_budget_exceeded() == [pytest.approx(2*stream.MAX_LOW_SPEC_MS)]


def test_flag_is_bool_speed_option_and_off_config_is_unchanged(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.launcher import LauncherConfigError, build_pipeline_config, parse_user_config
    from src.phase_j.live_speed import SPEED_DEFAULTS, configure
    assert SPEED_DEFAULTS['live_x_direct_publish'] is False
    base = dict(device_name='OBS Virtual Camera')
    assert parse_user_config(base, tmp_path).live_x_direct_publish is False
    assert 'live_x_direct_publish' not in build_pipeline_config(parse_user_config(base, tmp_path))
    enabled = parse_user_config(dict(base, live_x_direct_publish=True), tmp_path)
    assert build_pipeline_config(enabled)['live_x_direct_publish'] is True
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(base, live_x_direct_publish='true'), tmp_path)
    for flag in (False, True):
        options = argparse.Namespace(**dict(SPEED_DEFAULTS, live_x_direct_publish=flag), cpu_threads=1,
            source='dshow', b3_judgment=True, split_evaluation=True, worker_mode='process', compare=False)
        monkeypatch.setattr('os.environ', {})
        configure(options, argparse.ArgumentParser())
        import os
        assert os.environ[ENV_NAME] == str(int(flag))
