"""候補ごとの推論と保持・失効の検査。"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import math
import numpy as np
import pytest

from src.phase_j import live_light_probability as probability
from src.phase_j.live_light_ex import clone_engine
from src.phase_j.live_ex_hold import LiveExHold
from src.phase_j._live_ex_frozen.src.dl1k_columns import COLUMNS, LEARNING_COLUMNS


class _PhysicalMustNotTravel:
    def __reduce__(self) -> tuple:
        raise AssertionError('特徴を作るだけの配送に物理探索の全候補は不要')


class _SnapshotForecast:
    def __init__(self) -> None:
        self.fc = SimpleNamespace(total=123, observed=(40, 80))
        self._light_physics = _PhysicalMustNotTravel()

    def current(self) -> SimpleNamespace:
        return self.fc


def test_feature_snapshot_keeps_choices_without_search_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import pickle
    from src import b3_async
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    selected = [choice(1, .5), choice(3, .3), choice(5, .2)]
    monkeypatch.setattr(probability, 'scenarios', lambda owners: selected)
    monkeypatch.setattr(b3_async, '_ledger_snapshot', lambda state: None)
    owner = _SnapshotForecast()
    state = SimpleNamespace(sides=[SimpleNamespace(forecast=owner), SimpleNamespace(forecast=None)])
    rows = SimpleNamespace(state=state, trigger=7, table=None, video_id='snapshot-test')
    fixed, trigger = pickle.loads(b3_async.snapshot(rows))
    assert trigger == 7
    assert fixed.sides[0].forecast.current().total == 123
    assert fixed.sides[0].forecast._light_physics is None
    assert isinstance(owner._light_physics, _PhysicalMustNotTravel)
    assert not hasattr(state, '_light_snapshot_choices')
    rows.state = fixed
    monkeypatch.setattr(probability, 'scenarios', lambda owners: pytest.fail('配送前の候補を再計算しない'))
    row = {}
    probability.attach_context(rows, row)
    assert row[probability.CANDIDATES] == selected
    assert probability.CONTEXT in row
    fixed.sides[0].forecast.fc.total = 999
    assert owner.current().total == 123


def test_no_candidate_row_does_not_encode_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import b3_async
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    monkeypatch.setattr(probability, 'scenarios', lambda owners: [])
    monkeypatch.setattr(b3_async, 'snapshot', lambda rows: pytest.fail('候補なしの文脈は使わない'))
    rows = SimpleNamespace(state=SimpleNamespace(sides=[SimpleNamespace(forecast=None)]))
    row = {'existing': 42}
    probability.attach_context(rows, row)
    assert row == {'existing': 42, probability.CANDIDATES: []}


def choice(total: int, weight: float) -> dict:
    extra = dict.fromkeys(COLUMNS, float(total))
    extra.update(av_ex_p1=.2, av_ex_p2=.2)
    return dict(sides=({'total': total, 'chain_id': 0.}, None), weight=weight,
                retained_mass=1., extra=extra, result={})


def test_average_is_after_nonlinear_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [dict(choice(1, .75), values={}), dict(choice(3, .25), values={})]
    monkeypatch.setattr(probability, 'candidate_row', lambda context, row: {'score': row['sides'][0]['total']})
    monkeypatch.setattr(probability.pickle, 'loads', lambda _: ())
    monkeypatch.setattr(probability.zlib, 'decompress', lambda _: b'')
    owner = SimpleNamespace(infer_single=lambda row, values: row['score']**2/10)
    actual = probability.weighted_infer(owner, {probability.CONTEXT: ''}, {probability.SCENARIOS: selected})
    assert actual == pytest.approx(.3)
    assert actual != pytest.approx((.75*1+.25*3)**2/10)


def test_cached_rows_keep_new_weights_and_ex(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []
    def build(context: tuple, selected: dict) -> dict:
        built.append(selected['sides'][0]['total'])
        return {'score': selected['sides'][0]['total']}
    monkeypatch.setattr(probability, 'candidate_row', build)
    monkeypatch.setattr(probability.pickle, 'loads', lambda _: ())
    monkeypatch.setattr(probability.zlib, 'decompress', lambda _: b'')
    def infer(row: dict, values: dict) -> float:
        score = row.pop('score')
        return (score + values['extra'])/10
    owner = SimpleNamespace(infer_single=infer)
    selected = [dict(choice(1, .75), values={'extra': 0}), dict(choice(3, .25), values={'extra': 0})]
    row = {probability.CONTEXT: ''}
    assert probability.weighted_infer(owner, row, {probability.SCENARIOS: selected}) == pytest.approx(.15)
    selected[0].update(weight=.25, values={'extra': 1})
    selected[1]['weight'] = .75
    assert probability.weighted_infer(owner, row, {probability.SCENARIOS: selected}) == pytest.approx(.275)
    assert built == [1, 3]


def test_cached_rows_invalidate_context_and_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []
    def build(context: tuple, selected: dict) -> dict | None:
        built.append((context, probability.identity(selected)))
        return {'score': 1} if context == b'active' else None
    monkeypatch.setattr(probability, 'candidate_row', build)
    monkeypatch.setattr(probability.base64, 'b85decode', lambda value: value.encode())
    monkeypatch.setattr(probability.zlib, 'decompress', lambda value: value)
    monkeypatch.setattr(probability.pickle, 'loads', lambda value: value)
    owner = SimpleNamespace()
    selected = choice(1, 1.)
    assert probability.prepared_rows(owner, {probability.CONTEXT: 'active'}, [selected]) == [{'score': 1}]
    selected['sides'][0]['final'] = [[1]]
    assert probability.prepared_rows(owner, {probability.CONTEXT: 'active'}, [selected]) == [{'score': 1}]
    assert probability.prepared_rows(owner, {probability.CONTEXT: 'ended'}, [selected]) == [None]
    assert probability.prepared_rows(owner, {probability.CONTEXT: 'ended'}, [selected]) == [None]
    assert len(built) == 3
    assert len(owner._light_row_cache[1]) == 1


def test_batch_preserves_modes_columns_and_float32() -> None:
    from src.b3_ensemble import EnsembleTwoStateModel, aggregate_probabilities
    from src.b3_predict import RowPredictor
    from src.phase_j.live_light_batch import predict_rows
    class Model:
        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            assert matrix.dtype == np.float32
            probability = 1/(1+np.exp(-np.nan_to_num(matrix[:, 0]-matrix[:, 1], nan=0.)))
            return np.column_stack((1-probability, probability))
    members = [SimpleNamespace(static=SimpleNamespace(columns=['a', 'b'], model=Model()),
                               exchange=SimpleNamespace(columns=['b', 'a'], model=Model())) for _ in range(3)]
    model = EnsembleTwoStateModel(members)
    rows = [dict(mode_exchange=0., a=1.00000003, b=.25),
            dict(mode_exchange=1., a=1., b=.25), dict(mode_exchange=0., a=math.nan, b=0.)]
    expected = aggregate_probabilities(np.asarray([[RowPredictor(member).predict(row)
                                                   for row in rows] for member in members]), model.aggregate)
    assert predict_rows(model, rows) == expected.tolist()
    assert math.isnan(predict_rows(model, [dict(mode_exchange=2., a=0., b=0.)])[0])
    with pytest.raises(ValueError, match='mode_exchange'):
        predict_rows(model, [dict(mode_exchange=math.nan, a=0., b=0.)])


def test_batch_custom_model_keeps_row_api() -> None:
    from src.phase_j.live_light_batch import predict_rows
    model = SimpleNamespace(predict_row=lambda row: row['probability'])
    assert predict_rows(model, [{'probability': .2}, {'probability': .8}]) == [.2, .8]


def test_batch_calibrates_each_probability_before_averaging(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_24h
    selected = [dict(choice(1, .75), values={}), dict(choice(3, .25), values={})]
    monkeypatch.setattr(probability, 'prepared_rows', lambda *_: [{'score': .1}, {'score': .3}])
    monkeypatch.setattr(live_24h, 'apply_temperature', lambda value, row, temperature: value**2)
    owner = live_24h.Live24h.__new__(live_24h.Live24h)
    owner.model = SimpleNamespace(predict_row=lambda row: row['score'])
    owner.temperature, owner.fill_frac_columns = None, ()
    actual = probability.weighted_infer(owner, {}, {probability.SCENARIOS: selected})
    assert actual == pytest.approx(.03)


@pytest.mark.parametrize('predictions', [[], [math.nan]])
def test_batch_rejects_missing_or_invalid_predictions(predictions: list[float]) -> None:
    from src.phase_j.live_24h import Live24h
    owner = Live24h.__new__(Live24h)
    owner.model = SimpleNamespace(predict_light_rows=lambda _: predictions)
    owner.temperature, owner.fill_frac_columns = None, ()
    with pytest.raises(ValueError, match='有限な確率'):
        owner.infer_candidates([{'score': .1}], [{}])


def test_clone_does_not_share_mutable_state_or_executor() -> None:
    pool = ThreadPoolExecutor(max_workers=1)
    original = SimpleNamespace(native=math, pool=pool, knowledge=SimpleNamespace(lookup=None),
        state={'scores': [1]}, retr_ama=SimpleNamespace(proc=None, cache={'a': np.array([1])}))
    clone = clone_engine(original)
    try:
        clone.state['scores'].append(2)
        clone.retr_ama.cache['a'][0] = 2
        assert original.state == {'scores': [1]}
        assert original.retr_ama.cache['a'].tolist() == [1]
        assert clone.pool is not pool
    finally:
        pool.shutdown()
        clone.pool.shutdown()


def test_xc_snapshot_uses_fixed_forecast_and_keeps_main_state() -> None:
    from src.phase_j.live_light_ex import forecast_memo, FixedForecast
    from src.phase_j.live_ex_speed import snapshot
    class Owner:
        def __init__(self) -> None:
            self.forecast = SimpleNamespace(values=[1])
        def __deepcopy__(self, memo: dict) -> object:
            raise AssertionError('観測計算には物理候補の全コピーが不要')
        def current(self, now: float) -> object:
            return self.forecast
    class Tracker:
        def __init__(self) -> None:
            self.sides = [Owner()]
        def current(self, side: int, now: float) -> object:
            return self.sides[side].current(now)
    engine = SimpleNamespace(native=math, knowledge=SimpleNamespace(lookup=None),
        state=SimpleNamespace(t=1., values=[2]), retr_fire=Tracker(), chains=[])
    clone, _ = snapshot(engine, {}, forecast_memo(engine))
    assert isinstance(clone.retr_fire.sides[0], FixedForecast)
    clone.retr_fire.sides[0].forecast.values.append(3)
    clone.state.values.append(4)
    assert engine.retr_fire.sides[0].forecast.values == [1]
    assert engine.state.values == [2]


@pytest.mark.parametrize('latest,expected', [(20., .2), (math.nan, .8)])
def test_candidate_arrival_reselects_causal_snapshot_without_missing_pulse(latest: float, expected: float) -> None:
    from src.phase_j.live_24h import Live24h
    owner = Live24h.__new__(Live24h)
    owner.ama, owner.revision, owner.settings = True, 0, {}
    owner.ex_columns = ['xc_net_recv_p1']
    owner.clock = lambda: 0.
    old = {'_light_scenarios': [dict(choice(1, 1.), values={'xc_net_recv_p1': 10.})]}
    new = {'_light_scenarios': [dict(choice(1, 1.), values={'xc_net_recv_p1': latest})]}
    selections = iter([(old, 'old', 'available'), (new, 'new', 'available')])
    owner.selection = lambda result: next(selections)
    calls, logs = [], []
    def infer(row: dict, values: dict) -> float:
        calls.append(values['_light_scenarios'][0]['values']['xc_net_recv_p1'])
        owner.light_audit = {'reason': 'posterior_mean'}
        return .2 if math.isfinite(calls[-1]) else .8
    owner.infer, owner.write = infer, logs.append
    owner.flush_events = lambda: None
    owner.selection_audit = {}
    result = dict(row={}, frame=1, t_sec=1.)
    assert owner.predict(result) == expected
    assert calls[0] == 10.
    assert calls[1] == latest or math.isnan(calls[1]) and math.isnan(latest)
    assert owner.signature == 'new'
    assert owner.last['missing_reason'] == 'reselected_during_inference:available'
    assert logs[0]['learning'] is new


def test_candidate_values_survive_hold_without_mixing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    hold = LiveExHold()
    key = dict(key=(1, 1, (0., None)), fight=True)
    choices = [choice(1, .75), choice(3, .25)]
    source = dict(extra=choices[0]['extra'], input_t=0., eligible_at=10.2, light_scenarios=choices)
    first, _ = hold.available(10.2, .2, 10.2, key, source, 'available', 1)
    held, _ = hold.available(10.3, .3, 10.3, key, None, 'not_ready_or_stale', 1)
    for result in (first, held):
        rows = result['light_scenarios']
        assert [r['weight'] for r in rows] == [.75, .25]
        assert [r['extra'][LEARNING_COLUMNS[0]] for r in rows] == [1., 3.]
    hold.observe(dict(key, resets=[{'side': 'p1'}]))
    changed, _ = hold.available(10.4, .4, 10.4, key, None, 'not_ready_or_stale', 1)
    assert all(math.isnan(r['extra'][c]) for r in changed['light_scenarios']
               for c in LEARNING_COLUMNS if c.endswith('_p1'))


def test_disabled_hold_has_no_candidate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIGHT_ROBUST_CELLS', raising=False)
    hold = LiveExHold()
    key = dict(key=(1, 1, (0., None)), fight=True)
    row = choice(1, 1.)
    source = dict(extra=row['extra'], input_t=0., eligible_at=10.2, light_scenarios=[row])
    value, _ = hold.available(10.2, .2, 10.2, key, source, 'available', 1)
    assert 'light_scenarios' not in value and not hasattr(hold, '_light_holds')


def test_row_time_hold_does_not_copy_candidate_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_ex_row_time import RowTimeEx

    class SearchResult:
        def __deepcopy__(self, memo: dict) -> object:
            raise AssertionError('保持する列に使わない探索結果を複製した')

    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    history = RowTimeEx()
    history.boundary(1)
    key = dict(epoch=1, input_t=.2, key=(1, 1, (0., None)), fight=True)
    history.observe(key)
    payload = SearchResult()
    choices = [dict(choice(total, weight), result=payload)
               for total, weight in ((1, .5), (3, .3), (5, .2))]
    source = dict(extra=choices[0]['extra'], input_t=0., eligible_at=10.2,
                  serial=1, light_scenarios=choices)
    first, _ = history.held(10.2, .2, 10.2, source, 'available')
    held, _ = history.held(10.3, .3, 10.3, None, 'not_ready_or_stale')
    for result in (first, held):
        rows = result['light_scenarios']
        assert [r['weight'] for r in rows] == [.5, .3, .2]
        assert [r['extra'][LEARNING_COLUMNS[0]] for r in rows] == [1., 3., 5.]
    history.observe(dict(key, input_t=.4, resets=[{'side': 'p1'}]))
    reset, _ = history.held(10.4, .4, 10.4, None, 'not_ready_or_stale')
    assert all(math.isnan(r['extra'][c]) for r in reset['light_scenarios']
               for c in LEARNING_COLUMNS if c.endswith('_p1'))
    previous, _ = history.held(10.25, .25, 10.25, None, 'not_ready_or_stale')
    assert [r['extra'][LEARNING_COLUMNS[0]] for r in previous['light_scenarios']] == [1., 3., 5.]
    assert all(row['result'] is payload for row in source['light_scenarios'])


def test_b3_posterior_is_used_before_ex_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    choices = [choice(1, .75), choice(3, .25)]
    owner = SimpleNamespace(worker=SimpleNamespace(), settings={}, ex_columns=('feature',))
    result = probability.selected_values(owner, None, 1., {probability.CANDIDATES: choices})
    assert [item['weight'] for item in result] == [.75, .25]
    assert all(math.isnan(item['values']['feature']) for item in result)
    assert all(not item['ex_available'] for item in result)


def test_new_b3_weights_do_not_revert_to_old_ex_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    choices = [choice(1, .75), choice(3, .25)]
    received = [dict(choice(1, .1), extra={'feature': 10.}), dict(choice(3, .9), extra={'feature': 30.})]
    owner = SimpleNamespace(worker=SimpleNamespace(), settings={}, ex_columns=('feature',))
    result = probability.selected_values(owner, {'light_scenarios': received}, 1.,
                                         {probability.CANDIDATES: choices})
    assert [item['weight'] for item in result] == [.75, .25]
    assert [item['values']['feature'] for item in result] == [10., 30.]


def test_b3_row_uses_one_consistent_forecast_and_keeps_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.b3_realtime import RealtimeRows
    from src.b3_timing import TimingTable
    from src.b3_state import Episode
    from src.fire_board import ChainForecast, Forecast
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    rows = RealtimeRows(TimingTable())
    rows.state.t, rows.state.frame, rows.state.av_frame = 1.5, 90, 1.5
    side = rows.state.sides[0]
    side.episode = Episode(1., 1., steps=[(1., 4., 40.)])
    side.forecast = ChainForecast()
    board = np.zeros((13, 6), dtype=np.int8)
    side.forecast.fc = Forecast(1., (40,), (40, 320), 'refind', board, 1.)
    side._apply_forecast(1.5)
    side.episode.update_prediction()
    selected = dict(sides=({'chain_id': 1., 'steps': (40, 320, 900), 'final': board.tolist()}, None))
    monkeypatch.setattr(probability, 'scenarios', lambda owners: [selected])
    row: dict = {}
    probability.attach_context(rows, row)
    context = probability.pickle.loads(probability.zlib.decompress(
        probability.base64.b85decode(row[probability.CONTEXT])))
    result = probability.candidate_row(context, selected)
    assert result['chain_pred_score_p1'] == 1260
    assert result['chain_pred_l_p1'] == 3
    assert side.episode.pred_score == 360
    from copy import copy
    from src.b3_async import snapshot
    transported = copy(rows)
    transported.state, transported.trigger = probability.pickle.loads(snapshot(rows))
    direct_row, transported_row = rows.build(), transported.build()
    assert {k: repr(v) for k, v in direct_row.items() if k != probability.CONTEXT} == {
        k: repr(v) for k, v in transported_row.items() if k != probability.CONTEXT}
    selected['sides'][0]['chain_id'] = 2.
    assert probability.candidate_row(context, selected) is None
