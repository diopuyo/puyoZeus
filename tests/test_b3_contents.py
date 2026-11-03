"""B3の公開生値・側対応・採用世代・欠測の契約。"""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.b3_contents import contents_from_row
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS, row as live_row
from tests.test_b3_async_display import wrapper, result


def feature_row() -> dict:
    return dict(frame_idx=30, trigger_mask=1, t_sec=1., mode_exchange=1, sl_rate=70.,
        sl_pending_p1=12., sl_pending_p2=24., sl_generated_p1=100., sl_offset_p1=40.,
        sl_generated_p2=80., sl_offset_p2=20., ta_H_med_p1=2., ta_H_med_p2=5.,
        ta_q_med_p1=60., ta_q_med_p2=120., ta_m_sl_med_p1=36., ta_m_sl_med_p2=108.,
        ta_bound_p1=1., ta_bound_p2=0., mk_ta_p1=6, mk_ta_p2=0,
        chain_active_p1=1., chain_active_p2=0., chain_n_steps_p1=3.,
        chain_pred_l_p1=7., chain_rem_stages_p1=4., chain_rem_sec_p1=.123456789,
        k1_ojama_p1=30., cells_color_p1=48., cells_ojama_p1=6.)


def test_counts_receiver_direction_and_storage_precision() -> None:
    contents = contents_from_row(feature_row())
    assert contents['p1']['sl_pending'] == 12
    assert contents['p1']['sl_sent'] == 60
    assert contents['p2']['sl_sent'] == 60
    assert contents['p1']['ta_H_med'] == 5
    assert contents['p1']['ta_q_med'] == 120
    assert contents['p1']['ta_m_sl_med'] == 108
    assert contents['p2']['ta_H_med'] == 2
    assert contents['p2']['ta_bound'] == 1
    assert contents['p2']['unpredictable'] is True
    assert contents['p1']['chain_rem_sec'] == float(np.float32(.123456789))
    assert contents['model'] == 'exchange'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf')])
def test_missing_is_null_without_invented_zero(value: float | None) -> None:
    contents = contents_from_row(dict(feature_row(), sl_pending_p1=value))
    assert contents['p1']['sl_pending'] is None
    assert contents['p2']['chain_pred_l'] is None
    json.dumps(contents, allow_nan=False)


def test_sync_and_async_publish_identical_contents() -> None:
    source = feature_row()
    sync = wrapper()
    sync.model = SimpleNamespace(predict_row=lambda row: .8)
    sync.rows = SimpleNamespace(feature_ms=0., timings={})
    sync._predict(source, SimpleNamespace(frame=30, t_sec=1., captured_at=0.))
    asynchronous = wrapper()
    asynchronous._accept_result(result(mode=1, contents=contents_from_row(source)),
                                dict(game_idx=1, frame=30, t_sec=1.))
    published = []
    for live in (sync, asynchronous):
        live.display.update(1.)
        data = dict(live_row(), frame=30)
        live.decorate(data)
        dto = result_snapshot(initial_snapshot(ASSETS), data, 1.2, 30, None)
        assert validate_snapshot(dto).is_valid
        published.append(dto.to_mapping()['evaluations']['b3_contents'])
    assert published[0] == published[1]


@pytest.mark.parametrize('invalid', [dict(epoch=0), dict(game=9), dict(frame=61),
                                    dict(t_sec=-5.), dict(error='失敗')])
def test_rejected_result_does_not_replace_contents(invalid: dict) -> None:
    live = wrapper()
    expected = contents_from_row(feature_row())
    live._accept_result(result(contents=expected), dict(game_idx=1, frame=60, t_sec=1.4))
    live._accept_result(result(frame=40, contents={'wrong': True}, **invalid)
        if 'frame' not in invalid else result(contents={'wrong': True}, **invalid),
        dict(game_idx=1, frame=60, t_sec=1.4))
    assert live.display.contents == expected


@pytest.mark.parametrize('state', ['expired', 'boundary', 'terminal'])
def test_old_contents_not_published(state: str) -> None:
    live = wrapper()
    live._accept_result(result(contents=contents_from_row(feature_row())),
                        dict(game_idx=1, frame=30, t_sec=1.))
    if state == 'boundary':
        live.display.boundary(2)
    live.display.update(100. if state == 'expired' else 1.,
                        'confirmed_death' if state == 'terminal' else None, 1.)
    data = live_row()
    live.decorate(data)
    assert 'b3_contents' not in data


def test_optional_extension_preserves_v1_and_validates_types() -> None:
    initial = initial_snapshot(ASSETS)
    old = result_snapshot(initial, live_row(), 1.2, 1, None)
    assert validate_snapshot(old).is_valid
    data = dict(live_row(), b3_contents=contents_from_row(feature_row()))
    new = result_snapshot(initial, data, 1.2, 1, None)
    assert old.schema_version == new.schema_version == 'puyo-overlay-snapshot/v1'
    assert validate_snapshot(new).is_valid
    data['b3_contents']['p1']['sl_pending'] = '不正'
    assert not validate_snapshot(result_snapshot(initial, data, 1.2, 1, None)).is_valid


@pytest.mark.parametrize('value,valid', [(None, True), (1, True), (19, True),
                                      (0, False), (20, False), (1.5, False), ('14', False)])
def test_screen_stage_optional_schema(value: object, valid: bool) -> None:
    contents = contents_from_row(feature_row())
    contents['p1']['chain_display_step'] = value
    contents.update(chain_display_frame=60, chain_display_t_sec=2.)
    data = dict(live_row(), b3_contents=contents)
    dto = result_snapshot(initial_snapshot(ASSETS), data, 2., 60, None)
    assert validate_snapshot(dto).is_valid is valid


def test_async_snapshot_keeps_bound_episode_generation() -> None:
    """v7照合で見つかった6920フレームの上限補完差を退行防止する。"""
    from src.b3_async import snapshot
    from src.b3_timeaxis import _episode_generated

    step = SimpleNamespace(t_formula=2., generated=31.)
    chain = SimpleNamespace(side=0, steps=[step])
    core = SimpleNamespace(chains=[], st=[SimpleNamespace(chain=chain), SimpleNamespace(chain=None)])
    sides = [SimpleNamespace(name='1P', episode=SimpleNamespace(active=True, first_trigger=1.)),
             SimpleNamespace(name='2P', episode=None)]
    state = SimpleNamespace(sl=SimpleNamespace(core=core), sides=sides)
    expected = _episode_generated(state.sl, sides[0])
    frozen, trigger = pickle.loads(snapshot(SimpleNamespace(state=state, trigger=1)))
    step.generated = 99.
    assert _episode_generated(frozen.sl, frozen.sides[0]) == expected == 31.
    assert trigger == 1


def test_failed_newer_result_keeps_successful_content_frame() -> None:
    live = wrapper()
    live._accept_result(result(contents=contents_from_row(feature_row())),
                        dict(game_idx=1, frame=30, t_sec=1.))
    live._accept_result(result(frame=40, error='失敗'), dict(game_idx=1, frame=40, t_sec=1.2))
    live.display.update(1.2)
    data = dict(live_row(), frame=40, t_sec=1.2)
    live.decorate(data)
    assert data['b3_state_frame'] == data['b3_contents']['frame'] == 30
