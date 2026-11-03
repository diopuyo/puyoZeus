"""決着公開の期限、境界、入力保留、OFF互換を実公開経路で固定する。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import time

import pytest

from src.phase_j.live_publish import LivePublisher
from src.phase_j.live_terminal_hold import TERMINAL_DISPLAY_HOLD_SEC
from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError
from tests.phase_j.test_live_publish import ASSETS, row


def terminal(game: int = 0, probability: float = 1.) -> dict:
    return dict(row('confirmed_death'), game=game, probability=probability,
                raw_probability=probability, advantage=probability*200-100)


def publish(owner: LivePublisher, data: dict, stamp: float) -> dict:
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=stamp):
        owner.offer(data)
        owner._publish(owner.latest, 0, None)
    return owner.state.latest()


def publisher(enabled: bool = True) -> LivePublisher:
    return LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True,
                         freshness_gate=True, terminal_display_hold=enabled)


def probability(data: dict) -> float | None:
    return data['evaluations']['practical']['p1_win_probability']


@pytest.mark.parametrize('p', [0., 1.])
@pytest.mark.parametrize('active', [True, False])
def test_menu_before_confirmation(p: float, active: bool) -> None:
    owner = publisher()
    data = dict(terminal(probability=p), match_active=active)
    original = deepcopy(data)
    result = publish(owner, data, 10.)
    assert probability(result) == p
    assert result['display']['message'] == f"決着・{'1P' if p else '2P'}勝利"
    assert result['display']['status'] == 'live'
    assert data == original


@pytest.mark.parametrize('phase', ['verifying', 'calibrating', 'no_puyo_screen'])
def test_same_device_epoch_hold_survives_and_expires(phase: str) -> None:
    owner = publisher()
    owner.input_pending(dict(phase='ready', progress=100, epoch=0, device_id='OBS', at=1.))
    publish(owner, terminal(), 10.)
    owner.input_pending(dict(phase=phase, progress=0, epoch=1, device_id='OBS', at=10.3))
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=10.4):
        owner._publish(owner.latest, 0, 10.3)
        assert probability(owner.server.sendable(owner.state.latest())) == 1.
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=15.):
        owner._publish(owner.latest, 0, 10.3)
    assert probability(owner.state.latest()) is None
    assert owner.state.latest()['display']['input_status'] == phase


@pytest.mark.parametrize('status', [dict(epoch=1, device_id='other'), dict(epoch=1)])
def test_device_change_rejects_inflight_old_confirmation(status: dict) -> None:
    owner = publisher()
    owner.input_pending(dict(phase='ready', progress=100, epoch=0, device_id='OBS', at=0.))
    publish(owner, terminal(), 10.)
    owner.input_pending(dict(phase='verifying', progress=0, at=10.2, **status))
    result = publish(owner, terminal(), 10.3)
    assert probability(result) is None


@pytest.mark.parametrize('gap', [0.01, 1., 2., 4.999, 5., 30.])
def test_next_game_first_publication_never_carries_terminal(gap: float) -> None:
    owner = publisher()
    publish(owner, terminal(), 10.)
    result = publish(owner, dict(row(), game=1, captured_at=10.+gap), 10.+gap)
    assert result['identity']['match_id'] == '1'
    assert probability(result) == .6
    # 古い試合の遅着でも次試合へ逆戻りさせない。
    result = publish(owner, terminal(), 10.+gap+.01)
    assert probability(result) == .6
    assert result['identity']['match_id'] == '1'


def test_repeated_confirmation_does_not_extend_or_rearm() -> None:
    owner = publisher()
    data = terminal()
    assert probability(publish(owner, data, 10.)) == 1.
    assert probability(publish(owner, dict(data, frame=2), 14.99)) == 1.
    for stamp in (15., 16., 100.):
        assert probability(publish(owner, dict(data, frame=int(stamp)), stamp)) is None
    assert probability(publish(owner, terminal(game=1), 101.)) is None
    publish(owner, dict(row(), game=1), 102.)
    assert probability(publish(owner, terminal(game=1), 103.)) == 1.


@pytest.mark.parametrize('first', [dict(raw_probability=None, source='null'),
                                 dict(match_active=False), {}])
def test_held_terminal_boundary_requires_normal_evaluation_before_rearming(first: dict) -> None:
    owner = publisher()
    publish(owner, terminal(), 10.)
    for stamp in (10.1, 11., 20.):
        assert probability(publish(owner, terminal(game=1), stamp)) is None
    publish(owner, dict(row(), game=1, **first), 21.)
    expected = None if first else 1.
    assert probability(publish(owner, terminal(game=1), 22.)) == expected


@pytest.mark.parametrize('p', [0., 1.])
def test_pending_boundary_null_then_first_confirmation_is_visible(p: float) -> None:
    """P1: 入力保留で番号が増え、通常評価ゼロでも本物の初確定を表示する。"""
    owner = publisher()
    owner.input_pending(dict(phase='ready', progress=100, epoch=0, device_id='OBS', at=1.))
    publish(owner, dict(row(), captured_at=10.), 10.)
    for stamp, phase in ((10.1, 'verifying'), (10.2, 'no_puyo_screen'), (10.3, 'ready')):
        owner.input_pending(dict(phase=phase, progress=0, epoch=1, device_id='OBS', at=stamp))
    for stamp in (10.4, 10.5):
        result = publish(owner, dict(row('null'), game=1, raw_probability=None,
                                     captured_at=stamp), stamp)
        assert probability(result) is None
    result = publish(owner, dict(terminal(game=1, probability=p), captured_at=11.), 11.)
    assert probability(result) == p
    assert result['display']['message'] == f"決着・{'1P' if p else '2P'}勝利"
    assert owner.terminal.deadline == 11. + TERMINAL_DISPLAY_HOLD_SEC


def test_unheld_final_confirmation_blocks_next_game_carryover() -> None:
    """10/11事故: 保持未成立でも旧最終行が確定なら次番号先頭の確定を隠す。"""
    owner = publisher()
    publish(owner, dict(terminal(), raw_probability=None), 10.)
    assert owner.terminal.deadline is None
    result = publish(owner, terminal(game=1), 10.1)
    assert probability(result) is None
    assert not owner.terminal.armed
    publish(owner, dict(row(), game=1), 11.)
    assert probability(publish(owner, terminal(game=1), 12.)) == 1.


@pytest.mark.parametrize('stamp', [13., 16.])
def test_held_history_blocks_carryover_after_nonconfirmed_final_row(stamp: float) -> None:
    """旧最終行がnullでも、期限切れを含む保持履歴で持越しを遮断する。"""
    owner = publisher()
    publish(owner, terminal(), 10.)
    publish(owner, dict(row('null'), raw_probability=None), stamp)
    assert not owner.terminal.last_conf
    assert probability(publish(owner, terminal(game=1), stamp+1.)) is None


def test_confirmation_survives_latest_overwrite_before_publish() -> None:
    owner = publisher()
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=10.):
        owner.offer(terminal())
    result = publish(owner, dict(row(), match_active=False), 10.1)
    assert probability(result) == 1.


def test_timeout_is_safe_after_device_invalidation() -> None:
    owner = publisher()
    owner.input_pending(dict(phase='ready', progress=100, epoch=0, device_id='OBS', at=0.))
    publish(owner, terminal(), 10.)
    owner.input_pending(dict(phase='verifying', progress=0, epoch=1, device_id='other', at=10.2))
    assert owner._wait_timeout() == 0.


def test_sse_never_resends_expired_or_previous_game_confirmation() -> None:
    owner = publisher()
    original = publish(owner, terminal(), 10.)
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=15.):
        assert owner.server.sendable(original) is None
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=10.1):
        owner.offer(dict(row(), game=1))
        assert owner.server.sendable(original) is None


def test_new_terminal_deadline_cannot_revive_queued_old_game() -> None:
    owner = publisher()
    original = publish(owner, terminal(), 10.)
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=10.1):
        owner.offer(dict(terminal(game=1, probability=0.), captured_at=10.))
        assert owner.server.sendable(original) is None


def test_same_judgment_id_does_not_hide_new_game_boundary() -> None:
    owner = publisher()
    publish(owner, dict(terminal(), judgment_id=(1,)), 10.)
    result = publish(owner, dict(row(), game=1, judgment_id=(1,)), 10.1)
    assert result['identity']['match_id'] == '1' and probability(result) == .6


def test_evaluation_error_in_new_game_clears_old_confirmation() -> None:
    from types import SimpleNamespace
    owner = publisher()
    publish(owner, terminal(), 10.)
    notice = SimpleNamespace(frame=2, t_sec=2., captured_at=10.1, recognized_at=10.2)
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=10.3):
        owner.evaluation_hold(notice, 1)
        owner._publish(owner.latest, 0, 10.3)
    assert probability(owner.state.latest()) is None
    assert owner.terminal.game == 1 and owner.terminal.row is None


def test_off_snapshot_matches_stateless_original_exactly() -> None:
    from src.phase_j.live_publish import result_snapshot
    owner = publisher(False)
    data = terminal()
    with patch('src.phase_j.live_publish.utc_now', return_value='2026-10-12T00:00:00+00:00'):
        result = publish(owner, data, 10.)
        expected = result_snapshot(owner.initial, data, 10.,
            result['identity']['reducer_revision'], None, freshness_gate=True).to_mapping()
    # SnapshotHubが付与する送信連番以外は旧DTOと同一。
    result['identity']['stream_seq'] = expected['identity']['stream_seq']
    assert result == expected


@pytest.mark.parametrize('source', ['S3_landing', 'unavoidable_death', 'exchange'])
def test_unconfirmed_sources_never_start_hold(source: str) -> None:
    owner = publisher()
    data = dict(terminal(), source=source, match_active=False)
    assert probability(publish(owner, data, 10.)) is None
    assert owner.terminal.deadline is None


@pytest.mark.parametrize('active', [False, True])
def test_off_keeps_original_publication(active: bool) -> None:
    owner = publisher(False)
    result = publish(owner, dict(terminal(), match_active=active), 100.)
    assert probability(result) == (1. if active else None)
    assert result['display'].get('message') != '決着・1P勝利'


@pytest.mark.parametrize('completed,evaluation', [(True, True), (False, True), (False, False)])
def test_no_new_rows_still_expires(completed: bool, evaluation: bool) -> None:
    owner = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=completed,
                          evaluation_driven=evaluation, terminal_display_hold=True)
    owner.start()
    try:
        stamp = time.perf_counter()
        owner.offer(dict(terminal(), captured_at=stamp, recognized_at=stamp, evaluated_at=stamp))
        deadline = stamp + TERMINAL_DISPLAY_HOLD_SEC + 1.
        seen = False
        while time.perf_counter() < deadline:
            latest = owner.state.latest()
            seen |= latest is not None and probability(latest) == 1.
            if seen and probability(latest) is None:
                break
            assert owner.error is None
            time.sleep(.02)
        assert seen and probability(owner.state.latest()) is None
    finally:
        owner.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_config_wiring(tmp_path: Path, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from scripts.run_live_pipeline_20260928 import parse_args
    config = build_pipeline_config(parse_user_config(
        dict(device_name='OBS', terminal_display_hold=enabled), tmp_path))
    assert config.get('terminal_display_hold', False) is enabled
    path = tmp_path/'pipeline.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['live', '--config', str(path)])
    assert parse_args().terminal_display_hold is enabled


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_flag_requires_bool(tmp_path: Path, value: object) -> None:
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(device_name='OBS', terminal_display_hold=value), tmp_path)
