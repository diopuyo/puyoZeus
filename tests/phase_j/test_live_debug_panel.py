"""試験用パネル(debug_panel)の追加欄: 既定OFFで公開値が変わらないこと、ONで追加欄が入ること。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.phase_j.live_debug_panel import (BOARD_COLS, BOARD_ROWS, CODES, DEBUG_PANEL_KEY, build_debug_panel,
                                          build_debug_panel_safe, encode_board, panel_enabled, with_debug_panel)
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from src.phase_j.overlay_config import validate_overlay
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS, row


def grid(fill: dict[tuple[int, int], int] | None = None) -> list[list[int]]:
    cells = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    for (r, c), value in (fill or {}).items():
        cells[r, c] = value
    return cells.tolist()


def test_board_codes_cover_every_color_and_special_cell() -> None:
    """赤1 青2 緑3 黄4 紫5 おじゃま9 空0 不明10 を区別し、想定外の値は不明へ寄せる。"""
    board = encode_board(grid({(12, 0): 1, (12, 1): 2, (12, 2): 3, (12, 3): 4, (12, 4): 5, (12, 5): 9, (0, 0): 10, (0, 1): 7}))
    assert board is not None and len(board) == BOARD_ROWS
    assert board[12] == '12345o'
    assert board[0] == '??....'
    assert board[5] == '......'
    assert set(CODES.values()) == {'.', '1', '2', '3', '4', '5', 'o', '?'}


@pytest.mark.parametrize('bad', [None, [], [[0] * BOARD_COLS] * 12, [[0] * 5] * BOARD_ROWS])
def test_wrong_shape_is_none(bad) -> None:
    assert encode_board(bad) is None


def side_frame(confirmed=None, reading=None) -> dict:
    return dict(confirmed_board=confirmed, raw_cnn_board=reading, state='STABLE', board_provenance='observed',
                score=1200, tsumo_count=9, formula=dict(valid=True, left=40, right=8))


def test_build_from_frame_keeps_confirmed_and_reading_apart() -> None:
    confirmed, reading = grid({(12, 0): 1}), grid({(12, 0): 2})
    frame = dict(frame=77, t_sec=3.5, sides=[side_frame(confirmed, reading), side_frame(None, reading)])
    panel = build_debug_panel(frame)
    assert panel['frame'] == 77 and panel['version'] == 1
    assert panel['p1']['confirmed'][12][0] == '1' and panel['p1']['reading'][12][0] == '2'
    assert panel['p2']['confirmed'] is None and panel['p2']['reading'][12][0] == '2'
    assert panel['p1']['formula'] == dict(left=40, right=8) and panel['p1']['score'] == 1200


def test_build_from_result_when_no_frame() -> None:
    board = SimpleNamespace(_grid=np.array(grid({(11, 3): 4})))
    side = SimpleNamespace(confirmed_board=board, state=SimpleNamespace(name='CHAIN'), board_provenance='observed', score=10)
    panel = build_debug_panel(None, SimpleNamespace(p1=side, p2=side))
    assert panel['p1']['confirmed'][11][3] == '4' and panel['p1']['state'] == 'CHAIN'
    assert panel['p1']['reading'] is None
    assert build_debug_panel(None, None) is None


def test_builder_failure_never_raises() -> None:
    assert build_debug_panel_safe(dict(sides=[object(), object()])) is None


def test_panel_enabled_reads_overlay_flag_only() -> None:
    enabled = lambda overlay: panel_enabled(SimpleNamespace(server=SimpleNamespace(overlay=overlay)))
    assert enabled({'debug_panel': True}) and not enabled({'debug_panel': False}) and not enabled({})
    assert not panel_enabled(SimpleNamespace())


def test_overlay_config_accepts_only_bool_and_defaults_off() -> None:
    assert validate_overlay({'layout': 'l', 'debug_panel': True})['debug_panel'] is True
    assert 'debug_panel' not in validate_overlay({'layout': 'l'})
    with pytest.raises(ValueError):
        validate_overlay({'debug_panel': 'yes'})


def panel_row() -> dict:
    return dict(row(), **{DEBUG_PANEL_KEY: build_debug_panel(dict(frame=1, t_sec=1.0, sides=[side_frame(grid()), side_frame(grid())]))})


@pytest.mark.parametrize('hold', [None, 1.3])
def test_payload_identical_except_added_field(hold: float | None) -> None:
    """パネルONでも、追加欄を除いた公開値(勝率・判定・決着など)はOFFと完全に同じ。"""
    initial = initial_snapshot(ASSETS)
    off = result_snapshot(initial, row(), 1.5, 1, hold)
    on = with_debug_panel(result_snapshot(initial, panel_row(), 1.5, 1, hold), panel_row())
    assert DEBUG_PANEL_KEY not in off.to_mapping()
    assert with_debug_panel(off, row()) is off
    added = on.to_mapping()
    assert added.pop(DEBUG_PANEL_KEY)['version'] == 1
    expected = off.to_mapping()
    for payload in (added, expected):
        payload['timing'].pop('published_at_utc')  # 壁時計の刻印だけは呼び出しごとに違う
    assert added == expected
    assert validate_snapshot(on).is_valid


def test_hub_publishes_added_field_without_fail_closed() -> None:
    from src.phase_j.snapshot_hub import SnapshotHub
    initial = initial_snapshot(ASSETS)
    hub = SnapshotHub(initial)
    snapshot = with_debug_panel(result_snapshot(initial, panel_row(), 1.5, 1, None), panel_row())
    published = hub.publish(snapshot)
    assert not published.fail_closed
    assert published.snapshot.to_mapping()[DEBUG_PANEL_KEY]['p1']['confirmed'][0] == '......'


def test_result_sink_adds_field_only_when_enabled() -> None:
    """ResultSink: OFFなら offer される row は従来と同じ。ONなら追加欄だけが増え、評価履歴には入らない。"""
    from scripts.run_live_pipeline_20260928 import ResultSink
    from src.phase_j.live_bridge import RecognitionBridge
    offered: list[dict] = []
    outputs = {}
    for enabled in (False, True):
        overlay = {'debug_panel': True} if enabled else {}
        publisher = SimpleNamespace(offer=offered.append, server=SimpleNamespace(overlay=overlay))
        sink = ResultSink(publisher)
        sink.bridge = RecognitionBridge(False, sink.observe)
        side = SimpleNamespace(state=SimpleNamespace(name='STABLE'), score=0, confirmed_board=None)
        notice = SimpleNamespace(frame=1, t_sec=1.0, captured_at=1.0, recognized_at=1.1,
                                 dropped_before=0, result_bytes=b'test', pipeline=None)
        overlay_obj = SimpleNamespace(tracker=SimpleNamespace(source='G_fe', probability=0.5, _static_probability=0.5))
        sink.observe(notice, 0.5, 0.0, overlay_obj, SimpleNamespace(p1=side, p2=side), 1, 0, 2.0)
        outputs[enabled] = (offered[-1], sink.rows[0])
    off_row, off_history = outputs[False]
    on_row, on_history = outputs[True]
    assert DEBUG_PANEL_KEY not in off_row and DEBUG_PANEL_KEY not in off_history
    assert DEBUG_PANEL_KEY not in on_history
    assert on_row.pop(DEBUG_PANEL_KEY)['p1']['state'] == 'STABLE'
    assert on_row == off_row


def test_publisher_latest_differs_only_by_added_field() -> None:
    """LivePublisher 経由: overlay.debug_panel がONで row に追加欄があるときだけ /latest 相当に載り、他は同じ。"""
    from src.phase_j.live_publish import LivePublisher
    latest = {}
    for enabled in (False, True):
        publisher = LivePublisher(ASSETS, '127.0.0.1', 0, overlay={'layout': 'l', 'debug_panel': enabled})
        publisher.offer(panel_row() if enabled else row())
        publisher._publish(publisher.latest, 1, None)
        latest[enabled] = publisher.state.latest()
    on, off = latest[True], latest[False]
    assert DEBUG_PANEL_KEY not in off and on.pop(DEBUG_PANEL_KEY)['version'] == 1
    for payload in (on, off):  # 起動ごとに変わる識別子・壁時計は比較から外す
        payload['identity'].pop('session_id'), payload['timing'].pop('published_at_utc')
        payload['identity'].pop('stream_seq', None)
        payload['evaluations']['practical'].pop('calculation_age_ms')  # 公開時の単調時計から決まる
    for part in ('evaluations', 'display', 'integrity', 'mode', 'input', 'terminal', 'runtime', 'identity'):
        assert on[part] == off[part], part
    assert on['timing'].keys() == off['timing'].keys()


# ---- 2026-10-13 デバッグ期間の既定 (L字 + 試験用パネル) ----

def test_debug_defaults_come_from_production_config_and_user_wins() -> None:
    """既定は production_config の1か所。puyo_live.json の overlay に書いた値は常に優先する。"""
    from src.phase_j.overlay_config import with_debug_defaults
    from src.production_config import LIVE_DEBUG_OVERLAY_VALUES
    assert dict(LIVE_DEBUG_OVERLAY_VALUES) == {'layout': 'l', 'debug_panel': True}
    assert with_debug_defaults({}) == {'layout': 'l', 'debug_panel': True}
    assert with_debug_defaults({'scale': 75})['scale'] == 75
    assert with_debug_defaults({'layout': 'card', 'debug_panel': False}) == {'layout': 'card', 'debug_panel': False}


def test_debug_defaults_switch_restores_previous_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """LIVE_DEBUG_OVERLAY_DEFAULT=False で従来 (既定OFF・中央カード=overlay 無指定) と完全一致。"""
    import src.production_config as config
    from src.phase_j.overlay_config import with_debug_defaults
    monkeypatch.setattr(config, 'LIVE_DEBUG_OVERLAY_DEFAULT', False)
    assert with_debug_defaults({}) == {} and with_debug_defaults({'layout': 'ribbon'}) == {'layout': 'ribbon'}


def test_live_server_overlay_default_is_l_with_panel() -> None:
    from src.phase_j.live_publish import LivePublisher
    assert LivePublisher(ASSETS, '127.0.0.1', 0).server.overlay == {'layout': 'l', 'debug_panel': True}
    explicit = LivePublisher(ASSETS, '127.0.0.1', 0, overlay={'layout': 'card'}).server.overlay
    assert explicit == {'layout': 'card', 'debug_panel': True}  # 配置は利用者指定、パネルは既定ON (L字以外では出ない)


def test_validate_overlay_itself_is_unchanged() -> None:
    """検証関数は省略キーを保持しない従来仕様のまま (既定は配信サーバ側でだけ足す)。"""
    assert validate_overlay({}) == {}


def test_served_html_carries_defaults() -> None:
    from src.phase_j.overlay_config import configured_html, with_debug_defaults
    html = configured_html('/*OVERLAY_DEFAULTS*/{}', with_debug_defaults({})).decode('utf-8')
    assert '"layout": "l"' in html and '"debug_panel": true' in html


def test_match_active_is_added_from_frame_and_result() -> None:
    frame = dict(frame=1, t_sec=.1, match_active=False, sides=[dict(state='MENU'), dict(state='MENU')])
    assert build_debug_panel(frame)['match_active'] is False
    side = SimpleNamespace(confirmed_board=None, state=SimpleNamespace(name='STABLE'), board_provenance=None, score=0)
    result = SimpleNamespace(p1=side, p2=side, is_match_active=True)
    assert build_debug_panel(None, result)['match_active'] is True
