"""段3 (live_x_prefire) の単体試験。合成盤で表・照合・拒否・隠し段・取り下げ・既定OFFの同一性を見る。

連鎖計算は実物の puyo_core (enumerate_placements_py と simulate_chain_trace_py / summary_py) を使う。
WSL の既定 venv の puyo_core は trace を持たないため、実行時は trace 入りの build を PYTHONPATH へ置く
(例: /mnt/c/puyo1_work/wt_exch_review/native/puyo_core/target/review_v36)。無い環境では skip ではなく失敗にする。
"""
from __future__ import annotations

from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any
import json

import numpy as np
import pytest

from src.phase_j import live_v49f_prefire as prefire
from src.phase_j.live_v49f_prefire import (PrefireTracker, SideView, build_table, match_board, match_reads,
                                           COLS, CELLS, RETRACT_SEC)

RED, BLUE, GREEN, YELLOW = 1, 2, 3, 4
PAIR = (RED, BLUE)
K0 = ((RED, BLUE), (GREEN, GREEN))
K1 = ((GREEN, GREEN), (YELLOW, YELLOW))   # K0 から滑った (新 NEXT1 = 旧 NEXT2)
K2 = ((YELLOW, YELLOW), (BLUE, BLUE))


@pytest.fixture(scope='module')
def native() -> Any:
    import puyo_core
    if not (hasattr(puyo_core, 'simulate_chain_trace_py') or hasattr(puyo_core, 'simulate_chain_summary_py')):
        pytest.fail('puyo_core に simulate_chain_trace_py / summary_py がありません (trace 入り build を PYTHONPATH へ)')
    return puyo_core


def cell(row: int, col: int) -> int:
    return row*COLS+col


def board(*cells: tuple[int, int, int]) -> tuple[int, ...]:
    grid = [0]*CELLS
    for row, col, color in cells:
        grid[cell(row, col)] = color
    return tuple(grid)


BASE = board((12, 0, RED), (12, 1, RED), (12, 2, RED))
FIRE = (*((12, c, RED) for c in range(3)), (12, 3, RED), (11, 3, BLUE))   # 縦置き col3、赤が下 → 4 個消え
QUIET = (*((12, c, RED) for c in range(3)), (12, 5, RED), (11, 5, BLUE))  # 縦置き col5 → 消えない


def view(state: str = 'STABLE', grid: tuple | None = None, key: tuple = K1, trigger: float | None = None,
         ledger: dict | None = None) -> SideView:
    return SideView(state, grid, key[0], trigger, ledger, key[1])


def armed(native: Any, base: tuple = BASE) -> tuple[PrefireTracker, list[dict]]:
    """K0 を確定 → 前の組が着地して盤が base に → K1 へ滑る。撃つ組 = (赤, 青)。"""
    tracker, events = PrefireTracker(native), []
    for t, v in ((0., view(grid=board(), key=K0)), (.1, view(grid=board(), key=K0)),
                 (.2, view(grid=base, key=K0)), (.3, view(grid=base)), (.4, view(grid=base))):
        events += tracker.observe(1, 0, t, v)
    return tracker, events


def kinds(events: list[dict]) -> list[str]:
    return [e['kind'] for e in events]


def test_table_lists_every_placement_and_fires(native: Any) -> None:
    table = build_table(BASE, PAIR, native)
    assert len(table.entries) == 22 and not table.base_fires
    fire = next(e for e in table.entries if e.added == ((cell(11, 3), BLUE), (cell(12, 3), RED)))
    assert (fire.chain, fire.score, fire.erased) == (1, 40, (4,))
    quiet = next(e for e in table.entries if e.added == ((cell(11, 5), BLUE), (cell(12, 5), RED)))
    assert quiet.chain == 0 and quiet.score == 0
    assert table.firing() == sum(e.chain > 0 for e in table.entries) > 0
    assert set(table.candidates) == {i for e in table.entries for i, _ in e.added}
    assert cell(12, 0) in table.supports and cell(11, 0) in table.candidates


def test_board_lookup_returns_entry_and_rejects_other_changes(native: Any) -> None:
    table = build_table(BASE, PAIR, native)
    entry, reason = match_board(table, board(*FIRE))
    assert reason == 'match' and entry.score == 40
    assert match_board(table, board(*FIRE[:3], (12, 3, RED), (11, 3, YELLOW)))[1] == 'color_mismatch'
    assert match_board(table, board(*FIRE, (12, 5, GREEN)))[1] == 'board_changed'


def reads_of(truth: tuple[int, ...], cells: tuple[int, ...]) -> dict[int, int]:
    return {i: truth[i] for i in cells}


def test_image_lookup_needs_both_classifiers_colors_and_support(native: Any) -> None:
    table = build_table(BASE, PAIR, native)
    cells = table.cells()
    good = reads_of(board(*FIRE), cells)
    entry, reason = match_reads(table, [good, good])
    assert reason == 'match' and (entry.chain, entry.score) == (1, 40)
    wrong = reads_of(board(*FIRE[:3], (12, 3, RED), (11, 3, YELLOW)), cells)
    assert match_reads(table, [wrong, wrong]) == (None, 'color_mismatch')
    assert match_reads(table, [good, reads_of(BASE, cells)]) == (None, 'classifier_disagree')
    stale = build_table(board(*BASE_CELLS, (12, 4, GREEN)), PAIR, native)
    floating = reads_of(board(*BASE_CELLS, (11, 4, RED), (10, 4, BLUE)), stale.cells())
    assert match_reads(stale, [floating, floating]) == (None, 'unsupported_air')


BASE_CELLS = ((12, 0, RED), (12, 1, RED), (12, 2, RED))


def pattern(rows: range, cols: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """隣り合う色が必ず違う詰め物 (色 2〜5、赤なし)。"""
    return [(r, c, 2+(r+c) % 4) for r in rows for c in cols]


def test_hidden_row_never_connects_and_is_not_matched(native: Any) -> None:
    # 可視の赤 (1,3)(1,4) + 隠し段の赤 (0,4)。col5 縦置きで赤を (1,5) に置くと、隠し段を数えれば 4 個。
    hidden = board(*pattern(range(2, 13), (3, 4, 5)), (1, 3, RED), (1, 4, RED), (0, 4, RED))
    table = build_table(hidden, PAIR, native)
    entry = next(e for e in table.entries if (cell(1, 5), RED) in e.added and e.hidden)
    assert entry.chain == 0
    placed = list(hidden)
    for i, color in entry.added:
        placed[i] = color
    assert native.simulate_chain_py(placed, False).chain_count == 1, '隠し段を数える物理なら消える (試験の感度)'
    assert all(i >= COLS for i in table.candidates)
    read = {i: placed[i] for i in table.cells()}
    assert match_reads(table, [read, read])[0] is None


def test_confirmed_landing_then_chain_confirms(native: Any) -> None:
    tracker, events = armed(native)
    assert kinds(events).count('table') == 1
    events = tracker.observe(1, 0, .6, view(grid=board(*FIRE)))
    assert kinds(events) == ['provisional'] and events[0]['score'] == 40 and events[0]['source'] == 'confirmed'
    assert tracker.sides[0].shown['status'] == 'pending'
    events = tracker.observe(1, 0, .8, view('CHAIN', board(*FIRE), trigger=.8))
    assert kinds(events) == ['confirm'] and events[0]['delay_sec'] == pytest.approx(.2)
    assert tracker.sides[0].shown['status'] == 'confirmed'


@pytest.mark.parametrize('step,reason', [
    ((.8, view(grid=board(*FIRE), key=K2)), 'next_moved'),
    ((.8, view('OJAMA_FALL', None)), 'ojama'),
    ((.6+RETRACT_SEC+.01, view(grid=board(*FIRE))), 'timeout'),
])
def test_retractions(native: Any, step: tuple, reason: str) -> None:
    tracker, _ = armed(native)
    tracker.observe(1, 0, .6, view(grid=board(*FIRE)))
    t, v = step
    events = tracker.observe(1, 0, t-.05, v) if reason == 'next_moved' else []
    events += tracker.observe(1, 0, t, v)
    retracts = [e for e in events if e['kind'] == 'retract']
    assert [e['reason'] for e in retracts] == [reason]
    assert retracts[0]['delay_sec'] == pytest.approx(t-.6)
    assert tracker.sides[0].shown is None and tracker.sides[0].pending is None


def test_image_landing_and_rejections(native: Any) -> None:
    tracker, _ = armed(native)
    truth = board(*FIRE)
    calls: list[tuple] = []

    def reader(*truths: tuple[int, ...]) -> prefire.Reader:
        def read(cells: tuple[int, ...], which: int) -> dict[int, int]:
            calls.append((len(cells), which))
            return reads_of(truths[which], cells)
        return read
    assert tracker.observe(1, 0, .45, view('TSUMO_FALL', None), reader(BASE, BASE)) == []
    assert [w for _, w in calls] == [0], '空中のあいだは median の着地候補だけ読む'
    events = tracker.observe(1, 0, .5, view('TSUMO_FALL', None), reader(truth, truth))
    assert kinds(events) == ['provisional'] and events[0]['source'] == 'image'
    tracker, _ = armed(native)
    wrong = board(*BASE_CELLS, (12, 3, RED), (11, 3, YELLOW))
    events = tracker.observe(1, 0, .5, view('TSUMO_FALL', None), reader(wrong, wrong))
    assert kinds(events) == ['reject'] and events[0]['reason'] == 'color_mismatch'
    events = tracker.observe(1, 0, .6, view('TSUMO_FALL', None), reader(truth, BASE))
    assert kinds(events) == ['reject'] and events[0]['reason'] == 'classifier_disagree'
    assert tracker.sides[0].pending is None


def test_quiet_landing_and_misread_base_do_not_fire(native: Any) -> None:
    tracker, _ = armed(native)
    events = tracker.observe(1, 0, .6, view(grid=board(*QUIET)))
    assert kinds(events) == ['landing'] and events[0]['chain'] == 0
    misread = board(*BASE_CELLS, (11, 0, RED))   # 確定盤そのものに 4 個 = 読み違い
    tracker, _ = armed(native, misread)
    events = tracker.observe(1, 0, .6, view(grid=board(*BASE_CELLS, (11, 0, RED), (12, 5, RED), (11, 5, BLUE))))
    assert kinds(events) == ['landing'] and events[0]['base_fires'] is True


def test_stale_base_waits_for_previous_landing(native: Any) -> None:
    tracker, events = PrefireTracker(native), []
    for t, v in ((0., view(grid=BASE, key=K0)), (.1, view(grid=BASE, key=K0)),
                 (.3, view(grid=BASE)), (.4, view(grid=BASE))):
        events += tracker.observe(1, 0, t, v)
    assert 'table' not in kinds(events) and tracker.sides[0].stale
    # 古い盤のまま表を作らない。着地盤が基準盤になるので、この組は引けない (誤発火より見逃しを選ぶ)。
    assert 'provisional' not in kinds(tracker.observe(1, 0, .6, view(grid=board(*FIRE))))


def test_unknown_slide_and_game_change_reset(native: Any) -> None:
    tracker, _ = armed(native)
    tracker.observe(1, 0, .6, view(grid=board(*FIRE)))
    odd = ((BLUE, BLUE), (RED, RED))   # 滑りでない NEXT 変化 = 読み違い
    events = tracker.observe(1, 0, .65, view(grid=board(*FIRE), key=odd))
    events += tracker.observe(1, 0, .7, view(grid=board(*FIRE), key=odd))
    assert [e['reason'] for e in events if e['kind'] == 'retract'] == ['next_moved']
    assert tracker.sides[0].consumed
    tracker, _ = armed(native)
    tracker.observe(1, 0, .6, view(grid=board(*FIRE)))
    events = tracker.observe(2, 0, .7, view(grid=board()))
    assert [e['reason'] for e in events if e['kind'] == 'retract'] == ['game']


def test_generated_uses_ledger_only_when_complete(native: Any) -> None:
    entry = next(e for e in build_table(BASE, PAIR, native).entries if e.chain)
    assert prefire.generated(entry, 10., None) is None
    assert prefire.generated(entry, 10., dict(carry=None, first_move=1.)) is None
    assert prefire.generated(entry, 10., dict(carry=30, first_move=1.)) == 1   # (30+40)//70


# ---- 既定OFFの同一性 (全列一致) ----

def raw_side(grid: tuple[int, ...], key: tuple) -> SimpleNamespace:
    from src.board import Board
    from src.board_state_machine import BoardState
    value = Board()
    value._grid = np.asarray(grid, dtype=np.uint8).reshape(13, COLS)
    return SimpleNamespace(state=BoardState.STABLE, confirmed_board=value, score=0, next_pair=key[0],
                           dnext_pair=key[1], next_slide_motion=False, chain_event=None)


class Images:
    def __init__(self, truth: tuple[int, ...]) -> None:
        self.truth, self.frames = truth, []

    def append(self, stamp: float, image: object) -> None:
        self.frames.append(stamp)

    def read_cells(self, image: object, side: int, cells: tuple[int, ...],
                   which: tuple[int, ...] = (0, 1)) -> list[dict[int, int]]:
        return [reads_of(self.truth, cells) for _ in which]


def capture_owner(prefire_capture: object | None) -> object:
    from src.phase_j.live_v49f_capture import CaptureV49F
    owner = CaptureV49F.__new__(CaptureV49F)
    owner.images, owner.policy, owner.game = Images(BASE), dict(first_move_quiet_sec=.1), None
    owner.prefire = prefire_capture
    owner.reset(None)
    return owner


def run_capture(owner: object) -> list[dict]:
    frames = [(0., board(), K0), (.1, board(), K0), (.2, BASE, K0), (.3, BASE, K1), (.4, BASE, K1),
              (.5, board(*FIRE), K1)]
    packets = []
    for t, grid, key in frames:
        result = SimpleNamespace(p1=raw_side(grid, key), p2=raw_side(board(), K0))
        packets.append(owner.capture(result, SimpleNamespace(media_sec=t, image=None),
                                     dict(game_idx=1, sides=[{}, {}]), SimpleNamespace(score_1p=0, score_2p=0)))
    return packets


def test_flags_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import production_config
    from src.phase_j import live_speed, launcher
    monkeypatch.delenv(prefire.ENV_SHADOW, raising=False)
    monkeypatch.delenv(prefire.ENV_DISPLAY, raising=False)
    assert prefire.flags() == (False, False)
    assert production_config.LIVE_X_PREFIRE_SHADOW_DEFAULT is False and production_config.LIVE_X_PREFIRE_DEFAULT is False
    for key in live_speed.PREFIRE_KEYS:
        assert live_speed.SPEED_DEFAULTS[key] is False and getattr(launcher.UserConfig, key) is False
    monkeypatch.setenv(prefire.ENV_DISPLAY, '1')
    assert prefire.flags() == (True, True)
    from src.phase_j.live_v49f_capture import CaptureV49F
    monkeypatch.delenv(prefire.ENV_DISPLAY)
    assert CaptureV49F.create_prefire(SimpleNamespace(images=None)) is None


def test_capture_packets_identical_except_prefire(native: Any) -> None:
    off = run_capture(capture_owner(None))
    capture = prefire.PrefireCapture(Images(BASE), native)
    capture.images.truth = board(*FIRE)
    on = run_capture(capture_owner(capture))
    assert all('prefire' not in p for p in off)
    assert [{k: v for k, v in p.items() if k != 'prefire'} for p in on] == off
    events = [e for p in on for e in p['prefire']['events']]
    assert 'provisional' in kinds(events)
    assert on[-1]['prefire']['shown'][0]['score'] == 40 and on[-1]['prefire']['shown'][1] is None


SHOWN = dict(side=0, landing_t=.9, source='image', col=3, rotation=0, score=40, chain=1, generated=None,
             erased=[4], scores=[40], pair=[RED, BLUE], status='pending')


def v49f_owner(view_: object | None) -> object:
    from tests.phase_j.test_live_v49f import view as v49f_view
    owner = v49f_view()
    owner.submitted, owner.inputs, owner.prefire = 0, Queue(4), view_
    return owner


def test_v49f_owner_strips_prefire_and_adds_separate_field(tmp_path: Path) -> None:
    packet = dict(t=1., game=1, raw=[], origins=[], capture_reason=None, input_epoch=(None, True))
    data = dict(t=1., events=[dict(kind='provisional', t=.9, **SHOWN), dict(kind='table', t=.5, build_ms=.4)],
                shown=[SHOWN, None], cost_ms=1.5, read_ms=.7, reads=2)
    off, on = v49f_owner(None), v49f_owner(prefire.PrefireView(tmp_path, display=True))
    off.update(dict(_v49f=packet, game_idx=1, frame=7), 1)
    on.update(dict(_v49f=dict(packet, prefire=data), game_idx=1, frame=7), 1)
    assert on.inputs.get_nowait() == off.inputs.get_nowait()
    row_off = dict(t_sec=1.25, raw_probability=.6, probability=.6, match_active=True, judgment_id=(1, 2))
    row_on = dict(row_off)
    off.decorate(row_off)
    on.decorate(row_on)
    field = row_on.pop('x_provisional')
    assert field['sides'][0]['score'] == 40 and field['sides'][0]['age_sec'] == pytest.approx(.35)
    assert field['sides'][1] is None
    assert row_on.pop('judgment_id')[:2] == row_off.pop('judgment_id') == (1, 2)
    assert row_on == row_off
    on.prefire.close()
    summary = json.loads((tmp_path/'live_x_prefire_summary.json').read_text(encoding='utf-8'))
    assert summary['counts'] == dict(provisional=1, table=1) and summary['cost_ms']['count'] == 1
    assert len((tmp_path/'live_x_prefire.jsonl').read_text(encoding='utf-8').splitlines()) == 1


def test_shadow_records_without_display(tmp_path: Path) -> None:
    owner = v49f_owner(prefire.PrefireView(tmp_path, display=False))
    data = dict(t=1., events=[], shown=[SHOWN, None], cost_ms=1., read_ms=0., reads=0)
    owner.update(dict(_v49f=dict(t=1., game=1, raw=[], origins=[], prefire=data), game_idx=1, frame=7), 1)
    row = dict(t_sec=1.25, raw_probability=.6, match_active=True, judgment_id=(1, 2))
    owner.decorate(row)
    assert 'x_provisional' not in row and row['judgment_id'] == (1, 2)
    owner.prefire.close()


def test_publish_copies_separate_field_only_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_publish import initial_snapshot, result_snapshot
    from src.phase_j.validator import validate_snapshot
    from tests.phase_j.test_live_publish import ASSETS, row
    monkeypatch.setattr('src.phase_j.live_publish.utc_now', lambda: '2026-10-22T00:00:00+00:00')
    initial = initial_snapshot(ASSETS)
    data = row()
    before = result_snapshot(initial, data, 4., 1, None)
    assert 'x_provisional' not in before.evaluations
    data['x_provisional'] = dict(label=prefire.LABEL, sides=[prefire.public(SHOWN, 1.), None])
    after = result_snapshot(initial, data, 4., 1, None)
    assert validate_snapshot(after).is_valid
    left, right = before.to_mapping(), after.to_mapping()
    assert right['evaluations'].pop('x_provisional') == data['x_provisional']
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_speed_and_launcher_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse
    import os
    from src.phase_j.live_speed import configure, SPEED_DEFAULTS
    from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError
    monkeypatch.setattr('os.environ', {})
    base = dict(SPEED_DEFAULTS, cpu_threads=1, source='dshow', b3_judgment=True, split_evaluation=True,
                worker_mode='process', compare=False)
    with pytest.raises(SystemExit):
        configure(argparse.Namespace(**dict(base, live_x_prefire=True, live_v49f=False)), argparse.ArgumentParser())
    configure(argparse.Namespace(**dict(base, live_x_prefire_shadow=True, live_v49f=True)), argparse.ArgumentParser())
    assert os.environ[prefire.ENV_SHADOW] == '1' and os.environ[prefire.ENV_DISPLAY] == '0'
    user = parse_user_config(dict(device_name='x', live_x_prefire=True), tmp_path)
    assert user.live_x_prefire is True and build_pipeline_config(user)['live_x_prefire'] is True
    assert 'live_x_prefire' not in build_pipeline_config(parse_user_config(dict(device_name='x'), tmp_path))
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(device_name='x', live_x_prefire='true'), tmp_path)


PALETTE = ((0, 0, 230), (0, 200, 0), (230, 60, 0), (0, 220, 230), (200, 0, 200), (20, 20, 20))


def test_read_cells_matches_existing_image_boards() -> None:
    """段3 の部分読取りは、既存の発火前画像票 (LiveImages.classify) と同じ規則・同じ値。"""
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    from src.phase_j.live_v49f_images import LiveImages
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for side, region in enumerate((DEFAULT_P1_REGION, DEFAULT_P2_REGION)):
        for i in range(COLS, CELLS):
            x1, y1, x2, y2 = region.cell_sample_rect(*divmod(i, COLS))
            image[y1:y2, x1:x2] = PALETTE[(i+side) % len(PALETTE)]
    live = LiveImages()
    for side in (0, 1):
        boards = [b['board'] for b in live.classify(image, 0., side)]
        reads = live.read_cells(image, side, tuple(range(CELLS)))
        assert [[r.get(i, 0) for i in range(CELLS)] for r in reads] == boards
        assert len({c for b in boards for c in b} - {0}) >= 2, '色が読めない合成画像では照合の意味がない'
        assert live.read_cells(image, side, (40, 41), (1,)) == [{40: boards[1][40], 41: boards[1][41]}]


def saved_side(state: str, trigger: float | None = None) -> dict:
    event = None if trigger is None else {'namespace': dict(trigger_sec=trigger, mechanism='formula_read',
                                                            chain_count=1, total_score=40)}
    return {'namespace': dict(state={'state': state}, confirmed_board=None, score=0, next_pair=None,
                              dnext_pair=None, next_slide_motion=False, chain_event=event)}


def test_replay_reader_tolerates_truncated_gzip(tmp_path: Path) -> None:
    import gzip
    from scripts.replay_live_x_prefire_21c import read_updates, landing_fires

    def update(t: float, first: dict, second: dict) -> dict:
        return dict(kind='update', args={'tuple': [{'namespace': dict(p1=first, p2=second)}, {}, {}, t, 1,
                                                   {'tuple': [0, 0]}, {'tuple': [0, 0]}, {'tuple': [False, False]}]})
    lines = [dict(kind='header'), update(1., saved_side('TSUMO_FALL'), saved_side('STABLE')),
             update(1.1, saved_side('STABLE'), saved_side('STABLE')),
             update(1.3, saved_side('CHAIN', 1.3), saved_side('STABLE'))]
    path = tmp_path/'inputs.jsonl.gz'
    text = '\n'.join(json.dumps(x) for x in lines)
    path.write_bytes(gzip.compress(text.encode('utf-8'))[:-8])  # gzip の末尾 (CRC・長さ) 欠け
    rows = read_updates(path)
    assert [r['t'] for r in rows] == [1., 1.1, 1.3]
    chain = dict(game=1, side=0, start=1.3, score=40)
    assert landing_fires(rows, [chain]) == [dict(chain, landing_t=1.1)]
