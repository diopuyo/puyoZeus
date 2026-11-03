"""live_verify_soft_hold: 失敗の種類の分け方・状態を捨てない短い保留・期限・OFF の同一性を検証する。

設計: C:/puyo1_work/codex_jobs/verify_hold_split_20261024/DESIGN.md (§2 分類表、§3 仕様、§7.3 試験計画)。
実機・録画を使わない。外枠の実照合 (Hough) は合成画像で行う。
"""
from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src import production_config
from src.board import Board
from src.phase_j import live_device, live_verify_split as split
from src.phase_j.live_border_grace import BorderGraceVerifier
from src.phase_j.live_device import (DeviceConfig, DirectShowSource, border_pair, failed_edge_count,
                                     frame_border_matches)
from src.phase_j.live_device_session import DeviceSession
from src.phase_j.live_screen_worker import AsyncScreenState, ScreenResult, ScreenWorker
from tests.phase_j.test_live_calibration import samples
from tests.phase_j.test_live_device import BOARD_H, BOARD_W, CROP_MARGIN, _fake_verifier, _frame_crop

BORDER_ONE = split.VerifyFailure(split.STAGE_BORDER, '1P', 1)
BORDER_TWO = split.VerifyFailure(split.STAGE_BORDER, '1P', 2)
TRANSIENT = {split.KIND_KEY: split.KIND_TRANSIENT}
INPUT = {split.KIND_KEY: split.KIND_INPUT}
PRE_SOFT_DIAGNOSTIC_KEYS = {'transport_ready', 'normalized', 'continuity',
                            'verifier_fail_streak', 'verifier_fail_limit'}


# ---- A. 分類 (a)/(b) ----

@pytest.mark.parametrize('failure,ready,expected', [
    (BORDER_ONE, True, split.KIND_TRANSIENT),
    (split.VerifyFailure(split.STAGE_BORDER, '2P', 0), True, split.KIND_TRANSIENT),
    (BORDER_TWO, True, split.KIND_INPUT),
    (split.VerifyFailure(split.STAGE_BORDER, '1P', 3), True, split.KIND_INPUT),
    (split.VerifyFailure(split.STAGE_BORDER, '1P', None), True, split.KIND_INPUT),
    (split.VerifyFailure(split.STAGE_SCORE, '1P'), True, split.KIND_INPUT),
    (split.VerifyFailure(split.STAGE_GRID_POSITION, '1P'), True, split.KIND_INPUT),
    (split.VerifyFailure(split.STAGE_FEEDBACK), True, split.KIND_INPUT),
    (None, True, split.KIND_INPUT),
    (BORDER_ONE, False, split.KIND_INPUT),          # 入力断・別サイズ・繰り返し・復旧待ちは必ず (a)
    (Mock(), True, split.KIND_INPUT),               # 型が違うものは (a) へ倒す
])
def test_classification_table(failure: Any, ready: bool, expected: str) -> None:
    assert split.classify_failure(failure, ready) == expected


def test_score_can_be_added_to_transient_stages_in_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    score = split.VerifyFailure(split.STAGE_SCORE, '1P')
    assert split.classify_failure(score) == split.KIND_INPUT
    monkeypatch.setattr(split, 'TRANSIENT_STAGES', (split.STAGE_BORDER, split.STAGE_SCORE))
    assert split.classify_failure(score) == split.KIND_TRANSIENT


def test_constants_are_pinned_to_the_design_basis() -> None:
    """定数を変えるときは DESIGN.md §3.2 の再生で根拠を取り直す。"""
    assert split.SOFT_HOLD_MAX_SEC == 8.0
    assert split.MAX_TRANSIENT_FAILED_EDGES == 1
    assert split.TRANSIENT_STAGES == (split.STAGE_BORDER,)


def test_flag_default_off_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(split.FLAG_ENV, raising=False)
    assert production_config.LIVE_VERIFY_SOFT_HOLD_DEFAULT is False
    assert not split.enabled()
    for value, expected in (('1', True), ('0', False), ('true', False), ('', False)):
        monkeypatch.setenv(split.FLAG_ENV, value)
        assert split.enabled() is expected


def test_diagnostics_fields() -> None:
    assert split.diagnostics(BORDER_ONE, True) == {
        split.KIND_KEY: split.KIND_TRANSIENT,
        split.DETAIL_KEY: dict(stage='border', side='1P', failed_edges=1)}
    assert split.diagnostics(None, True) == {split.KIND_KEY: split.KIND_INPUT, split.DETAIL_KEY: None}
    assert split.diagnostics(BORDER_ONE, False)[split.KIND_KEY] == split.KIND_INPUT


def test_timer_limit_is_inclusive_and_restarts_after_clear() -> None:
    now = [100.0]
    timer = split.SoftHoldTimer(limit_sec=8.0, clock=lambda: now[0])
    assert timer.since is None and timer.within_limit() and timer.since == 100.0
    now[0] = 108.0
    assert timer.within_limit()
    now[0] = 108.01
    assert not timer.within_limit()
    timer.clear()
    assert timer.since is None and timer.within_limit() and timer.since == 108.01


# ---- B. 外枠の失敗辺の数 ----

def test_failed_edge_count_agrees_with_border_pair() -> None:
    full = [(20, 20, 20, 120), (120, 20, 120, 120)]
    one_missing = [(20, 20, 20, 120)]
    for lines, passes in ((full, True), (one_missing, False), ([], False)):
        assert border_pair(lines, 0, 100, 100) is passes
        assert (failed_edge_count(lines, 0, 100, 100) == 0) is passes
    assert failed_edge_count(one_missing, 0, 100, 100) == 1
    assert failed_edge_count([], 0, 100, 100) == 2
    assert failed_edge_count([], 0, 100, 100, exempt_first=True) == 1


@pytest.mark.parametrize('missing,exempt_top,matched,edges', [
    ((), False, False, 1),                  # 上辺だけ無い (4 辺要求)
    ((), True, True, None),                 # 上辺免除なら合格 (失敗辺は記録しない)
    (('left',), True, False, 1),
    (('bottom',), True, False, 1),
    (('right',), True, False, 1),
    (('left', 'bottom'), True, False, 2),
    (('left', 'bottom', 'right'), True, False, 3),
])
def test_frame_border_failed_edges_with_synthetic_frame(missing: tuple, exempt_top: bool,
                                                        matched: bool, edges: int | None) -> None:
    crop = _frame_crop(top=False, **{name: False for name in missing})
    out: list[int] = []
    assert frame_border_matches(crop, BOARD_W, BOARD_H, exempt_top=exempt_top,
                                failed_edges_out=out) is matched
    assert out == ([] if edges is None else [edges])
    assert frame_border_matches(crop, BOARD_W, BOARD_H, exempt_top=exempt_top) is matched   # 引数なしは従来どおり


@pytest.mark.parametrize('exempt_top,edges', [(False, 4), (True, 3)])
def test_blank_crop_fails_every_checked_edge(exempt_top: bool, edges: int) -> None:
    blank = np.zeros((BOARD_H+2*CROP_MARGIN, BOARD_W+2*CROP_MARGIN, 3), np.uint8)
    out: list[int] = []
    assert not frame_border_matches(blank, BOARD_W, BOARD_H, exempt_top=exempt_top, failed_edges_out=out)
    assert out == [edges]


# ---- C. PuyoScreenVerifier の失敗記録 ----

def _verifier(record: bool, score: int | None = 1234, grid: Any = None) -> Any:
    verifier = _fake_verifier(score, False, relaxed=True)
    verifier.grid = SimpleNamespace(detect=lambda image: grid)
    verifier._record_failure = record
    return verifier


IMAGE = np.zeros((1080, 1920, 3), np.uint8)
DISPLACED_GRID = SimpleNamespace(top_left=(500, 500), bottom_right=(900, 900))


def _border_stub(edges: int | None, matched: bool):
    def stub(crop: Any, width: int, height: int, exempt_top: bool = False,
             failed_edges_out: list[int] | None = None) -> bool:
        if failed_edges_out is not None and edges is not None:
            failed_edges_out.append(edges)
        return matched
    return stub


@pytest.mark.parametrize('record', [True, False])
@pytest.mark.parametrize('case,expected', [
    ('feedback', (split.STAGE_FEEDBACK, '', None)),
    ('score', (split.STAGE_SCORE, '1P', None)),
    ('grid_position', (split.STAGE_GRID_POSITION, '1P', None)),
    ('border', (split.STAGE_BORDER, '1P', 1)),
    ('pass', None),
])
def test_verifier_records_failure_stage_only_when_on(case: str, expected: tuple | None, record: bool,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_input
    monkeypatch.setattr(distribution_input, 'has_feedback', lambda image: case == 'feedback')
    monkeypatch.setattr(live_device, 'frame_border_matches', _border_stub(1, case == 'pass'))
    verifier = _verifier(record, score=None if case == 'score' else 1234,
                         grid=DISPLACED_GRID if case == 'grid_position' else None)
    assert verifier(IMAGE) is (case == 'pass')       # 合否は ON/OFF で同じ
    if not record:
        assert 'failure' not in vars(verifier)       # OFF は何も残さない
        return
    assert verifier.failure == (None if expected is None else split.VerifyFailure(*expected))


def test_verifier_clears_previous_failure_on_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _verifier(True)
    monkeypatch.setattr(live_device, 'frame_border_matches', _border_stub(1, False))
    assert not verifier(IMAGE) and verifier.failure == BORDER_ONE
    monkeypatch.setattr(live_device, 'frame_border_matches', _border_stub(None, True))
    assert verifier(IMAGE) and verifier.failure is None


def test_off_calls_border_check_exactly_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    def spy(crop: Any, width: int, height: int, exempt_top: bool = False, **kwargs: Any) -> bool:
        calls.append(kwargs)
        return True
    monkeypatch.setattr(live_device, 'frame_border_matches', spy)
    _verifier(False)(IMAGE)
    assert calls == [{}, {}]                          # 追加の引数なし (従来と同じ呼び出し)
    calls.clear()
    _verifier(True)(IMAGE)
    assert all('failed_edges_out' in call for call in calls)


def test_real_hough_failure_edges_reach_the_classifier() -> None:
    """合成の 1080p 画像で、実際の外枠照合の失敗辺の数が (a)/(b) の判定へ届く。"""
    def frame(missing: tuple[str, ...]) -> np.ndarray:
        image = np.zeros((1080, 1920, 3), np.uint8)
        x0, y0, x1, y1 = 100, 100, 100+BOARD_W, 100+BOARD_H
        for name, a, b in (('left', (x0, y0), (x0, y1)), ('right', (x1, y0), (x1, y1)),
                           ('bottom', (x0, y1), (x1, y1))):
            if name not in missing:
                cv2.line(image, a, b, (255, 255, 255), 3)
        return image
    for missing, kind in ((('left',), split.KIND_TRANSIENT),
                          (('left', 'bottom'), split.KIND_INPUT),
                          (('left', 'bottom', 'right'), split.KIND_INPUT)):
        verifier = _verifier(True)
        assert not verifier(frame(missing))
        assert verifier.failure.stage == split.STAGE_BORDER
        assert split.classify_failure(verifier.failure) == kind


# ---- D. 保留の診断・非同期経路・猶予 ----

def _source(monkeypatch: pytest.MonkeyPatch, flag: str) -> DirectShowSource:
    monkeypatch.setenv(split.FLAG_ENV, flag)
    return DirectShowSource(DeviceConfig('OBS', 0, verify_fail_streak=3), 1, Mock(),
                            sleep=lambda seconds: None, on_status=Mock())


@pytest.mark.parametrize('flag', ['0', '1'])
@pytest.mark.parametrize('ready', [True, False])
def test_screen_hold_diagnostics_gain_keys_only_when_on(flag: str, ready: bool,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(monkeypatch, flag)
    source._screen_hold(ready, None, None, 3, SimpleNamespace(failure=BORDER_ONE, feedback=False))
    keys = set(source.input_diagnostics)
    if flag == '0':
        assert keys == PRE_SOFT_DIAGNOSTIC_KEYS
        return
    assert keys == PRE_SOFT_DIAGNOSTIC_KEYS | {split.KIND_KEY, split.DETAIL_KEY}
    assert source.input_diagnostics[split.KIND_KEY] == (split.KIND_TRANSIENT if ready else split.KIND_INPUT)


def test_screen_hold_without_failure_is_input_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(monkeypatch, '1')
    source._screen_hold(True, None, None, 3, lambda image: False)
    assert source.input_diagnostics[split.KIND_KEY] == split.KIND_INPUT


def _completed(state: AsyncScreenState, passed: bool, failure: Any = None) -> None:
    state.worker.future = Future()
    state.worker.pending_generation = state.worker.generation
    state.worker.future.set_result(ScreenResult(state.worker.generation, 0., passed, False, .08,
                                                failure=failure))


def _judge(passed: bool, verified: bool, streak: int) -> tuple[bool, int]:
    return (True, 0) if passed else (verified and streak+1 < 3, streak+1)


def test_async_state_carries_failure_and_clears_it() -> None:
    small = np.zeros((2, 2, 3), np.uint8)
    state = AsyncScreenState(lambda _: True, _judge, 1.)
    try:
        state.last_verify = 0.
        _completed(state, True)
        assert state.update(small, (2, 2), .2, True) == (True, 0) and state.failure is None
        _completed(state, False, BORDER_ONE)
        assert state.update(small, (2, 2), .4, True) == (True, 1) and state.failure == BORDER_ONE
        _completed(state, True)
        state.update(small, (2, 2), .6, True)
        assert state.failure is None
        _completed(state, False, BORDER_ONE)
        state.update(small, (2, 2), .8, True)
        state.update(small, (2, 2), .9, False)          # 入力断で捨てる
        assert state.failure is None
    finally:
        state.close()


def test_worker_copies_failure_only_for_failed_results_of_the_right_type() -> None:
    def make(passed: bool, failure: Any) -> ScreenResult:
        verifier = Mock(side_effect=lambda image: passed)
        verifier.failure, verifier.feedback, verifier.diagnostics = failure, False, None
        worker = ScreenWorker(verifier)
        try:
            return worker._verify(np.zeros((2, 2, 3), np.uint8), 0., 0, (2, 2))
        finally:
            worker.close()
    assert make(False, BORDER_ONE).failure == BORDER_ONE
    assert make(True, BORDER_ONE).failure is None        # 合格時は残さない
    assert make(False, Mock()).failure is None           # 型が違えば残さない
    assert make(False, None).failure is None


@pytest.mark.parametrize('flag', ['0', '1'])
def test_screen_result_record_keeps_old_keys_when_off(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(monkeypatch, flag)
    state = SimpleNamespace(latest_result=ScreenResult(0, 0., False, False, .1, None, BORDER_ONE),
                            wait_initial=True, update=lambda *args: (True, 1))
    seen: list[dict] = []
    monkeypatch.setattr(live_device.timing, 'detail', lambda **kwargs: seen.append(kwargs['screen_result']))
    source._async_check(state, None, None, 0., True)
    old = {'generation', 'captured_at', 'passed', 'feedback', 'duration_sec', 'diagnostics'}
    assert set(seen[0]) == (old if flag == '0' else old | {'failure'})


def test_grace_wrapper_forwards_failure() -> None:
    base = Mock(return_value=False)
    base.failure, base.feedback, base.regions = BORDER_ONE, False, ()
    wrapper = BorderGraceVerifier(base)
    assert wrapper.verify(np.zeros((2, 2, 3), np.uint8), 1., (2, 2), 0) is False
    assert wrapper.failure == BORDER_ONE


# ---- E. DeviceSession ----

def make_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, *,
                 fast_first: bool = True, match_active: bool = True) -> DeviceSession:
    monkeypatch.setenv('PUYO_CALIBRATION_DIR', str(tmp_path/'calibration'))
    monkeypatch.setenv(split.FLAG_ENV, flag)
    hsv = SimpleNamespace(_ranges={'base': 1}, _native_ranges_cache=None, _native_params_cache=None,
                          set_color_ranges_from_simple=Mock())
    classifier = SimpleNamespace(_hsv=hsv, predict_proba_and_hsv_grid=Mock(return_value=(None, None)))
    pipe = SimpleNamespace(reset=Mock(), update=Mock(), _online_hsv=None,
                           _reader=SimpleNamespace(_classifier=classifier))
    pipe.update.return_value = SimpleNamespace(is_match_active=match_active,
        p1=SimpleNamespace(state=SimpleNamespace(name='STABLE'), confirmed_board=Board()),
        p2=SimpleNamespace(state=SimpleNamespace(name='STABLE'), confirmed_board=Board()))
    instance = DeviceSession(pipe, tmp_path/'input.json', 1, Queue())
    instance.switch(DeviceConfig('OBS', 0, fast_first_judgment=fast_first))
    samples(instance.warmup, instance.warmup.target)
    instance.warmup.advance(classifier)
    return instance


def at(session: DeviceSession, clock: list[float]) -> DeviceSession:
    """期限の時計を差し替える (ON のときだけタイマーがある)。"""
    if session._soft_hold is not None:
        session._soft_hold = split.SoftHoldTimer(clock=lambda: clock[0])
    return session


def trace(session: DeviceSession) -> list[tuple]:
    """送出した状態通知を、時刻 (perf_counter) を除いて取り出す。"""
    rows = []
    while not session.queue.empty():
        kind, body = session.queue.get_nowait()
        rows.append((kind, {key: value for key, value in body.items() if key != 'at'}))
    return rows


def hold_times(session: DeviceSession, diagnostics: dict, count: int = 3, phase: str = 'no_puyo_screen') -> None:
    session.remember_input_diagnostics(diagnostics)
    for _ in range(count):
        session.hold(phase)


def test_off_resets_even_for_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '0')
    base = session.pipe.reset.call_count
    session.update(1, 0., None)
    hold_times(session, TRANSIENT)
    assert session.pipe.reset.call_count == base+1 and session.epoch == 1
    assert session._soft_hold is None and '_in_match' not in vars(session)   # OFF は何も持たない


def test_on_transient_in_match_keeps_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    base, epoch, warmup = session.pipe.reset.call_count, session.epoch, session.warmup
    session.update(1, 0., None)
    assert session._in_match
    hold_times(session, TRANSIENT)
    assert session.pipe.reset.call_count == base and session.epoch == epoch
    assert session.warmup is warmup and not session.publishing_ready
    assert session._soft_hold.since is not None


def test_on_keeps_state_without_discarding_calibration_on_legacy_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fast_first_judgment=false の従来経路は reset に加えて較正も捨てる。soft hold はどちらも抑止する。"""
    for flag, resets in (('0', 1), ('1', 0)):
        session = make_session(tmp_path/flag, monkeypatch, flag, fast_first=False)
        base, warmup = session.pipe.reset.call_count, session.warmup
        session.update(1, 0., None)
        hold_times(session, TRANSIENT)
        assert session.pipe.reset.call_count == base+resets
        assert (session.warmup is warmup) is (resets == 0)


def test_deadline_then_legacy_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    session = at(make_session(tmp_path, monkeypatch, '1'), clock)
    base, epoch = session.pipe.reset.call_count, session.epoch
    session.update(1, 0., None)
    session.remember_input_diagnostics(TRANSIENT)
    for now in (0.0, 4.0, split.SOFT_HOLD_MAX_SEC):           # 期限ちょうどまでは保留
        clock[0] = now
        session.hold('no_puyo_screen')
        assert session.pipe.reset.call_count == base and session.epoch == epoch
    clock[0] = split.SOFT_HOLD_MAX_SEC+0.01                   # 超えたら (a) と同じ扱い
    session.hold('no_puyo_screen')
    assert session.pipe.reset.call_count == base+1 and session.epoch == epoch+1
    assert not session._in_match and session._soft_hold.since is None
    clock[0] += 5
    session.hold('no_puyo_screen')                             # 以後の保留は従来の「初回だけ reset」のまま
    assert session.pipe.reset.call_count == base+1 and session.epoch == epoch+1


@pytest.mark.parametrize('diagnostics', [INPUT, {}, {split.KIND_KEY: 'unknown'}])
def test_input_kind_or_missing_diagnostics_reset_like_before(
        diagnostics: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    base = session.pipe.reset.call_count
    session.update(1, 0., None)
    hold_times(session, diagnostics)
    assert session.pipe.reset.call_count == base+1 and session.epoch == 1


def test_outside_match_resets_like_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1', match_active=False)
    base = session.pipe.reset.call_count
    session.update(1, 0., None)
    assert not session._in_match
    hold_times(session, TRANSIENT)
    assert session.pipe.reset.call_count == base+1 and session.epoch == 1


def test_never_updated_session_is_not_in_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    base = session.pipe.reset.call_count
    hold_times(session, TRANSIENT)                              # 認識結果がまだ無い = 試合中と言えない
    assert session.pipe.reset.call_count == base+1


def test_change_to_input_kind_during_hold_resets_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    base = session.pipe.reset.call_count
    session.update(1, 0., None)
    hold_times(session, TRANSIENT, count=2)
    assert session.pipe.reset.call_count == base
    hold_times(session, INPUT, count=1)                         # 外枠が複数辺とも見えなくなった等
    assert session.pipe.reset.call_count == base+1 and session.epoch == 1


def test_recovery_restarts_the_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    session = at(make_session(tmp_path, monkeypatch, '1'), clock)
    base = session.pipe.reset.call_count
    session.update(1, 0., None)
    hold_times(session, TRANSIENT, count=1)
    assert session._soft_hold.since == 0.0
    clock[0] = 7.0
    session.update(2, 7., None)                                 # 復帰
    assert session._soft_hold.since is None and session.publishing_ready
    clock[0] = 10.0
    hold_times(session, TRANSIENT, count=1)                     # 次の保留は新しい起点 (10 秒) から
    assert session._soft_hold.since == 10.0
    clock[0] = 17.5
    hold_times(session, TRANSIENT, count=1)                     # 7.5 秒目: まだ期限内
    assert session.epoch == 0 and session.pipe.reset.call_count == base


def test_eligible_hold_does_not_advance_the_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    session = at(make_session(tmp_path, monkeypatch, '1'), clock)
    base = session.pipe.reset.call_count
    session.update(1, 0., None)
    session.remember_input_diagnostics(TRANSIENT)
    session.hold('no_puyo_screen')                              # まず soft hold で起点が 0 秒に付く
    assert session._soft_hold.since == 0.0
    session._match_end_hold_eligible = True                     # 1fd4b1cf: ゲーム画面らしい保留は無期限に抑止
    for now in (30.0, 60.0, 90.0):
        clock[0] = now
        session.hold('no_puyo_screen')
        assert session.pipe.reset.call_count == base and session._soft_hold.since is None
    session._match_end_hold_eligible = False                    # eligible が外れたら、その時刻から新しく数える
    clock[0] = 91.0
    session.hold('no_puyo_screen')
    assert session._soft_hold.since == 91.0 and session.pipe.reset.call_count == base


def test_hold_before_calibration_is_untouched_by_soft_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """較正前は従来も reset しない保留。soft hold は関与せず、期限の起点も付けない。"""
    monkeypatch.setenv('PUYO_CALIBRATION_DIR', str(tmp_path/'calibration'))
    monkeypatch.setenv(split.FLAG_ENV, '1')
    pipe = SimpleNamespace(reset=Mock(), update=Mock(), _online_hsv=None,
        _reader=SimpleNamespace(_classifier=SimpleNamespace(_hsv=SimpleNamespace(
            _ranges={}, _native_ranges_cache=None, _native_params_cache=None))))
    session = DeviceSession(pipe, tmp_path/'input.json', 1, Queue())
    session.switch(DeviceConfig('OBS', 0, fast_first_judgment=True))
    base = pipe.reset.call_count
    assert not session.warmup.ready and session.warmup.progress == 0
    session._in_match = True
    hold_times(session, TRANSIENT)
    assert pipe.reset.call_count == base and session._soft_hold.since is None and session.epoch == 0


@pytest.mark.parametrize('diagnostics,match_active', [(INPUT, True), ({}, True), (TRANSIENT, False)])
def test_on_but_not_applicable_is_identical_to_off(diagnostics: dict, match_active: bool,
                                                    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ON でも (a)・診断なし・試合外は、reset 回数・epoch・較正・送出した通知が OFF と同一。"""
    results = []
    for flag in ('0', '1'):
        session = make_session(tmp_path/flag, monkeypatch, flag, match_active=match_active)
        trace(session)
        session.update(1, 0., None)
        hold_times(session, diagnostics, count=4)
        session.update(2, 1., None)
        results.append((session.pipe.reset.call_count, session.epoch, session.publishing_ready,
                        session.warmup.ready, trace(session)))
    assert results[0] == results[1]


def test_reset_paths_forget_the_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    session.update(1, 0., None)
    assert session._in_match
    session.switch(session.warmup.device)                       # 機器切替・入力喪失後の再初期化
    assert not session._in_match and session._soft_hold.since is None


# ---- F. 通し (DirectShowSource の保留 → DeviceSession) ----

class MovingDevice:
    """毎コマ画素を変える疑似カメラ (同一画像の繰り返しを入力断とみなす監視を避ける)。"""
    def __init__(self) -> None:
        self.count = 0

    def isOpened(self) -> bool:
        return True

    def set(self, key: int, value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self.count += 1
        image = np.zeros((1080, 1920, 3), np.uint8)
        image[0, 0, 0] = self.count % 256
        return True, image

    def release(self) -> None:
        pass


class ScriptedVerifier:
    """1 秒ごとの確認結果を順に返す。None = 合格。"""
    feedback = False

    def __init__(self, script: list) -> None:
        self.script, self.failure = iter(script), None

    def __call__(self, image: Any) -> bool:
        failure = next(self.script, None)
        self.failure = failure
        return failure is None



def run_through(session: DeviceSession, script: list, duration: float) -> None:
    """確認の結果列を DirectShowSource → ConfiguredSource.hold と同じ順で DeviceSession へ通す。"""
    now = [0.0]
    at(session, now)                    # 実機では入力の時計と期限の時計は同じ perf_counter
    source: list[DirectShowSource] = []
    def on_status(phase: str) -> None:
        session.remember_input_diagnostics(dict(source[0].input_diagnostics))
        session.hold(phase)
    source.append(DirectShowSource(DeviceConfig('OBS', 0, verify_fail_streak=3), duration, lambda: None,
        lambda *args: MovingDevice(), ScriptedVerifier(script), lambda: now[0],
        lambda seconds: now.__setitem__(0, now[0]+seconds), on_status=on_status))
    for frame in source[0]:
        session.update(frame.index, frame.media_sec, None)


@pytest.mark.parametrize('flag', ['0', '1'])
def test_through_short_one_edge_failure(flag: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """外枠 1 辺の不足が 3 回続いて保留に入り、数秒で戻る。ON は状態を捨てず、OFF は従来どおり 1 回 reset する。"""
    session = make_session(tmp_path, monkeypatch, flag)
    base = session.pipe.reset.call_count
    run_through(session, [None, None] + [BORDER_ONE]*5, duration=12.0)
    assert session.pipe.reset.call_count == base+(0 if flag == '1' else 1)
    assert session.epoch == (0 if flag == '1' else 1)
    assert session.publishing_ready                                  # 復帰後は公開を再開


def test_through_long_one_edge_failure_hits_the_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    base = session.pipe.reset.call_count
    run_through(session, [None, None] + [BORDER_ONE]*14, duration=24.0)
    assert session.pipe.reset.call_count == base+1 and session.epoch == 1   # 期限超過 = 従来の reset


def test_through_two_edge_failure_is_input_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session(tmp_path, monkeypatch, '1')
    base = session.pipe.reset.call_count
    run_through(session, [None, None] + [BORDER_TWO]*5, duration=12.0)
    assert session.pipe.reset.call_count == base+1 and session.epoch == 1
