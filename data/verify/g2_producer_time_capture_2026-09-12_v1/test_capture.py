"""実cold採録の時刻捕捉と、原共有境界の遅延・失敗寿命を限定検査。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import capture as C

ROOT = Path(__file__).resolve().parent
for name in ('g2_frozen_accounting_upgrade_2026-09-12_v1',
             'g2_collector_continuous_guard_2026-09-11_v1',
             'g2_empty_tail_desync_trigger_2026-09-11_v1'):
    sys.path.insert(0, str(ROOT.parent / name))
import upgrade_v2 as U
import live_configuration as L
import test_collector_continuous as T
real = T.real


@pytest.mark.parametrize('first', [0, T.FIRST])
def test_original_updates(real: Any, tmp_path: Path, monkeypatch: Any, first: int) -> None:
    c, b = real
    monkeypatch.setattr(b.O, 'FIRST', first)
    monkeypatch.setattr(b.O, 'LAST', first + T.STRIDE)
    state, pipe = {'output': tmp_path}, T.pipeline(c)
    with ExitStack() as stack:
        b.install(stack, c, None, state, enabled=True)
        tail = T.M.install(stack, state)
        factory = c.EventAccountingRecorder
        loop = U.build(c, b.O.SOURCE, overrides=L.settings())
        stack.callback(loop.generator.close)
        original = loop.collect_lean
        loop.RecognitionPipeline = type(pipe)
        capture = C.install(stack, loop, tail, {'source': 'artificial', 'run': tmp_path.name})
        cap = N(read=lambda: (True, T.np.zeros((1080, 1920, 3), dtype=T.np.uint8)))
        for frame in (first, first + T.STRIDE):
            loop.collect_lean(cap, pipe, frame, 1, T.STRIDE, T.FPS)
        saved = capture.snapshot()
        assert saved['first_frame'] == first and saved['last_frame'] == first + T.STRIDE
        assert saved['observed_count'] == saved['accounting']['observed_frame_count'] == 2
        assert len(saved['metadata']) == 2 and not saved['game_anchor_qualified']
        assert saved['accounting']['observer_version'] == 'event-accounting-observer/v1.1'
        assert c.EventAccountingRecorder is factory
    assert capture.closed and loop.collect_lean is original
    b.finish(state)
    b.verify(tmp_path)
    (tmp_path / 'CAPTURE.json').write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')


def isolated(c: Any) -> tuple[Any, Any]:
    shared = c._SharedGameCounter(multisignal_mode=True, require_newmatch_evidence=True)
    recorder = N(_observed_frame_count=0)
    loop = N(runtime_state={'accounting_recorder': recorder, 'shared_game': shared}, error=None)
    tail = N(count=0, stream=N(count=0), rows=[], check=lambda frame: None)
    def original(cap: Any, pipe: Any, frame: int, n: int, stride: int, fps: int) -> None:
        shared.observe_visual_signal(frame >= 60, frame / fps, pipe(frame))
        recorder._observed_frame_count += 1
        loop.runtime_state.update(fi=frame, t_sec=frame / fps)
        tail.count += 2
        tail.stream.count = tail.count
    loop.collect_lean = original
    return loop, tail


@pytest.mark.parametrize('evidence_frame,confirmation', [(60, 90), (120, 120), (1000, None)])
def test_original_shared_counter(real: Any, evidence_frame: int, confirmation: int | None) -> None:
    c, _ = real
    loop, tail = isolated(c)
    capture = C.Capture(loop, tail, {'source': 'artificial-shared-counter'})
    end = int((1 + c.BOUNDARY_VISUAL_RISE_PERSIST_SEC + c.BOUNDARY_NEWMATCH_EVIDENCE_WINDOW_SEC) * C.FPS) + C.STRIDE
    for frame in range(0, end + C.STRIDE, C.STRIDE):
        loop.collect_lean(None, lambda fi: fi >= evidence_frame, frame, 1, C.STRIDE, C.FPS)
    accepted = [e for e in capture.events if e['list'] == 'advance_times']
    if confirmation is None:
        assert not accepted
        assert any(e['list'] == 'rejected_rise_times' for e in capture.events)
    else:
        assert [(e['raw_value'], e['available_frame']) for e in accepted] == [(1.0, confirmation)]
    capture.close()


@pytest.mark.parametrize('failure', ['original', 'tail', 'count', 'clock'])
def test_sticky_failure(real: Any, failure: str) -> None:
    loop, tail = isolated(real[0])
    sentinel = RuntimeError('original_identity')
    if failure == 'original':
        def broken(*args: Any) -> None:
            raise sentinel
        loop.collect_lean = broken
    if failure == 'tail':
        tail.check = lambda fi: (_ for _ in ()).throw(sentinel)
    capture = C.Capture(loop, tail, {'source': 'artificial-failure'})
    if failure == 'count':
        capture.recorder._observed_frame_count = 10
    with pytest.raises(BaseException) as first:
        loop.collect_lean(None, lambda fi: True, 0, 1, C.STRIDE, 30 if failure == 'clock' else C.FPS)
    if failure in ('original', 'tail'):
        assert first.value is sentinel
    for action in (lambda: loop.collect_lean(None, None, 2, 1, C.STRIDE, C.FPS), capture.snapshot):
        with pytest.raises(BaseException) as again:
            action()
        assert again.value is first.value is loop.error is capture.error
    assert capture.count == 0 and capture.events == []
    capture.close()


def test_reentry_late_foreign(real: Any) -> None:
    loop, tail = isolated(real[0])
    original = loop.collect_lean
    capture = C.Capture(loop, tail, {'source': 'artificial-ownership'})
    with pytest.raises(ValueError, match='reentry'):
        C.Capture(loop, tail, {'source': 'duplicate'})
    foreign = lambda *args: None
    loop.collect_lean = foreign
    with pytest.raises(ValueError, match='foreign_hook'):
        capture.close()
    assert loop.collect_lean is foreign
    loop.collect_lean = capture.hook
    capture.close()
    assert loop.collect_lean is original
    capture.recorder._observed_frame_count = 1
    with pytest.raises(ValueError, match='late_recorder'):
        C.Capture(loop, tail, {'source': 'late'})
