"""実step/J/世代/旧veto→現在資格→採録を人工初期条件で結合検査する。"""
from __future__ import annotations

from contextlib import ExitStack
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from scripts import g3_palette_reset_hold as P
from scripts import chain_prediction_generation_hooks_v1 as H
from tests.test_g3_admission import A, Ocr, collector, owned_replace, admission_root  # noqa: F401  admission_root は autouse fixture。採録先の根を一時領域へ差し替える (W45)
from tests.test_g3_current_exit import result
from tests.test_next_enqueue_live_shadow_v1 import CpuReader, CpuMatch

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/mnt/d/puyo_analyzer/verify/g3_repair_2026-09-15_v1/PALETTE_RESET_INPUT_ROW.json')
FRAME, CLOCK = 8724, 8724 / 60


def load(name: str, folder: str) -> Any:
    """原moduleを私有名で読む。旧ファイルは変更しない。"""
    path = ROOT / 'data/verify' / folder
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def modules(collector: Any) -> tuple:
    """実paletteと実Jを共有し、各試験で所有参照を復元する。"""
    return (load('_g3_hold_palette_test', 'g2_palette_evidence_veto_2026-09-09_v1/adapter.py'),
            load('_g3_hold_journal_test', 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'))


def setup_pipe(c: Any, patch: Any, reset: bool) -> tuple:
    """既存実constructor。SM.updateの固定とcounterは人工条件として明示する。"""
    from src.match_state import MatchState
    from src.board_state_machine import BoardState
    row = json.loads(INPUT.read_bytes())['events'][0]
    board = c.Board.from_list(row['confirmed']['grid']) if reset else c.Board()
    if not reset:
        for col in range(3):
            board.set(12, col, 4)
    cnn = c.Board.from_list(row['boards']['cnn_board']['grid']) if reset else board.copy()
    ocr = Ocr()
    ocr.values['1P'] = 0
    pipe = c.RecognitionPipeline(image_reader=CpuReader(cnn), match_state_detector=CpuMatch(MatchState.IN_MATCH),
                                score_ocr=ocr, chain_tracker_1p=None, chain_tracker_2p=None)
    ctx = pipe._sm_2p.context
    ctx.state, ctx.frame_idx, ctx.confirmed_board = BoardState.STABLE, FRAME, board
    ctx.next_queue.extend(row['next_queue'] if reset else [(4, 4)] * 3)
    pipe._baseline_broken_consec_2p = 59 if reset else 0
    patch.setattr(pipe._sm_2p, 'update', lambda *args: ctx)
    other = result(FRAME).p1
    pipe._sm_1p.context.state, pipe._sm_1p.context.frame_idx = BoardState.STABLE, FRAME
    pipe._sm_1p.context.confirmed_board = other.confirmed_board.copy()
    return pipe, cnn, other


def install_journal(stack: Any, c: Any, pipe: Any, module: Any, output: Path) -> tuple:
    """実Recorder/実scope/実wrap_step。NEXT controller履歴0だけ人工で置く。"""
    tracker = H.PipelineGenerationRecorder(lambda row: None)
    tracker.bind_pipeline(pipe)
    H._install_machine_hooks(stack, type(pipe._sm_2p), tracker)
    controller = N(instances={id(pipe): N(histories={'2P': N(epoch=0)})})
    history = N(frame=FRAME, time_sec=CLOCK)
    journal = module.Recorder(output, history, tracker, controller, 'video_38', 'artificial-full-step', [(FRAME, '2P')])
    original = c.RecognitionPipeline._step_side
    journal.codes = {original.__code__}
    stack.callback(journal.close)
    owned_replace()(stack, c.RecognitionPipeline, '_step_side', journal.wrap_step(original))
    return journal, tracker, history


def driver(c: Any, patch: Any, cnn: Any, other: Any, tracker: Any) -> None:
    """人工outer updateから実2P stepを呼び、実結果型で出口へ渡す。"""
    def update(pipe: Any, frame: int, clock: float, pixels: Any) -> Any:
        for side in A.SIDES:
            pipe._update_score_tracker(getattr(pipe, '_score_tracker_' + side.lower()), pixels)
        tracker.begin_frame(pipe, frame, clock)
        try:
            value = pipe._step_side('2P', frame, clock, True, cnn, None, score_d_2p_for_ojama=0,
                sm=pipe._sm_2p, gen=pipe._gen_2p, drift=pipe._drift_2p, score_tracker=None)
        finally:
            tracker.end_frame()
        return replace(result(frame), p1=other, p2=value)
    patch.setattr(c.RecognitionPipeline, 'update', update)


@pytest.mark.parametrize('mode', ['normal', 'reset', 'old_reset'])
def test_full_step_to_collector(collector: Any, modules: tuple, monkeypatch: Any, tmp_path: Path, mode: str) -> None:
    """元失敗と保留修復の到達を単一経路で検査する。全A40/画像GTではない。"""
    c, (palette, journal_module) = collector, modules
    pipe, cnn, other = setup_pipe(c, monkeypatch, mode != 'normal')
    state, acc = dict(output=tmp_path), c._LeanNpzAccumulator()
    with ExitStack() as stack:
        journal, tracker, history = install_journal(stack, c, pipe, journal_module, tmp_path)
        state['atomic_journal_observer'] = journal
        palette.install(stack, c, history, state, enabled=True)
        driver(c, monkeypatch, cnn, other, tracker)
        A.install(stack, c, state, owned_replace(), source_id='artificial-step-to-admission')
        P.G.C.install(stack, c, state, owned_replace())
        P.G.L.install(stack, state, owned_replace())
        P.G.install(stack, state, owned_replace())
        hold = P.install(stack, state, owned_replace()) if mode != 'old_reset' else None
        if mode == 'old_reset':
            with pytest.raises(ValueError, match='palette_not_stable_ctx'):
                pipe.update(FRAME, CLOCK, object())
        else:
            value = pipe.update(FRAME, CLOCK, object())
            c._process_side_lean(acc, c._SideState(), '2P', value.p2.confirmed_board, value.p2.state,
                0, 'artificial-full-step', CLOCK, FRAME, shared_game=c._SharedGameCounter(), tsumo_count=2,
                match_end_locked=False, post_match_lockdown_active=False)
            assert len(acc.grids) == (0 if mode == 'reset' else 1)
            assert hold.rows == hold.applied == (1 if mode == 'reset' else 0)
    assert journal.closed and state['palette_evidence_veto'].closed
    rows = [json.loads(line) for line in (tmp_path / 'atomic_journal.jsonl').read_text().splitlines()]
    assert rows[-1]['status'] == ('exception' if mode == 'old_reset' else 'returned')
    if mode == 'reset':
        saved = json.loads((tmp_path / 'G3_PALETTE_RESET_HOLD.jsonl').read_text())
        assert saved['reason'] == P.REASON and saved['live_frame'] == 0 and saved['frame_idx'] == FRAME
        current = json.loads((tmp_path / 'G3_CURRENT_EXIT.jsonl').read_text())
        assert current['status'] == 'HOLD' and current['visible_confirmed_boards'] is None
        assert json.loads((tmp_path / 'G3_PALETTE_RESET_STATUS.json').read_text())['complete']


def component(stack: Any, modules: tuple, output: Path, mode: str) -> tuple:
    """ガード境界用の人工scope。実J constructor/SM/reset hookを使う。"""
    from src.board_state_machine import BoardStateMachine, BoardState
    from src.board import Board
    palette, journal_module = modules
    pipe = N(_sm_1p=BoardStateMachine(), _sm_2p=BoardStateMachine())
    ctx = pipe._sm_2p.context
    ctx.state, ctx.frame_idx, ctx.confirmed_board = BoardState.STABLE, FRAME, Board()
    ctx.confirmed_board.set(12, 0, 4)
    tracker = H.PipelineGenerationRecorder(lambda row: None)
    tracker.bind_pipeline(pipe)
    tracker.begin_frame(pipe, FRAME, CLOCK)
    H._install_machine_hooks(stack, BoardStateMachine, tracker)
    controller = N(instances={id(pipe): N(histories={'2P': N(epoch=0)})})
    history = N(frame=FRAME, time_sec=CLOCK)
    journal = journal_module.Recorder(output, history, tracker, controller, 'video_38', 'artificial-boundary', [(FRAME, '2P')])
    journal.codes = {invoke.__code__}
    journal.active = dict(pipe=pipe, scope=journal.scope(pipe, '2P', FRAME, CLOCK), epoch=0)
    rec = palette.Recorder(output, history, journal)
    stack.callback(journal.close)
    stack.callback(rec.close)
    if mode not in ('normal_veto', 'cnn_missing'):
        pipe._sm_2p.reset(keep_match_state=False)
    if mode == 'epoch':
        journal.active['scope'] = journal.scope(pipe, '2P', FRAME, CLOCK)
    elif mode == 'clock':
        history.frame -= 2
    elif mode == 'source':
        journal.active['scope']['source_id'] = 'foreign'
    elif mode == 'live_frame':
        pipe._sm_2p.context.frame_idx = 1
    elif mode == 'caller':
        journal.codes.clear()
    return pipe, ctx, rec


def invoke(palette: Any, rec: Any, pipe: Any, context: Any, mode: str, calls: list) -> Any:
    """元wrapperのsys._getframeがこの人工callerを観測する。"""
    from src.board import Board
    self, side, ctx, sm = pipe, '2P', context, pipe._sm_2p
    frame_idx, time_sec = FRAME, CLOCK
    frame_bgr, region_for_validate = object(), object()
    cnn_board = None if mode == 'cnn_missing' else ctx.confirmed_board
    changed = Board()
    changed.set(12, 0, 1)
    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append('validator')
        return changed
    board = ctx.confirmed_board.copy() if mode == 'argument' else ctx.confirmed_board
    out = palette.wrapper(original, rec)(board, [], frame_bgr=frame_bgr, region=region_for_validate)
    return out, changed


@pytest.mark.parametrize('mode', ['reset_cells', 'normal_veto', 'cnn_missing', 'epoch', 'clock',
                                  'source', 'live_frame', 'caller', 'argument', 'save'])
def test_scope_rejection_and_reason(modules: tuple, tmp_path: Path, monkeypatch: Any, mode: str) -> None:
    """無関係なguardは通さず、reset/通常CNN欠測/正常vetoを別理由で保存する。"""
    palette, _ = modules
    good = mode in ('reset_cells', 'normal_veto', 'cnn_missing')
    expected = nullcontext() if good else pytest.raises((ValueError, OSError), match='palette_|hold_save_failure')
    calls: list[str] = []
    with expected:
        with ExitStack() as stack:
            pipe, ctx, rec = component(stack, modules, tmp_path, mode)
            original_bound = palette.bound_cnn
            def counted(*args: Any) -> Any:
                calls.append('bound')
                return original_bound(*args)
            owned_replace()(stack, palette, 'bound_cnn', counted)
            hold = P.Hold(palette, tmp_path)
            stack.push(hold.close)
            owned_replace()(stack, palette, 'bound_cnn', hold.bound)
            owned_replace()(stack, palette, 'apply_veto', hold.apply)
            if mode == 'normal_veto':
                monkeypatch.setattr(palette, 'measure', lambda *args: ((50, 100, 100),
                    {color: 0.0 if color == 4 else 99.0 for color in range(1, 6)}, 60, 60.0))
            if mode == 'save':
                def fail(row: dict) -> bytes:
                    raise OSError('hold_save_failure')
                monkeypatch.setattr(P, 'encoded', fail)
            out, changed = invoke(palette, rec, pipe, ctx, mode, calls)
            assert (out is changed) == (mode != 'normal_veto')
            assert changed.get(12, 0) == 1
    assert calls == ['validator', 'bound']
    assert hold.stream.closed and rec.closed
    if good:
        palette.finish({palette.KEY: rec})
        row = json.loads((tmp_path / palette.SIDECAR).read_text())
        expected_reason = {'reset_cells': P.REASON, 'normal_veto': 'cnn_and_full_hsv_support_original',
                           'cnn_missing': 'cnn_missing_or_disagrees'}[mode]
        assert row['cells'][0]['reason'] == expected_reason
        assert palette.verify(tmp_path)['vetoed_cells'] == int(mode == 'normal_veto')
    else:
        status = json.loads((tmp_path / 'G3_PALETTE_RESET_STATUS.json').read_text())
        assert not status['complete'] and status['original_error'] and hold.error
