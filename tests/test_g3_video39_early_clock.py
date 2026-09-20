"""元J/Witness/EarlyHistoryを使う30fps接続。人工CPU入力はGTではない。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any
import pytest
from scripts import g3_video39_early_clock as G
from tests.test_g3_video39_next import item

ROOT = G.G.P.M.ROOT / 'data/verify'
ASYNC = ROOT / 'g2_async_projected_evaluation_2026-09-13_v1'
PUB = ROOT / 'g2_belief_live_publication_2026-09-11_v1'
sys.path[:0] = [str(ASYNC), str(PUB)]
import test_journal_pair_reader as T
import journal_witness as W


def replace(stack: Any, module: Any, name: str, value: Any) -> None:
    old = getattr(module, name)
    stack.callback(setattr, module, name, old)
    setattr(module, name, value)


def emit(journal: Any, pipe: Any, frame: int, values: dict, change: bool = False) -> None:
    """元complete_stepを人工30fps入力で二側実行。原scope/原emitを使う。"""
    journal.history.frame, journal.history.time_sec = frame, frame / 30
    journal.expected.extend((frame, side) for side in ('1P', '2P'))
    journal.selected = set(journal.expected)
    for side in ('1P', '2P'):
        scope = journal.scope(pipe, side, frame, frame / 30)
        value = dict(scope=scope, token=f'step:{journal.steps}', events=[],
                     epoch=journal.epoch(pipe, side), return_line=None, frame=None)
        if change and side == '1P':
            values[side].action_revision += 1
        journal.steps += 1
        journal.complete_step(value, None, None, sys.getprofile())


def saved_clock_gate(module: Any, journal: Any, output: Path) -> None:
    """人工J票の時計検査→旧code拒否まで確認。実J全体検収とは呼ばない。"""
    raw = journal.stream.getvalue()
    (output / 'J_ORIGINAL_CPU.jsonl').write_text(raw)
    row = json.loads(raw.splitlines()[-1])
    # 一行の検査入力として再採番する。実run原票は変更しない。
    row['row_index'] = 0
    for name, clock, reason in (('good-clock', 1 / 30, 'journal_saved_code'),
                                 ('bad-clock', 1 / 60, 'journal_saved_clock')):
        target = output / name
        target.mkdir()
        row['time_sec'] = clock
        (target / module.SIDECAR).write_text(json.dumps(row) + '\n')
        receipt = dict(source_id=journal.source_id, run_id=journal.run_id, anchors={})
        with pytest.raises(ValueError, match=reason):
            module.verify_rows(target, receipt, [[1, row['side']]])


@pytest.mark.parametrize('fault', ['none', 'clock', 'source', 'generation'])
def test_actual_history(tmp_path: Path, fault: str) -> None:
    journal, pipe, values = T.setup()
    journal.expected = []
    with ExitStack() as modules:
        jmodule = modules.enter_context(G.G.P.loaded(ROOT / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py', G.JOURNAL_SHA))
        G.prepare_journal(modules, jmodule, replace)
        actual = jmodule.Recorder.__new__(jmodule.Recorder)
        actual.__dict__.update(vars(journal))
        journal = actual
        original = journal.emit
        reader = modules.enter_context(G.G.P.loaded(ASYNC / 'journal_pair_reader.py', G.READER_SHA))
        history_module = modules.enter_context(G.G.P.loaded(ASYNC / 'early_origin_history.py', G.HISTORY_SHA))
        proof = G.prepare(modules, reader, history_module, None, replace)
        assert proof['stride_after'] == 1 and history_module.FPS == 60
        try:
            with ExitStack() as owner:
                history = history_module.EarlyHistory(owner, journal, (0, 1), tmp_path, W, reader)
                emit(journal, pipe, 0, values)
                history.observe(pipe, 0)
                emit(journal, pipe, 1, values, fault == 'generation')
                if fault == 'clock':
                    journal.history.time_sec = 1 / 60
                elif fault == 'source':
                    row = json.loads(history.witness.rows['1P'])
                    row['source_id'] = 'foreign-source'
                    history.witness.rows['1P'] = json.dumps(row)
                history.observe(pipe, 1)
        except ValueError:
            assert fault in ('clock', 'source')
        else:
            assert fault in ('none', 'generation')
        saved_clock_gate(jmodule, journal, tmp_path)
    assert journal.emit == original and history.closed
    saved = [json.loads(line) for line in (tmp_path / 'EARLY_ORIGIN_HISTORY.jsonl').read_text().splitlines()]
    assert len(saved) == (1 if fault in ('clock', 'source') else 2)
    if fault == 'generation':
        assert saved[-1]['holds'] == ['1P:generation_changed_within_step']
    status = json.loads((tmp_path / 'EARLY_ORIGIN_HISTORY_CLOSE.json').read_text())
    assert not status['transferred'] and status['cleanup_errors'] == []


def test_canonical_expression() -> None:
    G.G.validate(item())
    assert all(frame * 1 / 30 == frame / 30 for frame in range(G.G.END))
