"""人工同一所有の終了票→原completed→原参考/root保存→親→close。実履歴再構築ではない。"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import test_terminal_prefix_reader as T

source = T.source
VERIFY = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(VERIFY/'g2_history_prediction_diagnosis_2026-09-13_v1'),
               str(VERIFY/'g2_belief_live_publication_2026-09-11_v1')]


def connect(session: Any, stream: Any, lease: Any, reader: Any, ledger: Any) -> Any:
    import journal_origin_capture_candidate as C
    import prefix_projected_input as P
    projection = lambda frame: P.read(session, lease, reader, T.R, None, frame, settled=lambda _:False)
    return C.OriginCapture(session.witness, session.journal, session.pipe, ledger, None, None,
                           lambda _:False, stream, projected_input=projection)


def test_completed_saved_and_closed_after_terminal_hold(source: Any, tmp_path: Path, monkeypatch: Any) -> None:
    with ExitStack() as owners:
        live, mode, base, step, lane, lease, reader = T.fixture(source, tmp_path, owners)
        import journal_origin_capture_candidate as C
        import session_origin_binding as B
        import session_runtime_binding_candidate as S
        import fixed_origin_reference as F
        import fixed_root_probability as Q
        owner = sys.modules[T.F.E.Z.T.A.A.A.A.V4.OWNED_ALIAS]
        load_source = owner.bootstrap().load
        commit = load_source('_terminal_test_commit', VERIFY.parents[1]/'src/chain_commit_candidate_v1.py')
        ledger = load_source('_terminal_test_prediction', VERIFY.parents[1]/'src/chain_prediction_ledger_v1.py',
                             {'src.chain_commit_candidate_v1':commit})
        for alias in ('_terminal_test_commit', '_terminal_test_prediction'):
            owners.callback(sys.modules.pop, alias, None)
        step['events'] = []
        other = deepcopy(step)
        other['side'] = '2P'
        for key in ('generation', 'generation_after'): other[key]['side'] = '2P'
        pair = N(holds=(), rows_json=json.dumps([step, other]))
        reader.read_pair = lambda *args:pair
        monkeypatch.setattr(C.R, 'read_pair', reader.read_pair)
        history = N(closed=False, error=None, sealed=True, transferred=True,
            probability=N(closed=True), journal=base.journal, pipe=base.pipe,
            last_frame=mode.arrival_ledger.clock)
        monkeypatch.setitem(sys.modules, '_async_live_fixed_origin_reference', F)
        def load(alias: str, path: Any, injection: dict) -> Any:
            assert injection['fixed_origin_reference'] is F
            return Q
        after, counts = S.root_hook(load, reader, lambda:history, True, F.save)
        calls = []
        class Session:
            def __init__(self, stack: Any) -> None: self.__dict__.update(vars(base))
            def completed(self, frame: int) -> None: calls.append(frame)
        original = Session.completed
        with ExitStack() as stack:
            B.install_class(stack, Session, lambda value,stream:connect(value,stream,lease,reader,ledger), after_completed=after)
            session = Session(stack)
            capture = session.projected_origin_binding.capture
            session.completed(step['frame_idx'])
            assert capture.last_frame == mode.arrival_capture.terminal_last == step['frame_idx']
        assert Session.completed is original and capture.closed
        assert session.projected_origin_binding.capture is None
        assert calls == [step['frame_idx']] and counts['HOLD'] == 1
    rows = [json.loads(line) for line in (tmp_path/B.STREAM_NAME).read_text().splitlines()]
    assert len(rows) == 3 and rows[0]['projected_input']['reason'] == 'match_ended_scope_frozen'
    assert rows[0]['projected_input']['status'] == rows[2]['status'] == 'HOLD'
    close = json.loads((tmp_path/B.CLOSE_NAME).read_text())
    assert close['last_frame'] == step['frame_idx'] and close['capture_closed'] and close['stream_closed']
    assert close['cleanup_errors'] == [] and close['original_error'] is None
