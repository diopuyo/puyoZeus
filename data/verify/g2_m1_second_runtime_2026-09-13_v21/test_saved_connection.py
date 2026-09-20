"""最終選択した2P型の原Native→別票実保存→原J照合を結合する人工CPU対照。"""
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
from typing import Any
import owned_adapter as A

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream)


def artificial_session(parts: Any, owner: Any, stack: Any, output: Path) -> Any:
    first = parts.mode.Mode.__new__(parts.mode.Mode)
    first.connection, first.native, first.activation = N(binding=None), None, None
    first.arrival_ledger, first.error, first.rows, first.stream = None, None, 0, io.StringIO()
    class Base:
        def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                     contract: Any, members: Any, frames: Any) -> None:
            self.physical, self.state = physical, context['state']
    binding = owner.bootstrap().load('_notice_test_binding_saved', A.CANDIDATE / 'second_mode_binding.py')
    stack.callback(lambda: sys.modules.pop('_notice_test_binding_saved', None))
    cls = binding.session_class(N(Session=Base), parts.mode)
    context = dict(state=dict(probabilistic_tracking_mode=first,
        probabilistic_basis_connection=first.connection, output=output))
    return cls(stack, context, None, N(Mode=parts.mode.Mode), None, None, ())


def sequence(F: Any, f: Any) -> list[dict]:
    rows = []
    for frame, events in ((35808, F.notice() + F.pop()), (35810, [])):
        if frame == 35810:
            f.pipe.observed = f.pipe.observed.copy()
            f.pipe.observed.set(12, 0, 5)
            f.pipe.observed.set(11, 0, 5)
        row = F.step(f, frame, events)
        scope = row['scope']
        rows.append(dict(scope, kind='step', token=f'step:{frame}', software_reset=2,
            generation_after=scope['generation'], status='returned', exception=None, events=events))
    return rows


def test_actual_selected_notice_saved(tmp_path: Path) -> None:
    output = Path(tempfile.mkdtemp(prefix='saved_cpu_', dir=ROOT))
    with ExitStack() as stack:
        A.configured(stack)
        import probe_native_merge
        owner = sys.modules[A.A.A.A.V4.OWNED_ALIAS]
        parts = owner.dependencies().modules()
        spec = importlib.util.spec_from_file_location('_notice_cpu_input', A.NOTICE / 'test_settled_notice.py')
        F = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(A.NOTICE))
        spec.loader.exec_module(F)
        session = artificial_session(parts, owner, stack, output)
        f = F.setup(parts)
        previous = f.mode
        f.mode = session.physical.Mode(f.c, previous.state, io.StringIO())
        f.mode.native = previous.native
        journal = sequence(F, f)
        assert len(f.mode.applied) == 1 and not f.mode.native.pending
        saved = sys.modules['_g2_second_notice_saved_v21']
        initial = dict(state=parts.binding.S.encode(f.before), source_call_token=f.c.binding.initial_call_token)
        modes = [dict(initial=initial, retired=None, closed=True, error=None)]
        write_json(output / 'BELIEF_M1_SESSION.json', dict(error=None, session_error=None, modes=modes,
            artificial_session_status=True, quality_gate_clear=False))
        with (output / 'atomic_journal.jsonl').open('x') as stream:
            for row in journal:
                stream.write(json.dumps(row) + '\n')
        session.second_settled_notice_stream.close()
        review = saved.verify(output, 35810)
    write_json(output / 'RESULT.json', dict(review=review, selected_physical_type=True,
        original_native_registry=True, actual_notice_stream=True, artificial_J_and_Session_base=True,
        actual_video=False, full_original_finalizer=False, quality_gate_clear=False))
