"""既存原J/Native/Registry/prefix全保存CPUへ人工inactive2更新を接続。実動画ではない。"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v39'))
import owned_adapter as A
sys.path.insert(0, str(ROOT.parent/'g2_prefix_lane_integration_2026-09-13_v1'))
import prefix_continuous_fixture as F
import test_full_prefix_path as OLD

DEADLINE = 24


def end_context(f: Any, frame: int) -> None:
    row = f.contexts[-1]
    if row['frame_idx'] != frame:
        row = deepcopy(row)
        f.contexts.append(row)
    row.update(frame_idx=frame, time_sec=frame/60, available_frame=frame,
        capture_status='CAPTURED', failures=[], upstream_failures={}, capture_token=f'artificial-end:{frame}',
        generation={'after':{'value':{'1P':{'side':'1P','reset_epoch':f.mode.arrival_ledger.scope[5]}}}},
        update=dict(returned=True, exception=None, returned_frame_idx=frame, returned_time_sec=frame/60,
            match_end_locked_observed=True, match_end_locked=True,
            post_match_lockdown_active_observed=True, post_match_lockdown_active=True))


def inactive(f: Any, frame_idx: int) -> None:
    is_active, time_sec = False, frame_idx/60
    f.j.update_before['1P'] = dict(accounting=f.J.account(f.pipe,'1P'))
    f.pipe._tsumo_count_1p.clear()  # 原pipeline inactive会計clearを人工入力として模擬。
    def original(p: Any, s: str, frame: int, clock: float, active: bool, pair: Any) -> None:
        return None
    f.j.wrap_enqueue(original)(f.pipe,'1P',frame_idx,time_sec,False,None)
    scope = f.j.scope(f.pipe,'1P',frame_idx,time_sec)
    item = dict(frame=sys._getframe(), pipe=f.pipe, scope=scope, epoch=3,
        token=f'step:{frame_idx}', events=[])
    row = f.mode.observe(item, N(state=N(value='menu'),confirmed_board=None), None)
    f.mode.save(row)
    f.j.emit(dict(scope, kind='step', token=item['token'], software_reset=3,
        status='returned', exception=None, events=[], generation_after=scope['generation'],
        code_sha256='artificial-unified-step-code', fixture_artificial_step=True))
    end_context(f, frame_idx)


def execute(output: Path) -> dict:
    with ExitStack() as outer:
        selected = A.configured(outer)
        sys.path.insert(0, str(ROOT.parent/'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
        import probe_native_merge
        owner = sys.modules[A.A.A.A.V4.OWNED_ALIAS]
        with selected.__globals__['S'].configured():
            return selected_body(owner, output)


def selected_body(owner: Any, output: Path) -> dict:
    parts = owner.dependencies().modules()
    F.END = DEADLINE  # 新規人工fixtureの開始時設定。旧run/保存票のdeadlineは変えない。
    f = F.setup(parts, owner.bootstrap().load, output)
    f.j.codes.add(inactive.__code__)
    f.state['provisional_context_observer'] = N(rows=f.contexts)
    with ExitStack() as inner:
        inner.callback(f.mode.stream.close)
        inner.callback(f.mode.close)
        f.mode.arrival_capture = f.M.E.install(inner, f.mode, output)
        F.sequence(f)
        end_context(f, 20)
        inactive(f, 22)
        inactive(f, 24)
    raw = [json.loads(line) for line in f.j.stream.getvalue().splitlines()]
    for row in raw:
        if row['kind'] == 'step':
            row.setdefault('code_sha256','artificial-unified-step-code')
    OLD.write(output,'atomic_journal.jsonl',raw)
    OLD.write(output,'provisional_context.jsonl',f.contexts)
    OLD.basis(f, output)
    final = sys.modules['probability_finish']
    assert final.SAVED is parts.arrival_saved
    verified = final.SAVED.verify(f.state, parts.binding.S,
        N(extract=f.M.CORE.Mode.observe.__globals__['N'].extract))
    assert verified['terminal']['terminal_end'] == DEADLINE
    assert verified['transitions'] == verified['consumptions'] == 2
    return verified


def main() -> None:
    attempt = sys.argv[1] if len(sys.argv)>1 else 'v1'
    if not attempt.isalnum():
        raise ValueError('invalid CPU attempt')
    output = ROOT/('full_terminal_cpu_' + attempt)
    output.mkdir(exist_ok=False)
    started, error, verified = time.perf_counter(), None, None
    try:
        verified = execute(output)
    except BaseException as caught:
        error = repr(caught)
    result = dict(error=error, verified=verified, seconds=time.perf_counter()-started,
        artificial_basis_pipe_step_images_pop_end_context=True, full_original_finalizer=False,
        actual_video=False, quality_gate_clear=False)
    OLD.write(output,'RESULT.json',result)
    print(json.dumps(result))
    if error is not None:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
