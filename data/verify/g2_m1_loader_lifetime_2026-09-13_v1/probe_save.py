"""新Sessionが保持する元SAVEで実親/実モデル/実保存を確認。live capture入力は人工代替。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace as N

from pathlib import Path
import late_loader
from probe_constructor import fixture, settings, ROOT, R

STOP = 'save_probe_without_claiming_live_registry_capture'


def bound(parts: object, session: object, request: dict) -> object:
    values = tuple(parts.binding.S.decode(s) for s in request['states'])
    boards = tuple(parts.belief.Board.from_dict({'grid': grid}) for grid in request['observed'])
    source = json.dumps(request['row'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return N(values=values, observed=boards, inputs=session.members.input_type(source),
             frame=request['frame'], tokens=tuple(request['tokens']), digest=request['context_digest'])


def main() -> None:
    output = ROOT / 'original_save_v1'
    output.mkdir(exist_ok=False)
    reports = []
    with ExitStack() as outer:
        R.configured(outer)
        import probe_native_merge
        owner = sys.modules[R.A.A.A.V4.OWNED_ALIAS]
        late_loader.install(outer, R.A.A.A.V4.A.D, owner, R.A.A.A.V4.replace_owned)
        create = R.A.A.A.V4.A.D.session_creator(owner.bootstrap().load, (35370, 35410))
        parts = owner.dependencies().modules()
        inputs = json.loads((R.A.ROOT / 'SAVE_INPUTS_v1.json').read_text(encoding='utf-8'))
        assert inputs['artificial'] is True
        identity, messages = inputs['identity'], inputs['messages']
        try:
            with ExitStack() as inner:
                f = fixture(parts, owner.bootstrap().load, output, inner)
                producer = f.state['joint_producer_capture']
                producer.identity = dict(source_id=identity[0], run_id=identity[1])
                session = create(inner, dict(state=f.state, pipe=f.pipe, factory=f.factory, stack=inner))
                thermal = settings(f.state['joint_parent_client'].child.pid)
                saver = type(session).completed.__globals__['SAVE']
                client = f.state['joint_parent_client']
                for message in messages[:2]:
                    request = message['observation']
                    value = bound(parts, session, request)
                    session.capture = lambda: value  # 人工代替。原live Registry/原J資格の証明には使わない。
                    producer.snapshot = lambda: deepcopy(message['producer'])
                    path = output / f"MODEL_CANDIDATE_{request['frame']}.json"
                    saved = saver.run(session.capture, session.members, path,
                                     sample_count=request['sample_count'], seed=request['seed'])
                    raw = path.read_bytes()
                    packet = json.loads(raw)
                    assert saved['sha256'] == hashlib.sha256(raw).hexdigest()
                    assert packet['result']['supported'] and packet['parent_recapture_verified']
                    reports.append(dict(frame=saved['frame'], sha256=saved['sha256'],
                                        supported=packet['result']['supported']))
                assert not session.saved and session.schedule_rows == 0
                raise RuntimeError(STOP)
        except RuntimeError as error:
            assert str(error) == STOP
        assert client.closed and client.child.returncode == 0 and client.seq == 2
    report = dict(original_selected_SAVE=True, original_parent_and_models=True, saved=reports,
        child_exit=client.child.returncode, accepted_count=client.seq, thermal=thermal,
        artificial_producer_and_live_capture=True,
        creator_before_modules=True, original_Session_completed=False, actual_video=False, quality_gate_clear=False)
    with (output / 'RESULT.json').open('x') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
