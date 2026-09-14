"""凍結親のsrcをimportせず、新v2子と原保存prefixを検査する。"""
from __future__ import annotations
import builtins
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PRIOR = ROOT.parent / 'g2_persistent_ledger_process_2026-09-12_v1'
FPS, MILLISECONDS = 60, 1000
PINS = {
    ROOT / 'worker_v2.py': 'cc0777a04a697db928d992301fe403057cba8e952bc3eb2e31d8b558778d16e2',
    ROOT / 'wire_dependencies.py': '1c01e4727df347563dfb870437fdf6f657cb028f078b769e190dadf647dfe898',
    ROOT / 'engine.py': '10415025d8a788e1ee8e0c83b8d57c6a2b55b2eb5fedac11f8f829ab186397e8',
    PRIOR / 'transport.py': 'e7ad54ce3d934bced50aca6ab51bd8f66e8f625702f6fdc6196e704fef851193',
    PRIOR / 'protocol.py': '2980b3f3c58de9e4251d5518ca355cc2558ee53e16682cf38d603f631b3ef948',
    REPO / 'src/event_source_v1.py': '7e2ccbcb367e77b82141113d55c8e96b8054223804d170d5ec08fc2466bbcf70'}


def guard() -> None:
    for path, expected in PINS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('parent_dependency_changed:' + path.name)


def load(name: str, path: Path, dependencies: dict | None = None) -> Any:
    guard()
    if name in sys.modules:
        raise ValueError('parent_private_alias_collision:' + name)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__builtins__ = dict(vars(builtins))
    original = builtins.__import__
    def imported(key: str, *args: Any, **kwargs: Any) -> Any:
        if dependencies and key in dependencies:
            return dependencies[key]
        return original(key, *args, **kwargs)
    module.__builtins__['__import__'] = imported
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


P = load('_joint_parent_protocol_v2', PRIOR / 'protocol.py')
P.SCHEMA = 'g2-joint-ledger-process/v2'
T = load('_joint_parent_transport_v2', PRIOR / 'transport.py', {'protocol': P})
S = load('_joint_parent_event_reader_v2', REPO / 'src/event_source_v1.py')


def readback(path: Path, identity: list[str], frame: int, old: tuple, expected_sha: str) -> tuple:
    guard()
    batches = tuple(S.iter_committed_batches(path, expected_first_seq=0))
    P.require(batches and batches[-1].end_offset == path.stat().st_size, 'parent_incomplete_part')
    codes = tuple(P.encoded(b.events) for b in batches)
    P.require(len(codes) > len(old) and codes[:len(old)] == old, 'parent_prefix_changed')
    P.require(all([e[k] for k in P.IDENTITY_KEYS] == identity for b in batches for e in b.events), 'parent_part_identity')
    last = batches[-1].events[0]['timing']
    P.require(last['available_frame'] == frame and last['available_ms'] == frame * MILLISECONDS // FPS, 'parent_part_cutoff')
    P.require(hashlib.sha256(b''.join(codes)).hexdigest() == expected_sha, 'parent_event_sha')
    return codes


class Client(T.Client):
    def __init__(self, part: Path, identity: tuple[str, str, str], *, timeout: float = P.TIMEOUT) -> None:
        guard()
        super().__init__(part, identity, timeout=timeout, worker=ROOT / 'worker_v2.py')

    def join(self, producer: dict, observation: dict) -> dict:
        if self.error is not None:
            raise self.error
        try:
            guard()
            value = dict(schema=P.SCHEMA, seq=self.seq, identity=self.identity, op='join',
                         producer=producer, observation=observation)
            raw = P.encoded(value)
            P.require(len(raw) <= P.MAX_BYTES, 'parent_request_size')
            # 隔離/例外でも元入力が残る。未完票を成功票として公開しない。
            ticket = self.part.with_suffix(self.part.suffix + '.request' + str(self.seq) + '.json')
            with ticket.open('xb') as stream:
                stream.write(raw)
                stream.flush()
            reply = self.exchange(value)
            P.require(set(reply) == {'schema', 'seq', 'identity', 'op', 'request_sha256', 'result'}
                      and reply['op'] == 'joined', 'parent_reply_keys')
            P.require(reply['request_sha256'] == hashlib.sha256(raw).hexdigest(), 'parent_canonical_request_sha')
            result = reply['result']
            from parent_validation import validated
            validated(result, producer, observation)
            codes = readback(self.part, self.identity, observation['frame'], self.prefix, result['event_sha256'])
            self.prefix, self.seq, self.ambiguous_commit = codes, self.seq + 1, False
            return result
        except BaseException as error:
            self.fail(error)
            raise

    def fail(self, error: BaseException) -> None:
        self.error = error
        try:
            self.abort()
        finally:
            value = dict(error_type=type(error).__name__, reason=str(error), seq=self.seq,
                         ambiguous_commit=self.ambiguous_commit, part=str(self.part),
                         production_permission=False, quality_gate_clear=False)
            try:
                with self.part.with_suffix(self.part.suffix + '.failure.json').open('xb') as stream:
                    stream.write(P.encoded(value))
            except BaseException as diagnostic_error:
                self.diagnostic_error = diagnostic_error  # 元例外を置換しない。

    def close(self) -> None:
        if self.error is not None:
            raise self.error
        if self.closed:
            return
        try:
            super().close()
            completion = dict(schema='g2-joint-parent-session-completion/v2', closed=True,
                child_exit_code=self.child.returncode, accepted_count=self.seq,
                event_sha256=hashlib.sha256(b''.join(self.prefix)).hexdigest(),
                production_permission=False, quality_gate_clear=False)
            with self.part.with_suffix(self.part.suffix + '.complete.json').open('xb') as stream:
                stream.write(P.encoded(completion))
        except BaseException as error:
            self.fail(error)
            raise
