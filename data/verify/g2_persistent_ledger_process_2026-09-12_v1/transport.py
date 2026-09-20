"""一つの子を所有し、不確定送信失敗は原票保持のまま停止するWSL通信。"""
from __future__ import annotations
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any
import protocol as P

ROOT = Path(__file__).resolve().parent


class Client:
    def __init__(self, part: Path, identity: tuple[str, str, str], *, timeout: float = P.TIMEOUT,
                 worker: Path = ROOT / 'worker.py') -> None:
        P.require(len(identity) == 3 and all(type(x) is str and x for x in identity), 'identity')
        P.require(timeout > 0 and not part.exists(), 'initial_path_or_timeout')
        self.identity, self.part, self.timeout = list(identity), part, timeout
        self.seq, self.closed, self.busy = 0, False, False
        self.error: BaseException | None = None
        self.ambiguous_commit = False
        self.prefix: tuple[bytes, ...] = ()
        self.stderr = part.with_suffix(part.suffix + '.stderr').open('xb')
        self.child: Any = None
        try:
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='-1')
            self.child = subprocess.Popen([sys.executable, '-I', str(worker), str(part), P.encoded(self.identity).decode()],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, cwd=ROOT.parents[2], env=env, bufsize=0)
            os.set_blocking(self.child.stdin.fileno(), False)
            os.set_blocking(self.child.stdout.fileno(), False)
            ready = self.read(time.monotonic() + timeout)
            P.require(ready == dict(schema=P.SCHEMA, seq=-1, identity=self.identity, op='ready'), 'ready')
        except BaseException as error:
            self.error = error
            self.abort()
            raise

    def wait_fd(self, fd: int, event: int, deadline: float) -> None:
        with selectors.DefaultSelector() as selector:
            selector.register(fd, event)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError('ledger_process_deadline')

    def read(self, deadline: float) -> Any:
        raw = bytearray()
        fd = self.child.stdout.fileno()
        while b'\n' not in raw:
            self.wait_fd(fd, selectors.EVENT_READ, deadline)
            chunk = os.read(fd, min(65536, P.MAX_BYTES + 2 - len(raw)))
            P.require(bool(chunk), 'child_eof_commit_uncertain')
            raw.extend(chunk)
            P.require(len(raw) <= P.MAX_BYTES + 1, 'response_size')
        P.require(raw.endswith(b'\n') and raw.count(b'\n') == 1, 'response_framing')
        return P.decoded(bytes(raw))

    def exchange(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        try:
            P.require(not self.closed and not self.busy, 'closed_or_busy')
            raw = P.encoded(value) + b'\n'
            P.require(len(raw) <= P.MAX_BYTES + 1, 'request_size')
            self.busy = True
            deadline, sent = time.monotonic() + self.timeout, 0
            while sent < len(raw):
                self.wait_fd(self.child.stdin.fileno(), selectors.EVENT_WRITE, deadline)
                self.ambiguous_commit = True
                sent += os.write(self.child.stdin.fileno(), raw[sent:])
            reply = self.read(deadline)
            P.require(type(reply) is dict and reply.get('schema') == P.SCHEMA
                      and type(reply.get('seq')) is int and reply['seq'] == self.seq
                      and reply.get('identity') == self.identity, 'reply_identity_sequence')
            return reply
        except BaseException as error:
            self.error = error
            self.abort()
            raise
        finally:
            self.busy = False

    def accept(self, prefix: Any, cutoff: dict[str, int]) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        try:
            codes = tuple(P.encoded(batch) for batch in prefix)
            P.require(len(codes) > len(self.prefix) and codes[:len(self.prefix)] == self.prefix, 'prefix_changed_or_duplicate')
            P.require(all(tuple(e[k] for k in P.IDENTITY_KEYS) == tuple(self.identity)
                          for batch in prefix for e in batch), 'prefix_identity')
            reply = self.exchange(dict(schema=P.SCHEMA, seq=self.seq, identity=self.identity,
                                       op='accept', prefix=prefix, cutoff=cutoff))
            P.require(set(reply) == {'schema', 'seq', 'identity', 'op', 'event_sha256', 'cutoff', 'tensors',
                                    'live_qualified', 'production_permission'}, 'reply_keys')
            P.require(reply['op'] == 'accepted' and reply['event_sha256'] == P.event_sha(prefix)
                      and reply['cutoff'] == cutoff, 'reply_body_binding')
            P.require(reply['live_qualified'] is False and reply['production_permission'] is False, 'reply_authority')
            P.require(type(reply['tensors']) is dict and set(reply['tensors']) ==
                      {'boards', 'queues', 'ledger_values', 'ledger_availability'}, 'tensor_keys')
            self.prefix, self.seq, self.ambiguous_commit = codes, self.seq + 1, False
            return reply
        except BaseException as error:
            self.error = error
            self.abort()
            raise

    def abort(self) -> None:
        if self.child is not None:
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait()
            self.child.stdin.close()
            self.child.stdout.close()
        self.stderr.close()
        self.closed = True

    def close(self) -> None:
        if self.error is not None:
            raise self.error
        if self.closed:
            return
        try:
            reply = self.exchange(dict(schema=P.SCHEMA, seq=self.seq, identity=self.identity, op='close'))
            P.require(reply == dict(schema=P.SCHEMA, seq=self.seq, identity=self.identity, op='closed'), 'close_ack')
            P.require(self.child.wait(timeout=self.timeout) == 0, 'child_exit')
            self.ambiguous_commit = False
        except BaseException as error:
            self.error = error
            raise
        finally:
            self.abort()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, kind: Any, body: Any, trace: Any) -> bool:
        try:
            self.close()
        except BaseException:
            if body is None:
                raise
        return False
