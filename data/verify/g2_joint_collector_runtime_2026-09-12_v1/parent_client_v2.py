"""検収済み親v1を保持し、実起動pathのpinを明示する採録用入口。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'g2_joint_ledger_engine_2026-09-12_v1'
sys.path.insert(0, str(PARENT))
import parent_transport as T


def worker_path() -> Path:
    T.guard()
    path = PARENT / 'worker_v2.py'
    T.P.require(path in T.PINS and hashlib.sha256(path.read_bytes()).hexdigest() == T.PINS[path],
                'runtime_worker_path_pin')
    return path


class Client(T.Client):
    def __init__(self, part: Path, identity: tuple[str, str, str], *, timeout: float = T.P.TIMEOUT) -> None:
        self.runtime_stage = 'starting'
        self.failed_request_seq: int | None = None
        try:
            T.T.Client.__init__(self, part, identity, timeout=timeout, worker=worker_path())
            self.runtime_stage = 'ready'
        except BaseException as error:
            if hasattr(self, 'stderr'):
                self.fail(error)
            raise

    def join(self, producer: dict, observation: dict) -> dict:
        self.failed_request_seq = self.seq
        self.runtime_stage = 'joining'
        result = super().join(producer, observation)
        self.runtime_stage = 'accepted_not_yet_published'
        return result

    def fail(self, error: BaseException) -> None:
        super().fail(error)
        detail = dict(stage=self.runtime_stage, failed_request_seq=self.failed_request_seq,
                      accepted_count=self.seq, reason=str(error), quality_gate_clear=False)
        try:
            with self.part.with_suffix(self.part.suffix + '.runtime_failure.json').open('xb') as stream:
                stream.write(T.P.encoded(detail))
        except BaseException as caught:
            self.runtime_diagnostic_error = caught
