"""正のama Rust関数を既存のキャッシュ・色変換へ接続する（明示呼出専用）。"""
from __future__ import annotations
from collections.abc import Callable
import json
from time import perf_counter, process_time
from src.ama_plain_reference import Reference

DEFAULT_THREADS = 2
NativeCall = Callable[[str, int], str]


class NativeTransport:
    """既存Referenceが使う1行入出力だけを提供する。子プロセスは起動しない。"""

    def __init__(self, threads: int = DEFAULT_THREADS, native_call: NativeCall | None = None) -> None:
        if threads < 1:
            raise ValueError('Rust amaのスレッド数は1以上が必要')
        if native_call is None:
            from puyo_core import ama_reference_json_py
            native_call = ama_reference_json_py
        self.native_call, self.threads = native_call, threads
        self.stdin = self.stdout = self
        self._input, self._output, self.closed = '', '', False

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError('Rust amaの入口は終了済み')
        self._input += text
        return len(text)

    def flush(self) -> None:
        if not self._input:
            return
        started, cpu_started = perf_counter(), process_time()
        raw = json.loads(self.native_call(self._input.rstrip('\n'), self.threads))
        raw['cpu_sec'] = process_time() - cpu_started
        raw['native_wall_sec'] = perf_counter() - started
        self._output = json.dumps(raw, separators=(',', ':')) + '\n'
        self._input = ''

    def readline(self) -> str:
        result, self._output = self._output, ''
        return result

    def close(self) -> None:
        self.closed = True

    def wait(self) -> int:
        return 0


class NativeReference(Reference):
    def __init__(self, threads: int = DEFAULT_THREADS, cache: bool = True,
                 native_call: NativeCall | None = None) -> None:
        super().__init__(cache=cache, process=NativeTransport(threads, native_call))
