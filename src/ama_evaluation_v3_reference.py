"""正のC++だけを使う常駐通信と工程別CPU計測。"""
from __future__ import annotations
from collections import Counter, OrderedDict
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

REFERENCE = Path('logs/ama_eval_v3/reference')
STAGES = ('extension', 'x_enumeration', 'y_search', 'z_search', 'other')
NODE_CACHE_LIMIT = 256


class ReferenceMismatch(RuntimeError):
    """正のamaの順位・合法手・遷移の食い違い。欠測に変換しない。"""


class Reference:
    def __init__(self, binary: Path = REFERENCE, process: object | None = None) -> None:
        self.process = process if process is not None else subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.cpu_sec, self.calls = 0., 0
        self.stage = 'extension'
        self.metrics = {k: dict(cpu_sec=0., think_calls=0, node_calls=0) for k in STAGES}
        self.events = Counter()
        self.cache: dict = {}
        self.node_cache: OrderedDict = OrderedDict()

    @contextmanager
    def phase(self, name: str):
        previous = self.stage
        self.stage = name
        try:
            yield
        finally:
            self.stage = previous

    def query(self, board: list, visible: list, colors: tuple, op: str,
              **options: int) -> dict:
        if not 1 <= len(visible) <= 3:
            self.fail('visible_count')
        mapping = {0: 0, 9: 9, **{c: i+1 for i, c in enumerate(colors)}}
        request = dict(board=[mapping[c] for c in board],
            visible=[[mapping[c] for c in p] for p in visible], op=op, **options)
        key = json.dumps(request, separators=(',', ':'))
        cache = self.node_cache if op == 'node' else self.cache
        if key not in cache:
            self.process.stdin.write(key+'\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                self.fail('reference_eof')
            result = json.loads(line)
            if 'error' in result:
                self.fail(result['error'])
            elapsed = result.pop('cpu_sec')
            self.cpu_sec += elapsed
            self.calls += int(op == 'ranked')
            stage = 'x_enumeration' if op == 'node' and self.stage == 'extension' else self.stage
            self.metrics[stage]['cpu_sec'] += elapsed
            self.metrics[stage]['think_calls' if op == 'ranked' else 'node_calls'] += 1
            cache[key] = result
            if op == 'node' and len(cache) > NODE_CACHE_LIMIT:
                cache.popitem(last=False)
        # 正規色のキャッシュを保持し、呼出側の色へ都度戻す。
        result = json.loads(json.dumps(cache[key]))
        reverse = {v: k for k, v in mapping.items()}
        for value in result.get('moves', [])+[result.get('initial', {})]:
            for name in ('board', 'placed'):
                if name in value:
                    value[name] = [reverse[c] for c in value[name]]
        return result

    def ranked(self, board: list, visible: list, colors: tuple) -> list:
        return self.query(board, visible, colors, 'ranked')['candidates']

    def fail(self, reason: str) -> None:
        self.events['exception:'+reason] += 1
        raise ReferenceMismatch(reason)

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            self.fail('reference_exit')
