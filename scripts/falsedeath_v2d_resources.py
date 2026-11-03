"""WSL資源と、このworktreeの計算プロセスのnice・BLAS制限を記録する。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from scripts.falsedeath_collect_20261001 import OUT, ROOT

BLAS_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')
MIN_MEMORY_KIB = 6*1024**2
REQUIRED_NICE = 10


def processes() -> list[dict]:
    """同じ作業ディレクトリのPythonだけを読み、環境変数は指定3個に限定する。"""
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc/'cwd').resolve() != ROOT:
                continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if not argv or b'python' not in argv[0]:
                continue
            entries = (proc/'environ').read_bytes().split(b'\0')
            env = dict(v.decode().split('=',1) for v in entries if b'=' in v)
            nice = os.getpriority(os.PRIO_PROCESS, int(proc.name))
            blas = {k:env.get(k) for k in BLAS_KEYS}
            assert nice == REQUIRED_NICE and set(blas.values()) == {'1'}, (proc.name,nice,blas)
            rows.append(dict(pid=int(proc.name), nice=nice, blas=blas,
                             command=[a.decode() for a in argv if a]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rows


def main() -> None:
    """上限を変更せず、現在値を追記専用の原票へ保存する。"""
    memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])
    assert available >= MIN_MEMORY_KIB
    row = dict(utc=datetime.now(timezone.utc).isoformat(), available_kib=available, processes=processes())
    with (OUT/'v2d/RESOURCE_SAMPLES.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
