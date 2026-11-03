"""統合回帰の起動・終了時の資源条件をpytestの外部原票へ残す。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

POOL_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def save(stage: str) -> None:
    """本体を変更せず、nice・空きメモリ・BLAS設定の実値を取得する。"""
    directory = Path(os.environ['INTEG_RESOURCE_OUTPUT'])
    directory.mkdir(parents=True, exist_ok=True)
    available = next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    row = dict(stage=stage, nice=os.getpriority(os.PRIO_PROCESS, 0),
               available_bytes=int(available)*1024, pools={key: os.environ.get(key) for key in POOL_ENV})
    (directory/f'resource_{stage}.json').write_text(json.dumps(row, indent=2)+'\n')
    assert row['nice'] == 10 and row['available_bytes'] >= 6*1024**3
    assert all(value == '1' for value in row['pools'].values())


def pytest_sessionstart(session: Any) -> None:
    """テスト収集前の実値を記録する。"""
    save('start')


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """テストによる資源設定の副作用が残っていないことを確認する。"""
    save('finish')
