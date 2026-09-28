"""LinuxのCPU会計。1コアを1.0として自process groupと外部を分離する。"""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

SAMPLE_SECONDS = 1.0
MAX_SAMPLE_SECONDS = 2.5
EXTERNAL_CORE_LIMIT = 1.0
STAT_CPU_FIELDS = 8  # guestはuser/niceに含まれるので二重加算しない。


def read_cpu(group: int, proc: Path = Path('/proc'), controlled_groups: tuple = ()) -> dict[str, Any]:
    """PID再利用をstarttimeで区別。終了済processの未回収時間は外部側へ保守的に残る。"""
    ticks = [int(v) for v in (proc/'stat').read_text().splitlines()[0].split()[1:]]
    owned, controlled = {}, {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            row = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            if int(row[2]) == group:
                owned[f'{entry.name}:{row[19]}'] = int(row[11])+int(row[12])
            elif int(row[2]) in controlled_groups:
                controlled[f'{entry.name}:{row[19]}'] = int(row[11])+int(row[12])
        except (OSError, ValueError, IndexError):
            continue
    return dict(at=time.perf_counter(), unix=time.time(), loadavg=os.getloadavg(),
                total=sum(ticks[:STAT_CPU_FIELDS]), idle=ticks[3]+ticks[4], owned=owned, controlled=controlled)


def cpu_interval(previous: dict, current: dict, hz: int,
                 threshold: float = EXTERNAL_CORE_LIMIT) -> dict:
    elapsed = current['at']-previous['at']
    total = current['total']-previous['total']
    busy = total-(current['idle']-previous['idle'])
    own = sum(max(0, value-previous['owned'].get(key, 0))
              for key, value in current['owned'].items())
    valid = 0 < elapsed <= MAX_SAMPLE_SECONDS and total > 0 and busy >= 0 and 0 <= own <= busy+hz
    scale = hz*elapsed if elapsed > 0 else 1
    external = max(0, busy-own)/scale
    controlled_ticks = sum(max(0, value-previous.get('controlled', {}).get(key, 0))
                           for key, value in current.get('controlled', {}).items())
    controlled = controlled_ticks/scale
    valid &= own+controlled_ticks <= busy+hz
    unplanned = max(0, external-controlled)
    return dict(start=previous['at'], end=current['at'], unix=current['unix'],
                elapsed_sec=elapsed, loadavg=current['loadavg'], total_ticks=total,
                busy_ticks=busy, own_ticks=own, own_cores=own/scale,
                external_cores=external, threshold_cores=threshold,
                controlled_ticks=controlled_ticks, controlled_cores=controlled, unplanned_cores=unplanned,
                unplanned_contaminated=bool(valid and unplanned > threshold),
                valid=valid, contaminated=bool(valid and external > threshold))


class CpuSampler:
    def __init__(self, group: int | None = None, threshold: float = EXTERNAL_CORE_LIMIT,
                 controlled_groups: tuple = ()) -> None:
        self.group = group if group is not None else os.getpgrp()
        self.threshold = threshold
        self.hz = os.sysconf('SC_CLK_TCK')
        self.controlled_groups = controlled_groups
        self.previous = read_cpu(self.group, controlled_groups=self.controlled_groups)

    def sample(self) -> dict:
        current = read_cpu(self.group, controlled_groups=self.controlled_groups)
        row = cpu_interval(self.previous, current, self.hz, self.threshold)
        self.previous = current
        return row
