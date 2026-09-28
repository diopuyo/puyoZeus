"""WSLの専用process groupだけを計装・終了確認する。"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

KIB = 1024
SMI_TIMEOUT_SEC = 2.


def group_members(group: int) -> list[dict]:
    output = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[2]) != group:
                continue
            status = dict(line.split(':', 1) for line in (path/'status').read_text().splitlines())
            output.append(dict(pid=int(path.name), parent=int(stat[1]), state=stat[0],
                rss_bytes=int(status.get('VmRSS', '0 kB').split()[0])*KIB,
                handles=len(list((path/'fd').iterdir())),
                command=(path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')))
        except (OSError, ValueError, ProcessLookupError):
            continue
    return output


def vram(pids: set[int]) -> int | None:
    """CPU条件でもGPU使用を監査する。照会不能は0と偽らずnullにする。"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=SMI_TIMEOUT_SEC)
        if result.returncode:
            return None
        rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
        return sum(int(memory.strip()) for pid, memory in rows if int(pid.strip()) in pids)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def sample(group: int) -> dict:
    members = group_members(group)
    return dict(at=time.perf_counter(), unix=time.time(), loadavg=os.getloadavg(),
        members=members, rss_bytes=sum(p['rss_bytes'] for p in members),
        handles=sum(p['handles'] for p in members), handle_kind='Linux /proc/PID/fd',
        vram_mib=vram({p['pid'] for p in members}))
