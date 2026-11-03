"""両OSの空きと実ワーカーRSSから安全な並列上限を決める。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

from scripts.image_free_parallel import RESERVE_BYTES, START_BYTES

RSS_MARGIN = 1.25
MIN_SAMPLE_FRAMES = 300


def unoccupied_cpus(cpus: list[int]) -> list[int]:
    """稼働中の比較ワーカーが固定しているコアを避け、保険ジョブを守る。"""
    busy = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path/'cmdline').read_bytes().replace(b'\0', b' ')
            if b'--worker ' not in command or b'--config ' not in command:
                continue
            affinity = os.sched_getaffinity(int(path.name))
            if len(affinity) == 1:
                busy.update(affinity)
        except (OSError, ProcessLookupError):
            continue
    return [cpu for cpu in cpus if cpu not in busy] or cpus


def rss_bytes(pid: int) -> int:
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def capacity(free: int, active: int, rss: int, cpus: int, load: float) -> int:
    """実行済みワーカー分を戻して総容量を算出し、外部CPU負荷を控除する。"""
    if rss <= 0:
        return int(free >= RESERVE_BYTES + START_BYTES)
    cost = math.ceil(rss * RSS_MARGIN)
    memory = max(0, (free + active * rss - RESERVE_BYTES) // cost)
    cpu = max(1, cpus - math.ceil(max(0., load - active)))
    return min(memory, cpu)


@dataclass
class ResourceBudget:
    peak_rss: int = 0
    qualified: bool = False
    reserve_bytes: int = RESERVE_BYTES

    def observe(self, children: dict, output: Path) -> None:
        import json
        for name, child in children.items():
            self.peak_rss = max(self.peak_rss, rss_bytes(child.pid))
            progress = output / name / 'progress.json'
            if progress.exists():
                record = json.loads(progress.read_text())
                measured = record.get('processed_frames', record['frames'])
                self.qualified |= (record.get('pid') == child.pid and self.peak_rss > 0
                                   and (measured >= MIN_SAMPLE_FRAMES
                                        or record.get('complete', False) and measured > 0))

    def limit(self, monitor: object, active: int, cpus: int) -> int:
        free = min(monitor.windows_free, monitor.wsl_free) - (self.reserve_bytes-RESERVE_BYTES)
        return capacity(free, active, self.peak_rss if self.qualified else 0,
                        cpus, os.getloadavg()[0])

    def can_start(self, monitor: object) -> bool:
        cost = max(START_BYTES if not self.qualified else 0,
                   math.ceil(self.peak_rss * RSS_MARGIN))
        return (not monitor.paused and monitor.failures == 0
                and min(monitor.windows_free, monitor.wsl_free) >= self.reserve_bytes + cost)
