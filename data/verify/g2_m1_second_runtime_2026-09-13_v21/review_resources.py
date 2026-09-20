"""ユーザー指定のRAM-only運用。温度の値/列/フラグを成功条件にしない。"""
from __future__ import annotations
import json
from pathlib import Path

RSS_LIMIT_KIB = 8 * 1024 * 1024
AVAILABLE_MIN_KIB = 2 * 1024 * 1024


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('review_resources:' + reason)


def verify(output: Path, pid: int) -> dict:
    supervisor = json.loads(Path(str(output) + '.supervisor.json').read_bytes())
    require(supervisor['source'] == 'actual_Popen_wait' and supervisor['child_pid'] == pid, 'supervisor_source')
    require(supervisor['child_exit_code'] == supervisor['resource_guard_exit'] == 0
            and supervisor['supervisor_error'] is None and supervisor['guard_exited_first'] is False, 'supervisor_exit')
    rows = [json.loads(line) for line in Path(str(output) + '.resources.jsonl').read_text().splitlines()]
    require(bool(rows), 'empty')
    for row in rows:
        require(row['pid'] == pid and row['safety_stop'] is False, 'pid_or_stop')
        require(row['rss_limit_kib'] == RSS_LIMIT_KIB and row['minimum_available_kib'] == AVAILABLE_MIN_KIB, 'limits')
        require(type(row['rss_kib']) is int and 0 <= row['rss_kib'] <= RSS_LIMIT_KIB, 'rss')
        require(type(row['available_kib']) is int and row['available_kib'] >= AVAILABLE_MIN_KIB, 'available')
    return dict(resource_samples=len(rows), thermal_policy='ignored_by_user_instruction', ram_guard_verified=True)
