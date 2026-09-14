"""旧形式だけで熱監視済みとしない。実終了前の全監視票と外側waitを検査する。"""
from __future__ import annotations
import json
from pathlib import Path

SCHEMA = 'g2_ram_thermal_guard_v1'


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
        require(row['schema'] == SCHEMA and row['pid'] == pid and row['safety_stop'] is False, 'schema_or_stop')
        require(row['nvidia_smi_status'] == 'ok' and row['nvidia_smi_error'] is None, 'gpu_status')
        require(row['hw_thermal'] == row['sw_thermal'] == 'Not Active', 'thermal')
        require(type(row['gpu0_temperature_c']) is int and row['gpu0_temperature_c'] >= 0, 'temperature')
        require(row['stop_reasons'] == [] and row['cpu_package_temperature_c'] is None, 'reasons')
        require(type(row['process_start_ticks']) is str and row['process_start_ticks'].isdigit(), 'identity')
    require(len({row['process_start_ticks'] for row in rows}) == 1, 'pid_reused')
    return dict(resource_samples=len(rows), gpu_temperature_max=max(r['gpu0_temperature_c'] for r in rows),
                thermal_active_samples=0, cpu_package_temperature_verified=False)
