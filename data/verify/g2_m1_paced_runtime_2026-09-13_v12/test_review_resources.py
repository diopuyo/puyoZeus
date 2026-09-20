"""旧監視票/欠測/停止を正常完了へ昇格させない人工保存対照。"""
import json
from pathlib import Path
from typing import Any
import pytest
import review_resources as R


@pytest.mark.parametrize('case', ['normal', 'legacy', 'active', 'missing', 'reused', 'early', 'error'])
def test_saved_monitor(tmp_path: Any, case: str) -> None:
    output = tmp_path / 'output'
    row = dict(schema=R.SCHEMA, pid=123, safety_stop=False, nvidia_smi_status='ok',
        nvidia_smi_error=None, hw_thermal='Not Active', sw_thermal='Not Active', gpu0_temperature_c=50,
        stop_reasons=[], cpu_package_temperature_c=None, process_start_ticks='100')
    supervisor = dict(source='actual_Popen_wait', child_pid=123, child_exit_code=0,
        resource_guard_exit=0, supervisor_error=None, guard_exited_first=False)
    if case == 'legacy': row = dict(pid=123, safety_stop=False)
    if case == 'active': row['sw_thermal'] = 'Active'
    if case == 'missing': row['gpu0_temperature_c'] = None
    if case == 'early': supervisor['guard_exited_first'] = True
    if case == 'error': supervisor['supervisor_error'] = 'guard_crashed'
    rows = [row, dict(row, process_start_ticks='200')] if case == 'reused' else [row]
    Path(str(output) + '.resources.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    Path(str(output) + '.supervisor.json').write_text(json.dumps(supervisor))
    if case == 'normal': assert R.verify(output, 123)['resource_samples'] == 1
    else:
        with pytest.raises((KeyError, ValueError)): R.verify(output, 123)
