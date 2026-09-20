"""熱条件は判定せず、元RAM保護と実waitを維持する限定検収。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any
import pytest
import review_resources as R
import target_entry as T


@pytest.mark.parametrize('case', ['normal', 'active', 'missing', 'rss', 'available', 'stop'])
def test_saved_policy(tmp_path: Any, case: str) -> None:
    output = tmp_path / 'output'
    row = dict(pid=123, safety_stop=False, rss_kib=1000, available_kib=4000000,
               rss_limit_kib=R.RSS_LIMIT_KIB, minimum_available_kib=R.AVAILABLE_MIN_KIB)
    if case == 'active': row.update(gpu0_temperature_c=99, hw_thermal='Active', sw_thermal='Active')
    if case == 'missing': row.update(gpu0_temperature_c=None, nvidia_smi_status='error')
    if case == 'rss': row['rss_kib'] = R.RSS_LIMIT_KIB + 1
    if case == 'available': row['available_kib'] = R.AVAILABLE_MIN_KIB - 1
    if case == 'stop': row['safety_stop'] = True
    supervisor = dict(source='actual_Popen_wait', child_pid=123, child_exit_code=0,
        resource_guard_exit=0, supervisor_error=None, guard_exited_first=False)
    Path(str(output) + '.resources.jsonl').write_text(json.dumps(row) + '\n')
    Path(str(output) + '.supervisor.json').write_text(json.dumps(supervisor))
    if case in ('normal', 'active', 'missing'): assert R.verify(output, 123)['ram_guard_verified']
    else:
        with pytest.raises(ValueError): R.verify(output, 123)


def test_launcher_uses_original_guard() -> None:
    source = (T.ROOT / 'run_whole_target.sh').read_text()
    assert 'guard=data/verify/g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py' in source
    assert 'taskset' not in source and 'OMP_NUM_THREADS=2' in source
    assert Path(R.__file__).resolve().parent == T.ROOT


def test_real_original_ram_guard(tmp_path: Any) -> None:
    import supervise as S
    runner = str(T.A.SAFETY / 'probe_guard_child.py')
    guard = str(T.ROOT.parent / 'g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py')
    path = tmp_path / 'resources.jsonl'
    result = S.run_pair([sys.executable, runner, '1'], guard, runner, path)
    assert result['child_exit_code'] == result['resource_guard_exit'] == 0
    assert result['supervisor_error'] is None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all('nvidia_smi_status' not in r and r['safety_stop'] is False for r in rows)
    assert all(r['rss_limit_kib'] == R.RSS_LIMIT_KIB for r in rows)
    assert not Path(f"/proc/{result['child_pid']}").exists()
    assert not Path(f"/proc/{result['guard_pid']}").exists()
