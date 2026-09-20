"""原fixture設定を再用し、通常110/条件86の両段を一回ずつ保存検査する。"""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterator
import fusion as F

ROOT = Path(__file__).resolve().parent


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


@contextmanager
def fixtures() -> Iterator[Any]:
    with F.modules() as (stage1, world):
        world.G.L.libraries()
        previous = sys.modules.get('compat')
        try:
            sys.modules['compat'] = stage1
            fixture = stage1.F.load('_fusion_original_fixture', F.STAGE1 / 'test_stage1.py')
            yield stage1, world, fixture
        finally:
            if previous is None:
                sys.modules.pop('compat', None)
            else:
                sys.modules['compat'] = previous


def arguments(fixture: Any, name: str) -> dict[str, Any]:
    value = fixture.data(name)
    hidden = (fixture.read(value['output'] / 'COMBINED_HIDDEN.json') if name == 'prefix_cpu_v2'
        else dict(events=[], history=[], lifetimes=[]))
    value.update(hidden_events=hidden['events'], hidden_history=hidden['history'],
        hidden_lifetime=hidden['lifetimes'], outer_rows=fixture.read(value['output'] / 'POSTCOMMIT_CONSUMER_ROWS.json'))
    return value


def main() -> int:
    output = ROOT / sys.argv[1]
    assert output.parent == ROOT and output.name.startswith('saved_cpu_v')
    output.mkdir(exist_ok=False)
    started, code, result = time.monotonic(), 0, {}
    try:
        with fixtures() as (stage1, world, fixture):
            paths = [*ROOT.glob('*.py'), ROOT / 'PLAN.md', *F.STAGE1.glob('*.py'), *F.WORLD.glob('*.py')]
            pins = stage1.F.guards() | {str(p): stage1.F.sha(p) for p in paths}
            for name in ('cpu_v2', 'prefix_cpu_v2'):
                pins.update({str(p):stage1.F.sha(p) for p in (fixture.SHARED / name).iterdir() if p.is_file()})
            write(output / 'INPUTS.json', pins)
            for name in ('cpu_v2', 'prefix_cpu_v2'):
                result[name] = F.audit(**arguments(fixture, name))
            assert result['prefix_cpu_v2']['world_PB_verified']
            assert not result['prefix_cpu_v2']['runtime_finalization_allowed']
            assert len(result['prefix_cpu_v2']['current_proofs']) == 1
            assert all(stage1.F.sha(Path(p)) == digest for p, digest in pins.items())
            result['input_sha_unchanged'], result['input_files'] = True, len(pins)
    except BaseException:
        result['error'], code = traceback.format_exc(), 1
    result.update(pid=os.getpid(), seconds=time.monotonic()-started, exit_code=code, original_updates_rerun=0)
    write(output / 'RESULT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cpu_v2','prefix_cpu_v2')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
