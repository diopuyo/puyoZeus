"""保存原経路のworld/PB検査。新しい動画処理やwriterは起動しない。"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import run_pb as R
import world_verify as W

ROOT = Path(__file__).resolve().parent
MAX_FUNCTION_LINES = 50


def inputs() -> dict[str, Any]:
    result = R.inputs()
    hidden = R.read(R.SOURCE / 'COMBINED_HIDDEN.json')
    result.update(conditional_rows=R.read(R.SOURCE / 'ADMISSION.json')['admission'],
        hidden_history=hidden['history'], hidden_lifetime=hidden['lifetimes'])
    return result


def main() -> int:
    output = ROOT / sys.argv[1]
    assert output.parent == ROOT and output.name.startswith('world_cpu_v')
    output.mkdir(exist_ok=False)
    paths = [*ROOT.glob('*.py'), ROOT / 'PLAN.md', *W.G.STAGE1.glob('*.py'),
        *W.G.ORIGIN.glob('*.py'), *W.G.SETTLE.glob('*.py')]
    paths += [R.SOURCE / name for name in (*R.NAMES, 'ADMISSION.json')]
    before = {str(path): R.sha(path) for path in paths}
    R.write(output / 'INPUTS.json', before)
    started, code = time.monotonic(), 0
    try:
        for path in ROOT.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.FunctionDef):
                    assert node.end_lineno-node.lineno+1 <= MAX_FUNCTION_LINES, (path.name, node.name)
        result = W.verify_world(**inputs())
        assert len(result['conditional_frames']) == 18 and result['prepared_worlds'] == 5
        assert result['origins_verified'] == result['settlements_verified'] == 1
        assert result['world_PB_verified'] and not result['runtime_finalization_allowed']
    except BaseException:
        result, code = dict(error=traceback.format_exc()), 1
    unchanged = all(R.sha(Path(path)) == digest for path, digest in before.items())
    result.update(pid=os.getpid(), seconds=time.monotonic()-started, exit_code=code,
        input_sha_unchanged=unchanged, input_files=len(before), original_updates_rerun=0)
    R.write(output / 'RESULT.json', result)
    R.write(output / 'INDEX.json', {p.name: R.sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != 'backend'}), flush=True)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
