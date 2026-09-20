"""v68の元終了検査を保ち、新保存名と今回追加した実行ソースの原文を保持する。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any
import probe_owner_finish as OLD

ROOT = Path(__file__).resolve().parent
SOURCES = ('probe_owner_finish_v2.py', 'probe_owner_finish.py', 'probability_owner.py',
           'probe_positive_v3.py', 'context.py', 'loader.py', 'core_fixture_v2.py',
           'constructor_fixture.py', 'first_static_reflection.py')


def driver() -> Any:
    tree = ast.parse(inspect.getsource(OLD.V.main))
    hits = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
            and node.value == 'POSITIVE_CORE_PROBE_v3.json']
    assert len(hits) == 1
    hits[0].value = 'POSITIVE_CORE_PROBE_v69.json'
    namespace = dict(vars(OLD.V))
    exec(compile(ast.fix_missing_locations(tree), OLD.V.__file__, 'exec'), namespace)
    return namespace['main']


def save_sources(before: dict[str, bytes], error: Any) -> None:
    output = OLD.BASE / sys.argv[1]
    assert output.parent == OLD.BASE and output.is_dir()
    folder = output / 'new_finish_source'
    folder.mkdir(exist_ok=False)
    hashes = {}
    for name, raw in before.items():
        (folder / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    unchanged = all((ROOT / name).read_bytes() == raw for name, raw in before.items())
    with (output / 'OWNER_FINISH_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(before_sha256=hashes, unchanged=unchanged, original_error=error,
                       quality_gate_clear=False), stream, indent=2)
    assert unchanged, 'owner_finish_source_changed'


def main() -> None:
    before = {name: (ROOT / name).read_bytes() for name in SOURCES}
    error = None
    try:
        with ExitStack() as stack:
            stack.callback(setattr, OLD, 'original_driver', OLD.original_driver)
            OLD.original_driver = driver
            OLD.main()
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        if '--preflight' not in sys.argv:
            try:
                save_sources(before, error)
            except BaseException as save_error:
                if error is None:
                    raise
                print(json.dumps(dict(source_save_error=repr(save_error), original_error=error)), file=sys.stderr)


if __name__ == '__main__':
    main()
