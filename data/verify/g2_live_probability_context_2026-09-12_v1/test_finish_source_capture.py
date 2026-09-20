"""新driverの追加ソース原文保存だけを一時領域で検査。実runの証明ではない。"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

SOURCE = Path(__file__).resolve().parent / 'probe_owner_finish_v2.py'


def function(root: Path, base: Path) -> Any:
    tree = ast.parse(SOURCE.read_bytes())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'save_sources')
    namespace = dict(ROOT=root, OLD=N(BASE=base), Any=Any, json=json, hashlib=hashlib, sys=sys)
    exec(compile(ast.Module([node], []), str(SOURCE), 'exec'), namespace)
    return namespace['save_sources']


@pytest.mark.parametrize('changed', (False, True))
def test_original_bytes_preserved_and_source_change_detected(tmp_path: Path, monkeypatch: Any, changed: bool) -> None:
    source, base = tmp_path / 'src', tmp_path / 'runs'
    source.mkdir()
    output = base / 'case'
    output.mkdir(parents=True)
    raw = b'original\r\n'
    (source / 'sample.py').write_bytes(b'changed\n' if changed else raw)
    monkeypatch.setattr(sys, 'argv', ['probe', 'case'])
    invoke = function(source, base)
    if changed:
        with pytest.raises(AssertionError, match='owner_finish_source_changed'):
            invoke({'sample.py': raw}, 'original_failure')
    else:
        invoke({'sample.py': raw}, None)
    assert (output / 'new_finish_source/sample.py').read_bytes() == raw
    saved = json.loads((output / 'OWNER_FINISH_SOURCE.json').read_bytes())
    assert saved['unchanged'] is not changed
    assert saved['before_sha256']['sample.py'] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        invoke({'sample.py': b'do not overwrite'}, None)
    assert (output / 'new_finish_source/sample.py').read_bytes() == raw
