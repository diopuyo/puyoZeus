"""既存の実行系を変更せず、凍結配布と同じコードだけを再利用する。"""
from __future__ import annotations
from hashlib import file_digest, sha256
import json
import marshal
from pathlib import Path
import types

PYC_HEADER_BYTES = 16
MAGIC_BYTES = 4
CODE_FIELDS = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
    'co_stacksize', 'co_flags', 'co_code', 'co_names', 'co_varnames', 'co_name',
    'co_qualname', 'co_firstlineno', 'co_linetable', 'co_exceptiontable',
    'co_freevars', 'co_cellvars')


def same_code(left: object, right: object) -> bool:
    """コード内の実ファイル名だけを除外し、命令・定数・行情報を全て比較する。"""
    if isinstance(left, types.CodeType):
        return (isinstance(right, types.CodeType)
            and all(getattr(left, key) == getattr(right, key) for key in CODE_FIELDS)
            and same_code(left.co_consts, right.co_consts))
    if isinstance(left, tuple):
        return (isinstance(right, tuple) and len(left) == len(right)
            and all(same_code(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def equivalent_pyc(expected: bytes, actual: bytes) -> bool:
    """検証済みソースに対応するpycだけ、ヘッダーと保存パスの差を許す。"""
    if expected[:MAGIC_BYTES] != actual[:MAGIC_BYTES]:
        return False
    try:
        left, right = (marshal.loads(data[PYC_HEADER_BYTES:]) for data in (expected, actual))
        return isinstance(left, types.CodeType) and same_code(left, right)
    except (EOFError, ValueError, TypeError):
        return False


def validate(bundle: Path, runtime: Path) -> dict:
    """全ファイルを照合する。source/DLL/EXEの差や欠落があれば使用を拒否する。"""
    manifest = json.loads((bundle/'obs_runtime_manifest.json').read_text(encoding='utf-8'))
    different, checked, exact = [], [], 0
    for name, expected in manifest['files'].items():
        path = runtime/name
        with path.open('rb') as stream:
            actual = file_digest(stream, 'sha256').hexdigest()
        if actual == expected:
            exact += 1
        else:
            different.append((name, actual))
    for name, actual in different:
        path = Path(name)
        source = path.parent.parent/(path.name.split('.cpython-')[0]+'.py')
        if path.suffix != '.pyc' or source.as_posix() not in manifest['files']:
            raise ValueError(f'実行系の不一致: {name}')
        expected_bytes = (bundle/'python'/name).read_bytes()
        if sha256(expected_bytes).hexdigest() != manifest['files'][name]:
            raise ValueError(f'比較元pycが破損: {name}')
        if not equivalent_pyc(expected_bytes, (runtime/name).read_bytes()):
            raise ValueError(f'pycコードの不一致: {name}')
        checked.append(dict(name=name, actual_sha256=actual, code_fields_equal=True))
    return dict(passed=True, runtime=str(runtime), files=len(manifest['files']),
                exact_files=exact, pyc_metadata_only=checked,
                policy='pycヘッダー・コード保存パス以外は全て一致。ソースと実行バイナリは完全一致。')
