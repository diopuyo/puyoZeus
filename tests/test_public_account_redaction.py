"""公開出力の匿名化と入力 SHA の保護を検証する。"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('account_builder', ROOT / 'packaging/build_public_tree.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
USER = 'private_example'


@pytest.mark.parametrize('prefix', ['/c/Users/', 'C:\\Users\\', '/mnt/c/Users/',
                                    '\\mnt\\c\\Users\\', 'C--Users-', '/home/'])
def test_account_path_forms(prefix: str) -> None:
    raw = (prefix + USER.upper() + '/AppData').encode()
    assert B.redact_account('example.txt', raw, USER) == (prefix + '<user>/AppData').encode()


@pytest.mark.parametrize('quoted', ['', '"', "'"])
def test_shell_placeholder_is_literal(quoted: str) -> None:
    text = f'path={quoted}/c/Users/{USER}/AppData{quoted}\n'
    clean = B.redact_account('tool.sh', text.encode(), USER).decode()
    placeholder = '<user>' if quoted else '"<user>"'
    assert clean == f'path={quoted}/c/Users/{placeholder}/AppData{quoted}\n'


def test_binary_unchanged_utf16_and_xml() -> None:
    raw = b'\x00' + USER.encode()
    assert B.redact_account('image.bin', raw, USER) == raw
    assert B.redact_account('note.txt', USER.encode('utf-16'), USER).decode('utf-16') == '<user>'
    assert B.redact_account('note.xml', f'<a>{USER}</a>'.encode(), USER) == b'<a>&lt;user&gt;</a>'


def test_protected_input_checked_and_output_policy_updated() -> None:
    path = 'tool.sh'
    raw = f'path=/c/Users/{USER}/AppData\n'.encode()
    policy = dict(exclude_parts=[], artifact_prefixes=[], retain_artifacts={},
                  preserve_bytes={path: dict(sha256=hashlib.sha256(raw).hexdigest())})
    files = {path: raw, 'packaging/public_tree_policy.json': json.dumps(policy).encode()}
    clean, audit = B.plan(files, policy, [], private_user=USER)
    exported = json.loads(clean['packaging/public_tree_policy.json'])
    assert exported['preserve_bytes'][path]['sha256'] == hashlib.sha256(clean[path]).hexdigest()
    assert audit['protected_redactions'][path]['input_sha256'] == policy['preserve_bytes'][path]['sha256']
    again, _ = B.plan(clean, exported, [], private_user=USER)
    assert again == clean
    files[path] += b'# changed\n'
    with pytest.raises(ValueError):
        B.plan(files, policy, [], private_user=USER)
