"""新入口で元source/voteの実codeも確認し、同globals別実装の混入を拒否する。"""
from __future__ import annotations
import hashlib
import marshal
from pathlib import Path
from types import CodeType
from typing import Any

PATH = Path(__file__).resolve().parent.parent / 'g2_hidden_tail_candidate_2026-09-10_v1/tail_witness.py'
SHA = '0aebf6bccb89eff1ae6fc5c8053b04ed02c0fbd11e346d750cf0274564828174'


def verify(history: Any) -> None:
    witness = history.T
    assert Path(witness.__file__).resolve() == PATH
    source = PATH.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA, 'suffix_origin_file'
    for name in ('source', 'vote'):
        actual = getattr(witness, name)
        assert actual.__globals__ is vars(witness) and actual.__closure__ is None
        code = compile(source, actual.__code__.co_filename, 'exec', dont_inherit=True)
        declared = [part for part in code.co_consts if isinstance(part, CodeType) and part.co_name == name]
        assert len(declared) == 1 and marshal.dumps(declared[0]) == marshal.dumps(actual.__code__), 'suffix_origin_code'
