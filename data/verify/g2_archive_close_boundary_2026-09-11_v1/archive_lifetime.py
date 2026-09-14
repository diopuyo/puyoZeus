"""元型moduleの寿命を限定延長し、登録解除後も全値/参照/出所検査を行う。"""
from __future__ import annotations
from copy import copy
import hashlib
from pathlib import Path
import sys
from types import FunctionType, MethodType, SimpleNamespace as N
from typing import Any

SOURCE = Path(__file__).resolve().parent.parent / 'g2_empty_tail_archive_candidate_2026-09-10_v1/empty_archive.py'
SOURCE_SHA = '2260627ec660d329c28785cb02ba1e67873dda6497dda5ac61296014921217a4'
KEY = '_g2_archive_lifetime'


def clone(function: Any, bindings: dict[str, Any]) -> Any:
    assert set(bindings) <= set(function.__code__.co_names), 'archive_lifetime_call_graph'
    result = FunctionType(function.__code__, dict(function.__globals__, **bindings),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    assert result.__closure__ is function.__closure__ and result.__defaults__ is function.__defaults__
    return result


def same_fields(value: Any, fields: Any) -> None:
    assert set(vars(value)) == set(fields), 'archive_lifetime_fields'
    assert all(vars(value)[key] is item for key, item in fields.items()), 'archive_lifetime_identity'


class Verifier:
    def __init__(self, archive: Any) -> None:
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
        module = sys.modules[type(archive).__module__]
        assert Path(module.__file__).resolve() == SOURCE and type(archive) is module.Archive
        archive.verify()  # 登録が生きている時点で元の全検査を通す。
        self.archive, self.original, self.extra = archive, archive.original, archive.extra
        self.modules = dict(sys.modules)  # 強参照で型定義だけを延命。登録表自体は変更しない。
        self.extra_fields = None if self.extra is None else dict(vars(self.extra))
        self.runtime_fields = None if self.extra is None else dict(vars(self.extra.runtime))
        self.private, self.check = self.build(module)
        self.verify(archive)

    def build(self, module: Any) -> tuple[Any, Any]:
        if self.extra is None:
            return None, None
        defined = clone(module.defined, {'sys': N(modules=self.modules)})
        runtime = copy(self.extra.runtime)
        project = MethodType(clone(module.RuntimeEvidence.project, {'defined': defined}), runtime)
        runtime.project = project
        assert runtime.project is project and runtime.sealed is True
        private = copy(self.extra)
        private.runtime = runtime
        bindings = dict(binding_values=clone(module.binding_values, {'defined': defined}),
                        column_digest=clone(module.column_digest, {'defined': defined}))
        check = MethodType(clone(module.Extra.verify, bindings), private)
        return private, check

    def verify(self, archive: Any) -> None:
        # id番号だけでなく強参照そのものを比較し、GC再利用を許さない。
        assert archive is self.archive and archive.original is self.original and archive.extra is self.extra
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
        self.original.verify()
        if self.extra is not None:
            same_fields(self.extra, self.extra_fields)
            same_fields(self.extra.runtime, self.runtime_fields)
            self.check()
