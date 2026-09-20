"""元私有配置/自然復帰の意味条件を保持し、kindと保存属性だけを分離する。"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent/'g2_private_suffix_continuation_2026-09-10_v1'
PINS = {'placement.py':'707c0244a870862c7f8d9c0b8903cf70d2a2899176a4c41344843dc1b7d46d61',
    'exit.py':'a2f8b8deff028270ec1b6c46a7c2bd993a6a85a8271e3874a250c90a00732cb9'}
NAMES = {'private_suffix_placement':'empty_tail_placement',
    'private_suffix_basis':'empty_tail_basis',
    'private_suffix_committed_source':'empty_tail_committed_source',
    'hidden_private_suffix_history/v1':'hidden_empty_tail_next_history/v1'}


def transformed(name: str, data: bytes) -> tuple[Any, dict[str,int]]:
    assert name in PINS and hashlib.sha256(data).hexdigest() == PINS[name], 'empty_tail_original_source'
    tree = ast.parse(data,filename=str(ORIGINAL/name))
    counts = {key:0 for key in NAMES}
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and node.attr in NAMES:
            counts[node.attr] += 1
            node.attr = NAMES[node.attr]
        elif isinstance(node,ast.Constant) and type(node.value) is str and node.value in NAMES:
            counts[node.value] += 1
            node.value = NAMES[node.value]
    assert counts['private_suffix_placement'] > 0, 'empty_tail_no_property_renamed'
    return ast.fix_missing_locations(tree),counts


def adapted(original: Any) -> Any:
    path = Path(original.__file__).resolve()
    assert path.parent == ORIGINAL and path.name in PINS
    tree,counts = transformed(path.name,path.read_bytes())
    module = ModuleType('_empty_tail_derived_'+path.stem)
    module.__dict__.update(vars(original))
    exec(compile(tree,str(path),'exec',dont_inherit=True),vars(module))
    module.empty_tail_derivation = dict(source=str(path),sha256=PINS[path.name],renamed=counts)
    return module
