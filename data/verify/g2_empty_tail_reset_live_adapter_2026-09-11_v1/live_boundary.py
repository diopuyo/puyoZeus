"""原公開結合を保持し、reset前の正当な整数公開を消さずに全件照合する。"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parent.parent/'g2_empty_tail_reset_integration_2026-09-11_v1/publication_boundary/verify.py'
SOURCE_SHA = '3bf639fb95c5d48e21ab551d957575e964e2adadaaa6371d32966e2da0d361c7'


def checker() -> Any:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA: raise ValueError('original_publication_source')
    spec = importlib.util.spec_from_file_location('_live_reset_original_publication',SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(raw)
    target = 'actual == issued and all((f > base[\'frame\'] for f in actual))'
    found = [n for n in ast.walk(tree) if isinstance(n,ast.Assert) and ast.unparse(n.test)==target]
    if len(found)!=1: raise ValueError('original_publication_anchor')
    found[0].test = ast.parse('actual == issued',mode='eval').body
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),namespace)
    return namespace['check']


def check(consumer: Any, history: Any, journal: Any, recovery: Any) -> Any:
    result = checker()(consumer,history,journal,recovery)
    baseline = result['baseline_frame']
    old = [frame for frame in result['issued_frames'] if frame<baseline]
    new = [frame for frame in result['issued_frames'] if frame>baseline]
    if len(old)+len(new)!=len(result['issued_frames']): raise ValueError('baseline_publication')
    return result|dict(prior_issued_frames=old,new_issued_frames=new,
                       prior_integer_publications_retained=True,quality_gate_clear=False)
