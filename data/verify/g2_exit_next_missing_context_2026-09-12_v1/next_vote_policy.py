"""DNEXT由来の一括抑止時だけ退出資格を分離するCPU候補。原NEXT値は変更しない。"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import inspect
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
VOTES = ROOT.parent / 'g2_transition_endpoint_votes_2026-09-11_v1'
sys.path.insert(0,str(VOTES))
import prior_votes_v2 as OLD

FIXED_THRESHOLD = 8.0  # 原NextSlideDetector.DEFAULT_DIFF_THRESHOLD。実接続時も同値を検査。
VALID_COLORS = frozenset(range(1,6))


@dataclass(frozen=True)
class ExitEvidence:
    frame: int
    previous_frame: int
    raw_next: tuple[int,int]
    slide_motion: bool
    next_diff: float
    dnext_diff: float
    fixed_threshold: float


def valid_pair(value: Any) -> bool:
    return type(value) is tuple and len(value) == 2 and all(type(c) is int and c in VALID_COLORS for c in value)


def missing_qualified(current: Any, pair: tuple[int,int]) -> bool:
    proof = getattr(current,'exit_evidence',None)
    if not isinstance(proof,ExitEvidence):
        return False
    values = (proof.next_diff,proof.dnext_diff,proof.fixed_threshold)
    if not all(type(v) is float and math.isfinite(v) and v >= 0 for v in values):
        return False
    return (type(proof.frame) is int and type(proof.previous_frame) is int
            and proof.frame == current.frame and proof.previous_frame == current.frame-OLD.OLD.FRAME_STRIDE
            and valid_pair(proof.raw_next) and proof.raw_next == pair and proof.slide_motion is True
            and proof.fixed_threshold == FIXED_THRESHOLD
            and proof.next_diff < proof.fixed_threshold <= proof.dnext_diff)


def next_consistent(ticks: list[Any]) -> bool:
    if len(ticks) != OLD.WINDOW:
        return False
    pair = ticks[0].next_pair
    if not valid_pair(pair) or any(t.next_pair != pair for t in ticks[:-1]):
        return False
    return missing_qualified(ticks[-1],pair) if ticks[-1].next_pair is None else ticks[-1].next_pair == pair


def build_original_guard() -> Any:
    """原関数のNEXT比較1式だけを置換し、他条件と原3過去票の返却を保つ。"""
    assert Path(OLD.__file__).resolve() == VOTES/'prior_votes_v2.py'
    tree = ast.parse(inspect.getsource(OLD.past_history))
    target = 'any(tick.next_pair != previous.next_pair for tick in ticks)'
    expected = ast.dump(ast.parse(target,mode='eval').body)
    hits = 0
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.dump(node.test) == expected:
            node.test = ast.parse('not next_consistent(ticks)',mode='eval').body
            hits += 1
    assert hits == 1
    namespace = dict(vars(OLD),next_consistent=next_consistent)
    exec(compile(ast.fix_missing_locations(tree),str(ROOT/'next_vote_policy.py'),'exec'),namespace)
    return namespace['past_history']


past_history = build_original_guard()
