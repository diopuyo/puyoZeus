"""元集計へ保存原Jと同runの発火証拠を渡す。証拠を別runから補わない。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import sys
from typing import Any
import common as K

ROOT = K.VERIFY/'g2_repeated_finalizer_compatibility_2026-09-10_v1'


def evaluate(goals: Any, rows: list[Any], legal: Any, output: Any, state: Any) -> dict[str, Any]:
    evidence = state['repeated_firing_constructor']
    K.require(evidence['installed'] and evidence['closed'] and evidence['references_restored'],
        'finalizer_constructor_unclosed')
    journal = [json.loads(line) for line in (output/'atomic_journal.jsonl').read_text().splitlines()]
    with ExitStack() as stack:
        paths = list(sys.path)
        stack.callback(setattr, sys, 'path', paths)
        sys.path.insert(0, str(ROOT))
        module = K.load('_live_repeated_finalizer', ROOT/'compat.py', stack)
        return module.evaluate(goals, rows, legal, output,
            firing_rows=evidence['rows'], journal_rows=journal)
