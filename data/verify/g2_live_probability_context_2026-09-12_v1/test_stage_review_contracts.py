"""Opus108未読条件を元本体で確認。計算/構造の検証ではない。"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import traceback
from types import SimpleNamespace as N
import pytest
import probability_saved as S

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'


def test_actual_multiline_evidence_decoder() -> None:
    output = BASE / 'prefix_cpu_v70_lower_finish'
    name = 'EMPTY_TAIL_LIVE_EVIDENCE.json'
    raw = (output / name).read_text()
    assert len(raw.splitlines()) > 1
    assert S.read(output, name) == json.loads(raw)


def test_launcher_prefix_through_original_main(tmp_path: Path) -> None:
    command = next(line for line in (ROOT / 'run_stage_finish.sh').read_text().splitlines()
                   if line.startswith('timeout '))
    argv = shlex.split(command)
    script = next(i for i, word in enumerate(argv) if word.endswith('/probe_stage_finish.py'))
    prefix = argv[script + 1]
    assert prefix == 'prefix_cpu_v71_stage_finish'
    path = ROOT.parent / 'g2_empty_tail_finalizer_2026-09-10_v1/run_fused.py'
    fn = next(node for node in ast.parse(path.read_bytes()).body
              if isinstance(node, ast.FunctionDef) and node.name == 'main')
    ns = dict(ROOT=tmp_path, EMPTY=tmp_path, ROLLING=tmp_path,
              sys=N(argv=['probe', prefix]), hashlib=hashlib, time=time,
              os=os, traceback=traceback, json=json, execute=lambda: 0)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    assert ns['main']() == 0
    assert (tmp_path / prefix / 'FUSION_RESULT.json').is_file()
    assert not (tmp_path / 'prefix_cpu_v70_lower_finish').exists()
    with pytest.raises(AssertionError):
        ns['main']()
