"""新予告票の版を原着弾数学へ渡す最小接続検査。盤面/台帳は既存CPU対照。"""
import importlib.util
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'g2_second_terminal_arrival_2026-09-14_v1'
sys.path.insert(0, str(OLD))
import test_terminal_drop_candidate as F
parts = F.parts


def test_new_receipt_contract(parts: Any) -> None:
    spec = importlib.util.spec_from_file_location('_new_warning_math', ROOT / 'observed_terminal_drop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current, ledger, target, evidence = F.fixture(parts)
    with pytest.raises(ValueError, match='terminal_drop_amount_condition'):
        module.candidate(parts, sys.modules['_g2_prefix_commit'], current, ledger, target, evidence)
    evidence['kind'] = 'observed_warning_lower_bound/v2'
    family, receipt = module.candidate(parts, sys.modules['_g2_prefix_commit'], current, ledger, target, evidence)
    assert family.prefix == 2 and receipt['first_drop_candidates'] == 371
    assert receipt['first_drop_nonterminal'] == 0 and not receipt['probability_calibrated']
