"""元Archive/Basis fixtureで終了後receipt再突入を再現。実同call検査とは別。"""
from pathlib import Path
import sys
from typing import Any
import pytest
import probability_owner as P

ROOT = Path(__file__).resolve().parent.parent / 'g2_archive_close_boundary_2026-09-11_v1'
sys.path.insert(0, str(ROOT))
import archive_lifetime as L
import test_original_close as F

modules, fixture, actual = F.modules, F.fixture, F.actual


def test_legacy_receipt_reenters_closed_type_registry(actual: Any) -> None:
    F.F.registry(actual)
    archive = F.F.X.Archive(actual.local.controller, actual.binding)
    assert archive.receipt()['empty_tail']['preservation_verified']
    verifier = L.Verifier(archive)
    name = type(actual.basis).__module__
    original = sys.modules.pop(name)
    try:
        verifier.verify(archive)
        with pytest.raises(AssertionError, match='empty_archive_basis_type'):
            archive.receipt()
    finally:
        sys.modules[name] = original


@pytest.mark.parametrize('fault', ('none', 'basis', 'column'))
def test_retained_receipt_preserves_full_content_and_rejects_tamper(actual: Any, fault: str) -> None:
    F.F.registry(actual)
    archive = F.F.X.Archive(actual.local.controller, actual.binding)
    expected = archive.receipt()
    verifier = L.Verifier(archive)
    name = type(actual.basis).__module__
    original = sys.modules.pop(name)
    try:
        actual_receipt = P.retained_receipt(verifier, archive)
        assert actual_receipt == expected and set(actual_receipt) == set(expected)
        assert {key: value for key, value in actual_receipt.items() if key != 'empty_tail'} == verifier.original.receipt()
        assert name not in sys.modules
        if fault == 'basis':
            actual.basis.frame += 2
        elif fault == 'column':
            actual.local.controller.empty_tail_prepop_checks[0]['saved_only'] = False
        if fault != 'none':
            reason = 'empty_archive_basis_changed' if fault == 'basis' else 'empty_archive_column_changed'
            with pytest.raises(AssertionError, match=reason):
                P.retained_receipt(verifier, archive)
        assert name not in sys.modules
    finally:
        sys.modules[name] = original
