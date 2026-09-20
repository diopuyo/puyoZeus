"""独立提案の旧2反例を実validatorで閉じる。factory carrierは人工のまま。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any
import pytest

ROOT=Path(__file__).resolve().parent


def reproduction() -> Any:
    spec=importlib.util.spec_from_file_location('_closed_opus_reproduction',ROOT/'opus_review_v1/reproduce.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_old_current_key_with_new_key_rejected() -> None:
    with pytest.raises(AssertionError,match='mixed_private_current_sources'):
        reproduction().stale_world()


def test_current_policy_rebinding_rejected_after_normal_control() -> None:
    # carrier内で正常検査を先に通す。実際の再束縛検査の行で拒否することを確認する。
    module=reproduction()
    original=module.carrier
    def finished(ctx: Any, C: Any) -> Any:
        factory,binding,data=original(ctx,C)
        for refs in factory.controller.empty_completion_refs.values(): refs[6]['frame']=None
        return factory,binding,data
    module.carrier=finished
    with pytest.raises(AssertionError) as caught:
        module.policy_rebind()
    assert 'current_policy[:len(value[' in str(caught.traceback[-1].statement)
