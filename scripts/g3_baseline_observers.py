"""基準armへ当該OCRと元resetの読取だけを追加する。修復は発行しない。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scripts import g3_admission as A
from scripts.g3_agent_review import ROOT, save
from scripts.g3_model_process import loaded

EPOCH = ROOT / 'scripts/match_start_epoch_shadow_v1.py'
EPOCH_SHA = 'd9fbb9f2de11cd48535c64bce51a0d282100aeefcc067632316a51f87cd6fe70'
KEY = 'g3_baseline_observers'


def captured(original: Callable[..., dict], source_id: str, run_id: str,
             *args: Any) -> dict:
    """観測失敗でも元resetを阻害せず、旧video38の認定名を持ち越さない。"""
    try:
        value = original(*args)
    except BaseException as error:
        value = dict(observation_error=repr(error))
    return value | dict(episode_id=None, source_video_sha256_required=None,
                        observed_source_id=source_id, observed_run_id=run_id,
                        review_basis='同runの元分岐の読取のみ。開始未認定。',
                        general_new_game_certification=False, accounting_basis_verified=False,
                        repair_forbidden=True)


def install(stack: Any, collector: Any, rec: Any, state: dict,
            replace: Callable[..., Any], *, source_id: str) -> dict:
    """旧installerの私有moduleと同じ内側stackを所有する。"""
    A.require(KEY not in state, 'duplicate_baseline_observers')
    output = Path(state['output'])
    result = dict(ocr_observation_only=True, admission_enforced=False,
                  reset_observations=0, observation_errors=[], forbidden_repairs=0, closed=False)
    state[KEY] = result
    def record_only(owner_stack: Any, owner: Any, name: str, value: Any) -> None:
        if owner is collector and name == '_should_emit':
            return
        replace(owner_stack, owner, name, value)
    observer = A.install(stack, collector, state, record_only, source_id=source_id)
    epoch = stack.enter_context(loaded(EPOCH, EPOCH_SHA))
    original = epoch._evidence
    epoch._evidence = lambda *args: captured(original, source_id, observer.run_id, *args)
    epoch._eligible = lambda evidence: False
    def forbidden(*args: Any, **kwargs: Any) -> None:
        result['forbidden_repairs'] += 1
        raise RuntimeError('g3_baseline_repair_forbidden')
    epoch._repair = forbidden
    def observation(owner: Any, evidence: dict, reason: str) -> None:
        result['reset_observations'] += 1
        error = evidence.get('observation_error')
        if error is not None:
            result['observation_errors'].append(error)
        rec.emit(dict(kind='g3_reset_observation', evidence=evidence, repair_applied=False))
        if error is not None:
            raise RuntimeError('g3_baseline_reset_observation_failed:' + error)
    epoch.EpochRepair.observation = observation
    def closed() -> None:
        result['closed'] = True
        save(output / 'G3_BASELINE_OBSERVERS_STATUS.json', result | dict(quality_gate_clear=False))
    stack.callback(closed)
    repair = epoch.install(stack, collector, rec)
    A.require(repair.wrap.__func__.__globals__ is vars(epoch), 'private_epoch_globals')
    return result
