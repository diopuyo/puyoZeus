"""私有条件付きfamilyの保存再計算。原動画資格・台帳・公開への認可は持たない。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any
import prefix_phase_math_v2 as P
import split_landing as S
import engine_identity as E

VERSION = 'prefix-phase-cpu-evidence/v2'
INPUT_KINDS = frozenset(('saved_observation_unqualified', 'synthetic_future'))


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def identity(engine: Any) -> dict:
    return dict(enumerator=S.identity(engine.H), engine=E.fingerprint(engine),
        math_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def encode(serializer: Any, family: P.Family, kinds: frozenset) -> dict:
    return normalized(dict(prefix=family.prefix, phase=family.phase, state=serializer.encode(family.value),
        derived_input_kinds=sorted(kinds), synthetic_derived='synthetic_future' in kinds,
        mixture_weight=None, source_qualification_connected=False, publication_permission=False))


def compute(parts: Any, initial: Any, arrivals: tuple, steps: list[dict], assumption: str) -> list:
    engine, serializer = parts.mode.C.T, parts.mode.BASE.V1.S
    prior = engine.H.uncalibrated_uniform(assumption)
    families = (P.Family(0, P.SETTLED, initial),)
    saved = []
    previous = initial.frame
    seen_fired, seen_kinds = frozenset(), frozenset()
    for step in steps:
        engine.B.require(set(step) == {'frame', 'observed', 'fired_tokens', 'input_kind'}
            and step['input_kind'] in INPUT_KINDS, 'prefix_saved_step_schema')
        engine.B.require(type(step['frame']) is int and previous < step['frame'] <= initial.deadline,
            'prefix_saved_step_clock')
        engine.B.require(type(step['fired_tokens']) is list
            and all(type(token) is str for token in step['fired_tokens'])
            and len(step['fired_tokens']) == len(set(step['fired_tokens'])), 'prefix_saved_fired_tokens')
        fired = frozenset(step['fired_tokens'])
        engine.B.require(seen_fired <= fired, 'prefix_saved_revoked_fire')
        observed = engine.B.Board.from_dict({'grid': step['observed']})
        seen_kinds |= frozenset((step['input_kind'],))
        families, lineage = advance(parts, families, arrivals, step['frame'], observed, fired, prior)
        engine.B.require(bool(families), 'prefix_saved_zero_support')
        engine.B.require(sum(len(family.value.worlds) for family in families) <= engine.B.MAX_WORLDS,
            'prefix_saved_total_support_limit')
        saved.append(dict(frame=step['frame'], lineage=lineage,
            families=[encode(serializer, family, seen_kinds) for family in families]))
        previous = step['frame']
        seen_fired = fired
    return saved


def make(parts: Any, initial: Any, arrivals: tuple, steps: list[dict], assumption: str) -> dict:
    engine, serializer = parts.mode.C.T, parts.mode.BASE.V1.S
    outputs = compute(parts, initial, arrivals, steps, assumption)
    return normalized(dict(version=VERSION, identity=identity(engine), initial=serializer.encode(initial),
        arrivals=[asdict(arrival) for arrival in arrivals], steps=steps, prior_assumption=assumption,
        outputs=outputs, source_qualification_connected=False, original_fifo_changed=False,
        publication_permission=False, runtime_connected=False, quality_gate_clear=False))


def replay(parts: Any, packet: dict) -> list:
    engine, serializer, ledger = parts.mode.C.T, parts.mode.BASE.V1.S, parts.mode.L
    expected_keys = {'version', 'identity', 'initial', 'arrivals', 'steps', 'prior_assumption', 'outputs',
        'source_qualification_connected', 'original_fifo_changed', 'publication_permission',
        'runtime_connected', 'quality_gate_clear'}
    engine.B.require(set(packet) == expected_keys and packet['version'] == VERSION, 'prefix_saved_schema')
    engine.B.require(all(packet[key] is False for key in ('source_qualification_connected',
        'original_fifo_changed', 'publication_permission', 'runtime_connected', 'quality_gate_clear')),
        'prefix_saved_authority')
    engine.B.require(packet['identity'] == identity(engine), 'prefix_saved_algorithm_identity')
    initial = serializer.decode(packet['initial'])
    arrivals = tuple(ledger.Arrival(tuple(arrival['scope']), arrival['token'], tuple(arrival['pair']),
        arrival['frame'], arrival['call_token']) for arrival in packet['arrivals'])
    outputs = compute(parts, initial, arrivals, packet['steps'], packet['prior_assumption'])
    engine.B.require(outputs == packet['outputs'], 'prefix_saved_replay_mismatch')
    return outputs



def advance(parts: Any, families: tuple, arrivals: tuple, frame: int,
            observed: Any, fired: frozenset, prior: Any) -> tuple:
    """入力familyごとに自然な支持消滅を記録する。別familyの確率は合算しない。"""
    result, lineage = [], []
    for index, family in enumerate(families):
        children = P.condition(parts.mode.C.T, family, arrivals, frame, observed, fired, prior)
        first = len(result)
        result.extend(children)
        lineage.append(dict(input_family_index=index, prefix=family.prefix, phase=family.phase,
            output_family_indices=list(range(first, len(result))),
            dropped_reason=None if children else 'zero_support_under_current_observation'))
    return tuple(result), lineage
