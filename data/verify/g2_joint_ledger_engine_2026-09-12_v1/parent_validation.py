"""子に公開権限を与えず、要求・モデル来歴・標本集計を親で検査する。"""
from __future__ import annotations
import hashlib
import math
from typing import Any
from parent_transport import P

MANIFEST = '0b15913525c4c535f210954f24997b0377539be6e404a29785a6cb476de85399'
SEEDS = ('20260904', '20260905', '20260906')
TOLERANCE = 1e-12
KEYS = {'frame', 'context_digest', 'journal_tokens', 'event_sha256', 'canonical_sha256',
        'supported', 'integrity_valid', 'details', 'actual_video', 'live_registry_authorized',
        'production_permission', 'quality_gate_clear', 'producer_sha256', 'observation_request_sha256'}
FALSE_FIELDS = ('actual_video', 'live_registry_authorized', 'production_permission', 'quality_gate_clear')
JOINT_FALSE = ('source_producer_connected', 'integer_current_permission', 'accounting_permission', 'quality_gate_clear')


def digest(value: Any) -> str:
    return hashlib.sha256(P.encoded(value)).hexdigest()


def probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def estimate(result: dict, request: dict, arrays: dict) -> None:
    counts = [len(s['hidden_worlds']) for s in request['states']]
    used = 1 if counts == [1, 1] else request['sample_count']
    P.require(result['frame'] == request['frame'] and result['seed'] == request['seed']
              and result['world_counts'] == counts and result['used_samples'] == used, 'parent_sample_scope')
    P.require(result['provisional'] is True and all(result[k] is False for k in JOINT_FALSE), 'parent_joint_authority')
    P.require(set(arrays) == set(SEEDS) and all(type(v) is list and len(v) == used
              and all(probability(p) for p in v) for v in arrays.values()), 'parent_seed_arrays')
    mean = math.fsum(math.fsum(arrays[s]) for s in SEEDS) / (len(SEEDS) * used)
    P.require(probability(result['probability_p1']) and abs(result['probability_p1'] - mean) <= TOLERANCE, 'parent_mean')


def validated(result: dict, producer: dict, request: dict) -> None:
    P.require(type(result) is dict and set(result) == KEYS, 'parent_result_schema')
    P.require(all(result[k] is False for k in FALSE_FIELDS) and type(result['supported']) is bool
              and result['integrity_valid'] is True, 'parent_result_authority')
    P.require(type(result['frame']) is int and result['frame'] == producer['last_frame'] == request['frame'], 'parent_frame')
    P.require(result['context_digest'] == request['context_digest']
              and result['journal_tokens'] == request['tokens'], 'parent_context_tokens')
    P.require(result['producer_sha256'] == digest(producer)
              and result['observation_request_sha256'] == digest(request), 'parent_input_digests')
    for key in ('event_sha256', 'canonical_sha256'):
        P.require(type(result[key]) is str and len(result[key]) == 64
                  and all(c in '0123456789abcdef' for c in result[key]), 'parent_digest_shape')
    details = result['details']
    P.require(type(details) is dict and set(details) == {'raw', 'calibrated', 'sample_digest',
              'seed_raw', 'seed_calibrated', 'members'}, 'parent_details_schema')
    P.require(digest(details['members']) == MANIFEST, 'parent_model_manifest')
    estimate(details['raw'], request, details['seed_raw'])
    estimate(details['calibrated'], request, details['seed_calibrated'])
