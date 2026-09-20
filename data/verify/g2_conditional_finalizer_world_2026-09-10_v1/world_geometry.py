"""原列挙器で条件世界を再構成し、保存した履歴と同じ解であることを検査する。"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import json
import sys
from types import SimpleNamespace as N
from typing import Any, Iterator
import world_libs as L
import world_pb as B

STAGE1 = L.VERIFY / 'g2_conditional_finalizer_compatibility_2026-09-10_v1'
ORIGIN = L.VERIFY / 'g2_conditional_firing_origin_policy_2026-09-10_v1'
SETTLE = L.VERIFY / 'g2_conditional_firing_settlement_2026-09-10_v1'
SNAPSHOT = L.ROOT.parents[2] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
PREFIX = 'hidden_two_hand_prefix_history/v1'
TAIL = 'hidden_single_tail_history/v1'
CONTINUE = 'hidden_conditional_continuation_history/v1'
FIRING = 'conditional_firing_placement/v1'
NEXT = 'conditional_after_settlement_next_placement/v1'
KINDS = frozenset((PREFIX, TAIL, CONTINUE, FIRING, NEXT))


@contextmanager
def libraries() -> Iterator[Any]:
    lib, previous = L.libraries(), list(sys.path)
    try:
        sys.path[:0] = [str(STAGE1), str(ORIGIN), str(SETTLE), str(SNAPSHOT)]
        fixed = importlib.import_module('conditional_finalizer_fixed')
        dto = importlib.import_module('conditional_finalizer_types')
        origin = importlib.import_module('origin_evidence')
        settle = importlib.import_module('conditional_settlement')
        for module, path in ((fixed, STAGE1), (dto, STAGE1), (origin, ORIGIN), (settle, SETTLE)):
            B.require(L.Path(module.__file__).resolve().parent == path, 'original_geometry_module')
        from src import puyo_core_bridge as core
        backend = origin.F.backend()
        with fixed.session() as m:
            p = N(S=m.S, encoded=m.E.encoded, digest=m.E.digest)
            yield N(lib=lib, core=core, backend=backend, origin=origin, settle=settle,
                dto=dto, m=m, p=p)
    finally:
        sys.path[:] = previous


def capture(lib: Any, source: Any, histories: Any, frame: int, side: str) -> None:
    row = histories[frame, side]
    B.require(source['captured_frame'] == frame, 'geometry_capture_frame')
    B.equal(lib, source['capture'], row['raw_capture'], 'geometry_actual_raw_capture')
    for key in ('raw', 'filtered'):
        B.equal(lib, source['raw_grid'], row['raw_capture'][key]['grid'], 'geometry_actual_raw')


def prefix(ctx: Any, proof: Any) -> None:
    lib, P = ctx.lib, ctx.lib.P
    saved = proof['inferred_path']
    supports = P.infer(ctx.core, proof['before_grid'], saved['pairs'][0], saved['pairs'][1], proof['observed_raw'])
    B.require(len(supports) == 1, 'prefix_unique_support')
    witness = L.load('_world_original_prefix_witness', L.VERIFY /
        'g2_hidden_two_hand_candidate_2026-09-10_v1/prefix_witness.py')
    B.require(witness.nonfiring(ctx.core, supports[0]), 'prefix_original_nonfiring')
    B.equal(lib, asdict(supports[0]), saved, 'prefix_original_support')
    B.equal(lib, proof['pair'], saved['pairs'][0], 'prefix_pair')
    B.equal(lib, proof['grid'], saved['prefix'], 'prefix_world')
    B.require(proof['intermediate_grid_is_observed'] is False, 'prefix_not_observed')


def placement(ctx: Any, proof: Any) -> None:
    lib, P = ctx.lib, ctx.lib.P
    before, final = P.grid(proof['before_grid']), P.grid(proof['grid'])
    raw, pair = P.grid(proof['observed_raw'], hidden_unknown=True), P.pair(proof['pair'])
    choices = tuple(g for g in P.options(ctx.core, before, pair) if P.compatible(raw, g))
    B.require(choices == (final,), 'placement_unique_support')
    if proof['kind'] != FIRING:
        # 原continuation producerと同じ既定引数。発火側の採用GHOST検査とは別。
        B.require(ctx.core.simulate_chain(P.board(ctx.core, final)).chain_count == 0, 'placement_nonfiring')
    B.equal(lib, final, proof['inferred_final'], 'placement_final')
    B.require(proof['inferred_grid_is_observed'] is False, 'placement_not_observed')


def prior(ctx: Any, proof: Any, checked: Any) -> None:
    encoded = proof['previous_conditional_certificate']
    matches = [value for value in checked.values() if value['evidence_json'] == encoded]
    B.require(len(matches) == 1, 'previous_current_saved_once')
    value, lib = matches[0], ctx.lib
    B.require(value['frame'] < proof['occurred'][0], 'previous_current_precedes')
    B.equal(lib, value['inferred_grid'], proof['before_grid'], 'previous_current_world')
    previous = json.loads(encoded)
    source = (previous['conditional_origin']['placement']['prefix_source']
        if proof['kind'] == NEXT else previous['prefix_source'])
    B.equal(lib, source, proof['prefix_source'], 'previous_prefix_source')


def histories(ctx: Any, rows: Any, checked: Any) -> dict[Any, Any]:
    lib, prepared, prefixes = ctx.lib, {}, {}
    indexed = {(row['scope']['frame_idx'], row['scope']['side']): row for row in rows}
    for row in rows:
        proof = row['prepared']
        if proof is None or proof['kind'] not in KINDS:
            continue
        frame, side = row['scope']['frame_idx'], row['scope']['side']
        B.require(proof['available_frame'] == frame and proof['physical_certified'] is False
            and proof['current_permission'] is False, 'geometry_history_permissions')
        capture(lib, proof['raw_capture'], indexed, frame, side)
        if proof['kind'] == PREFIX:
            prefix(ctx, proof)
            prefixes[frame, side] = proof
        else:
            placement(ctx, proof)
            if proof['kind'] == TAIL:
                first = prefixes[proof['prefix_consumed_frame'], side]
                B.equal(lib, first['inferred_path']['prefix'], proof['before_grid'], 'tail_prefix')
                B.equal(lib, first['inferred_path']['final'], proof['grid'], 'tail_final')
                B.equal(lib, first['inferred_path']['pairs'][1], proof['pair'], 'tail_pair')
                B.equal(lib, first['directional_commit'], proof['prefix_source'], 'tail_source')
            else:
                prior(ctx, proof, checked)
        prepared[frame, side] = proof
    return prepared


def origins(ctx: Any, rows: Any, prepared: Any, histories: Any) -> dict[str, Any]:
    found, lib = {}, ctx.lib
    for row in rows:
        if row.get('stage') != 'conditional_origin_registered':
            continue
        proof = row['proof']
        value = ctx.dto.origin(ctx.m, row['origin'])
        B.require(value.origin_id not in found, 'duplicate_conditional_origin')
        source = prepared[row['frame'], value.scope.side]
        B.require(source['kind'] == FIRING, 'origin_firing_history')
        B.equal(lib, source, proof['placement'], 'origin_actual_placement')
        B.equal(lib, row['origin'], proof['evidence'], 'origin_saved_DTO')
        ctx.origin.geometry(ctx.p, source, value, proof['backend'])
        ctx.origin.observations(ctx.p, source, proof['previous_anchor'])
        for observation in source['observations']:
            capture(lib, observation['raw_proof'], histories, observation['frame'], value.scope.side)
        found[value.origin_id] = (value, proof)
    return found


def settlements(ctx: Any, rows: Any, origins: Any, histories: Any) -> dict[str, Any]:
    found, lib = {}, ctx.lib
    for row in rows:
        if row.get('stage') != 'conditional_private_settled':
            continue
        proof = row['proof']
        key = proof['evidence']['origin']['origin_id']
        B.require(key not in found, 'duplicate_conditional_settlement')
        value, origin = origins[key]
        B.equal(lib, proof['origin_proof'], origin, 'settlement_actual_origin')
        observations = proof['observations']
        B.require(len(observations) == 2, 'settlement_two')
        for observation in observations:
            ctx.settle.observation(ctx.p, ctx.m.E, observation, value, origin)
            capture(lib, observation['raw_capture'], histories, observation['observed_at']['frame'], value.scope.side)
        first, last = observations
        B.require(first['observed_at']['frame'] + 2 == last['observed_at']['frame'] == row['frame']
            and first['capture_ref'] != last['capture_ref'], 'settlement_adjacent')
        B.equal(lib, first['raw_grid'], last['raw_grid'], 'settlement_same_raw')
        found[key] = proof
    return found
