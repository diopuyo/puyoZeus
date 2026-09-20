"""新私有配置の幾何/原履歴/PBを結合。既存の検査・通常経路は原委譲。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'g2_conditional_finalizer_world_2026-09-10_v1'
PRIVATE = 'hidden_private_suffix_history/v1'
SOURCE_KEY = 'private_suffix_committed_source'


def libraries() -> Any:
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(OLD))
        import world_verify as W
        assert Path(W.__file__).resolve() == OLD/'world_verify.py'
        return W
    finally:
        sys.path[:] = previous


def histories(W: Any, ctx: Any, rows: Any, checked: Any) -> Any:
    result = W.G.histories(ctx, rows, checked)
    indexed = {(r['scope']['frame_idx'], r['scope']['side']): r for r in rows}
    for key, row in indexed.items():
        proof = row['prepared']
        if proof is None or proof['kind'] != PRIVATE:
            continue
        frame, side = key
        W.B.require(proof['available_frame'] == frame and proof['physical_certified'] is False
            and proof['current_permission'] is False, 'private_geometry_permissions')
        W.G.capture(ctx.lib, proof['raw_capture'], indexed, frame, side)
        W.G.placement(ctx, proof)
        prior = result[proof['previous_private_frame'], side]
        W.B.require(prior['kind'] == W.G.TAIL, 'private_basis_kind')
        W.B.require(ctx.m.E.digest(prior) == proof['previous_private_placement'], 'private_basis_digest')
        W.B.equal(ctx.lib, prior['grid'], proof['before_grid'], 'private_basis_grid')
        W.B.equal(ctx.lib, prior['prefix_source'], proof['prefix_source'], 'private_basis_prefix')
        W.B.require(prior['available_frame'] < proof['occurred'][0] <= frame, 'private_basis_clock')
        result[key] = proof
    return {key: result[key] for key in indexed if key in result}


def hidden_histories(W: Any, lib: Any, rows: Any, prepared: Any, history: Any) -> None:
    private = {key: proof for key, proof in prepared.items() if proof['kind'] == PRIVATE}
    old_rows, joined = [], set()
    for row in rows:
        if row['kind'] != PRIVATE:
            old_rows.append(row)
            continue
        key = row['frame'], row['state']['scope']['side']
        W.B.require(key in private and key not in joined, 'private_hidden_join')
        proof, actual = private[key], history[key]
        W.B.equal(lib, row['source'], proof, 'private_hidden_source')
        W.B.equal(lib, row['state'], actual['decision']['history_state'], 'private_hidden_state')
        W.B.equal(lib, row['old_current'], actual['returned']['grid'], 'private_hidden_original')
        W.B.equal(lib, row['inferred_final'], proof['grid'], 'private_hidden_world')
        W.B.require(row['current_permission'] is False and row['physical_certified'] is False
            and row['native_counter_unchanged'] is True, 'private_hidden_permissions')
        joined.add(key)
    W.B.require(joined == set(private), 'private_hidden_missing')
    W.hidden_histories(lib, old_rows, {k: p for k, p in prepared.items() if k not in private}, history)


def current_sources(W: Any, lib: Any, currents: Any, prepared: Any, origins: Any, settled: Any) -> None:
    ordinary = {}
    for key, value in currents.items():
        frame, side = key
        proof = json.loads(value['evidence_json'])
        choices = [(k, p) for k, p in prepared.items() if k[1] == side and k[0] <= frame]
        W.B.require(bool(choices), 'private_current_without_history')
        _, source = max(choices, key=lambda item: item[0][0])
        if source['kind'] != PRIVATE:
            W.B.require(SOURCE_KEY not in proof, 'private_stale_source')
            ordinary[key] = value
            continue
        W.B.require(proof['kind'] == 'conditional_hidden_current/v1', 'private_current_kind')
        W.B.equal(lib, proof[SOURCE_KEY], source, 'private_current_source')
        W.B.equal(lib, value['inferred_grid'], source['grid'], 'private_current_grid')
        W.B.equal(lib, proof['prefix_source'], source['prefix_source'], 'private_current_prefix')
        first = prepared[proof['prefix_consumed_frame'], side]
        tail = prepared[proof['tail_consumed_frame'], side]
        W.B.require(first['kind'] == W.G.PREFIX and tail['kind'] == W.G.TAIL, 'private_current_initial')
        W.B.equal(lib, first['directional_commit'], proof['prefix_source'], 'private_current_direction')
        W.B.require(source['previous_private_frame'] == tail['available_frame'], 'private_current_tail')
    W.current_sources(lib, ordinary, prepared, origins, settled)


def verify_world(**kwargs: Any) -> Any:
    W = libraries()
    geometry = N(**(vars(W.G) | dict(histories=lambda *args: histories(W, *args))))
    env = dict(vars(W), G=geometry, hidden_histories=lambda *args: hidden_histories(W, *args),
        current_sources=lambda *args: current_sources(W, *args))
    delegated = FunctionType(W.verify_world.__code__, env)
    delegated.__kwdefaults__ = W.verify_world.__kwdefaults__
    result = delegated(**kwargs)
    found = any(r['prepared'] and r['prepared']['kind'] == PRIVATE for r in kwargs['history_rows'])
    return result | dict(private_suffix_geometry_verified=found, runtime_finalization_allowed=False)
