"""変更や救済を行わず、停止前の私有証拠とnative実値を写す。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

SIDE = '1P'


def native(pipe: Any) -> dict[str, Any]:
    return {side: dict(queue_id=id(getattr(pipe, '_pending_tsumo_'+side.lower())),
        queue=list(getattr(pipe, '_pending_tsumo_'+side.lower())),
        refs=[id(x) for x in getattr(pipe, '_pending_tsumo_'+side.lower())],
        counter=dict(getattr(pipe, '_tsumo_count_'+side.lower())),
        first_move=getattr(pipe, '_first_move_sec_'+side.lower())) for side in ('1P','2P')}


def scope(factory: Any, pipe: Any) -> tuple[Any, ...]:
    journal = factory.provider.journal
    runtime = journal.controller.instances.get(id(pipe))
    epoch = None if runtime is None else runtime.histories[SIDE].epoch
    generation = journal.tracker.generation(SIDE)
    return (journal.source_id, journal.run_id, epoch, id(pipe), id(pipe._sm_1p),
        generation.reset_epoch, SIDE)


def same(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return type(left) is tuple and type(right) is tuple and len(left) == len(right) and all(
        type(a) is type(b) and a == b for a,b in zip(left,right))


def snapshot(binding: Any, factory: Any, pipe: Any) -> dict[str, Any]:
    return dict(binding_id=id(binding), owner_id=id(binding.owner), scope=binding.scope,
        actual_scope=scope(factory,pipe), owner=asdict(binding.owner.state),
        grid=binding.grid, current=binding.current, next_token=binding.next_token,
        next_started=binding.next_started, consumed_tokens=sorted(binding.consumed_tokens),
        phase=binding.phase, native=native(pipe))
