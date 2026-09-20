"""既存current factory生成後に観測付きbaselineを私有接続する。"""
from __future__ import annotations
from typing import Any
import common as K

STATUS = 'BASELINE_ADOPTION_STATUS.json'


def factory_type(base: Any, adoption: Any, witness: Any) -> type:
    class Factory(base):
        def make_controller(self, provider: Any, legal: Any, *, enabled: bool = False) -> Any:
            control = super().make_controller(provider,legal,enabled=enabled)
            item = adoption.bind(witness,provider,control)
            self.baseline_adoption = item
            close = item.close
            def closed() -> None:
                close()
                K.write(provider.journal.output/STATUS,dict(closed=item.closed,error=item.error,
                    adoptions=item.adoptions,remaining_witness_slots=len(item.slots),
                    unwitnessed_appends=item.unwitnessed_appends,
                    instance_baseline_restored='baseline' not in vars(control),
                    physical_certified=False,quality_gate_clear=False,production_permission=False))
            item.close = closed
            return control
    return Factory


def configure(stack: Any, core: Any, current: Any) -> None:
    K.guards()
    witness = K.load('_probe_existing_slot_witness',K.ADOPTION/'witness.py',stack)
    adoption = K.load('_probe_existing_slot_binding',K.ADOPTION/'adoption.py',stack)
    original = current.factory_type
    def patched(assembly: Any, loaded: Any) -> type:
        return factory_type(original(assembly,loaded),adoption,witness)
    core.G.patch(stack,current,'factory_type',patched)
