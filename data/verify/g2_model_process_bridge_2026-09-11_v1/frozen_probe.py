"""原frozenパッケージと別モデル子の共存を検査。captureは保存票由来の人工対照。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import client as C
import pure_contract as P

ROOT=Path(__file__).resolve().parent
PUB=ROOT.parent/'g2_belief_live_publication_2026-09-11_v1'


def load(name: str,path: Path) -> Any:
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def saved_row(frame: int) -> Any:
    path=ROOT.parent/'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v45/provisional_context.jsonl'
    with path.open() as stream:
        for line in stream:
            value=json.loads(line)
            if value['frame_idx']==frame: return value
    raise AssertionError('frozen_probe_saved_row_missing')


def main() -> None:
    fixture=load('_model_process_frozen_fixture',
        C.PROJECT/'tests/test_diagnose_video38_next_enqueue_live_shadow_v1.py')
    generator=fixture.frozen.__wrapped__()
    collector=next(generator)
    update=collector.RecognitionPipeline.update
    try:
        sys.path.insert(0,str(ROOT.parent/'g2_probabilistic_scope_candidate_2026-09-11_v1'))
        import serialization as S
        reference=json.loads((PUB/'TRAINED_SIDECAR_v1.json').read_bytes())
        row=saved_row(reference['frame'])
        value=N(values=tuple(S.decode(s) for s in reference['states']),frame=reference['frame'],
            inputs=P._inputs(row),digest=P.digest(row),tokens=tuple(reference['journal_tokens']),
            observed=tuple(S.B.Board.from_dict({'grid':row['sides'][side]['before_hold']['confirmed']['grid']})
                for side in ('1P','2P')))
        original={k:v for k,v in sys.modules.items() if k=='src' or k.startswith('src.')}
        source_path=list(sys.modules['src'].__path__)
        receipt=C.run(lambda:value,C.Configuration(S,S.B.grid,P.RowInputs),
            ROOT/'FROZEN_MODEL_PACKET_v1.json',sample_count=32,seed=17)
        assert collector.RecognitionPipeline.update is update
        current={k:v for k,v in sys.modules.items() if k=='src' or k.startswith('src.')}
        assert original.keys()==current.keys() and all(current[k] is v for k,v in original.items())
        assert list(sys.modules['src'].__path__)==source_path
        assert not any(k.startswith('src.advantage_m') for k in current)
    finally: generator.close()
    packet=dict(receipt=receipt,src_references_unchanged=True,src_path_unchanged=True,
        pipeline_update_unchanged=True,pipeline_updates=0,artificial_capture=True,
        live_registry_authorized=False,quality_gate_clear=False)
    with (ROOT/'FROZEN_MODEL_RESULT_v1.json').open('x') as stream: json.dump(packet,stream,indent=2)
    print(json.dumps(packet))


if __name__=='__main__': main()
