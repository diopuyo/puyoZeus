"""修復の直接呼出でなく、実attachが所有するfactoryの生成結果を調べる。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'g2_whole_repairs_runtime_2026-09-12_v2'
sys.path.insert(0, str(OLD))
import repair_adapter as R
spec = importlib.util.spec_from_file_location('_owned_repro_cold', OLD / 'probe_cold.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def main() -> None:
    with ExitStack() as stack:
        selected = R.configured(stack)
        C.collector()
        factory = sys.modules['_g2_live_probability_owned_loader']
        # 新版selected()を先に呼ばない。実buildが参照するglobalsをそのまま使う。
        assert factory.build.__globals__ is vars(factory)
        context = factory.build(object, end_frame=36298)
        origins = [cls.__dict__['perform'].__globals__['__file__'] for cls in context.__mro__
                   if 'perform' in cls.__dict__]
        expected = str(R.SIDE / 'side_actual_connection.py')
        old = str(R.VERIFY / 'g2_hidden_basis_initialization_2026-09-11_v1/side_actual_connection.py')
        value = dict(generic_and_owned_same_module=factory is R.A.L,
                     generic_and_owned_same_dependencies=factory.dependencies is R.A.L.dependencies,
                     build_globals_owned=True, perform_origins=origins,
                     expected_selected=expected in origins, old_selected=old in origins,
                     updates=0, models=0, quality_gate_clear=False)
    (ROOT / 'BEFORE_v1.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)
    assert value['old_selected'] and not value['expected_selected'], 'old_counterexample_not_reproduced'


if __name__ == '__main__':
    main()
