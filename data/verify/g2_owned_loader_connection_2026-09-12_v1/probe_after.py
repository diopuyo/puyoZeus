"""実owned生成で新sideを選び、旧helperaliasの事前占有なしを確認する。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import FrameType
from typing import Any
import owned_adapter as R

ROOT = Path(__file__).resolve().parent


def collector() -> None:
    sys.path.insert(0, str(R.A.L.SNAPSHOT))
    spec = importlib.util.spec_from_file_location('_owned_after_collector',
        R.A.L.SNAPSHOT / 'scripts/collect_boards_lean.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def main() -> None:
    calls: list[str] = []
    def profile(frame: FrameType, event: str, arg: Any) -> None:
        if event == 'call' and frame.f_code.co_name == 'build' and frame.f_globals.get('__name__') == R.OWNED_ALIAS:
            calls.append(frame.f_code.co_filename)
    owner = R.A.A.load(R.OWNED_ALIAS, R.PRIOR / 'loader.py')
    originals = (owner.dependencies, owner.build, R.A.L.dependencies)
    sys.setprofile(profile)
    try:
        with ExitStack() as stack:
            R.configured(stack)
            assert not calls, 'configured_built_before_swap'
            collector()
            assert R.A.L.dependencies is originals[2], 'generic_loader_changed'
            assert '_g2_actual_side_lease_selected' not in sys.modules, 'lease_preprimed'
            context = owner.build(object, end_frame=36298)
            assert len(calls) == 1, 'owned_original_build_count'
            methods = [cls.__dict__['perform'] for cls in context.__mro__ if 'perform' in cls.__dict__]
            origins = [m.__globals__['__file__'] for m in methods]
            assert str(R.SIDE / 'side_actual_connection.py') in origins, 'new_side_missing'
            assert str(R.VERIFY / 'g2_hidden_basis_initialization_2026-09-11_v1/side_actual_connection.py') not in origins
            assert '_g2_live_probability_side_connection' in sys.modules, 'old_import_expected'
            assert '_g2_actual_side_lease_selected' not in sys.modules, 'old_import_selected_lease'
            side = sys.modules[R.SIDE_ALIAS]
            selected = side.selected()
            helper_path = Path(selected.parts().helper.V1.__file__).resolve()
            assert helper_path == R.SIDE / 'side_reset_op.py'
        assert originals == (owner.dependencies, owner.build, R.A.L.dependencies), 'owned_restore'
        assert R.SIDE_ALIAS not in sys.modules, 'owned_alias_restore'
    finally:
        sys.setprofile(None)
    result = dict(configured_builds=0, owned_builds=len(calls), new_side=True,
                  new_helper=str(helper_path), generic_unchanged=True, restored=True,
                  old_import_did_not_select_lease=True, actual_constructor=False,
                  updates=0, models=0, quality_gate_clear=False)
    (ROOT / 'AFTER_v1.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
