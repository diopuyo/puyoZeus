"""原run_liveと同じcollector冷読込後に私有予測依存を検査。動画/モデル生成なし。"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import probe_cold_prediction_dependency as P
from types import SimpleNamespace as N
from typing import Any


def exercise(stack: Any, loader: Any) -> dict:
    from src.board import Board
    alias = '_g2_probability_scope_frozen_physics'
    if alias not in sys.modules:
        wrapper = '_cold_private_prediction_physics'
        loader.load(wrapper, P.ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1/frozen_physics.py')
        owned = {name: sys.modules[name] for name in (alias, wrapper)}
        def release() -> None:
            for name, module in owned.items():
                if sys.modules.get(name) is module: sys.modules.pop(name)
        stack.callback(release)
    simulator = sys.modules[alias].ChainSimulator
    physics = P.B.physics(N(B=N(B=N(Board=Board, ChainSimulator=simulator))))
    result = P.exercise(stack, loader, physics=physics)
    result.update(bind_physics_guard_exercised=True, private_simulator_not_pipeline=True)
    return result


def main() -> None:
    sys.path.insert(0, str(P.RUNTIME))
    import owned_adapter as A
    initial, paths, result = Path.cwd(), list(sys.path), {}
    try:
        with ExitStack() as stack:
            selected = A.configured(stack)
            adapter = A.configured.__globals__['A'].A.A.V4
            owner = sys.modules[adapter.OWNED_ALIAS]
            with selected.__globals__['S'].configured() as env:
                base = env['runtime'].M.base
                os.chdir(base.SNAPSHOT)
                collector = base.load_collector()  # 原run_liveの冷読込入口そのもの。
                with ExitStack() as inner:
                    result = exercise(inner, owner.bootstrap())
                assert not any(alias in sys.modules for alias in P.B.ALIASES)
                result['collector_source'] = collector.__file__
        result.update(configured_context_exited=True, private_aliases_released=True,
                      model_instantiated=False, pipeline_created=False)
    except BaseException as error:
        result = dict(error=repr(error), quality_gate_clear=False, video_updates=0)
        raise
    finally:
        os.chdir(initial)
        sys.path[:] = paths
        with (P.ROOT / 'COLD_PREDICTION_DEPENDENCY_v3.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
