"""原run_liveと同じcollector冷読込後に私有予測依存を検査。動画/モデル生成なし。"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import probe_cold_prediction_dependency as P


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
                    result = P.exercise(inner, owner.bootstrap())
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
        with (P.ROOT / 'COLD_PREDICTION_DEPENDENCY_v2.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
