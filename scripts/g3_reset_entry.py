"""G3現在出口へreset観測と所有終了を接続する独立入口。"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
from scripts import g3_current_exit_entry as C
from scripts import g3_reset_scope as R
from scripts import g3_owned_exit as X

G = C.E.G
HOOK = G.ROOT / 'data/verify/g2_repeated_firing_runtime_candidate_2026-09-10_v1/constructor_v1/hook.py'


def install(stack: Any, adapter: Any) -> None:
    """ロード済み実所有moduleと、今後生成される私有確率moduleを束縛する。"""
    replace = adapter.A.A.A.V4.replace_owned
    matches = [module for module in tuple(sys.modules.values())
               if getattr(module, '__file__', None) and Path(module.__file__).resolve() == HOOK]
    G.require(len(matches) == 1, 'reset_constructor_module_identity')
    replace(stack, matches[0], 'attach', X.attach)
    original = adapter.E.modules
    def modules(owner: Any, load: Any, *, probability: dict | None = None) -> tuple:
        result = original(owner, load, probability=probability)
        if probability is not None:
            receipt = R.install(owner, probability['observation'], replace)
            cls = probability['capture'].Capture
            replace(owner, cls, 'release', X.release_wrapper(cls.release))
            # 私有moduleの生成codeを原票に残し、GT/品質許可へ昇格させない。
            receipt['private_module'] = probability['observation'].__name__
            init = cls.__init__
            def initialized(value: Any, scope: Any, journal: Any, state: dict,
                            observation: Any, replace_owned: Any, *, hook_stack: Any = None) -> None:
                init(value, scope, journal, state, observation, replace_owned, hook_stack=hook_stack)
                G.save(state['output'] / 'G3_RESET_OBSERVATION_CODE.json', receipt)
            replace(owner, cls, '__init__', initialized)
        return result
    replace(stack, adapter.E, 'modules', modules)


def main() -> int:
    """元entryの保存/guard/終了を維持し、新接続の実codeもplanへ要求する。"""
    plan = G.read(G.arguments().plan)
    for path in (Path(__file__).resolve(), Path(R.__file__).resolve(), Path(X.__file__).resolve()):
        G.require(str(path) in plan['entry_pins'], 'reset_entry_pin')
    previous = C.E.install
    def combined(stack: Any, adapter: Any) -> None:
        previous(stack, adapter)
        install(stack, adapter)
    C.E.install = combined
    try:
        return C.main()
    finally:
        C.E.install = previous


if __name__ == '__main__':
    raise SystemExit(main())
