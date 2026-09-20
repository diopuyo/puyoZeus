"""A40の実loadへ観測/採録reset保留を限定接続する。"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
from scripts import g3_palette_reset_entry as R
from scripts import g3_normal_reset_scope as N

G = R.G
PUBLICATION = G.ROOT / 'data/verify/g2_belief_live_publication_2026-09-11_v1'
OBSERVATION = PUBLICATION / 'second_observation.py'
TARGETS = {'_g2_pub_runtime_second_observation': OBSERVATION, '_g2_evaluation_flags': G.ROOT / 'data/verify/g2_m1_completion_candidate_2026-09-12_v1/evaluation_flags.py'}
FLAGS = G.ROOT / 'data/verify/g2_m1_completion_candidate_2026-09-12_v1/evaluation_flags.py'


def install(stack: Any, adapter: Any, output: Path) -> None:
    """selectorの初期ownerを維持し、実際に生成された私有関数だけ変換する。"""
    version = adapter.A.A.A.V4
    owner, replace = sys.modules[version.OWNED_ALIAS], version.replace_owned
    original = version.A.D.session_creator
    fired: set[Path] = set()
    def create(load: Any, frames: tuple[int, ...]) -> Any:
        bootstrap = owner.bootstrap()
        G.require(bootstrap.load is load, 'a40_reset_initial_loader')
        def selected(alias: str, path: Any, injection: Any = None) -> Any:
            module = load(alias, path, injection)
            target = Path(path).resolve()
            if alias in TARGETS and target == TARGETS[alias]:
                G.require(target not in fired, 'a40_reset_duplicate_load')
                receipt = (N.R.install if target == OBSERVATION else N.install_flags)(stack, module, replace)
                fired.add(target)
                G.save(output / ('G3_A40_RESET_' + target.stem + '.json'),
                    receipt | dict(private_module=module.__name__, actual_load=True))
            return module
        replace(stack, bootstrap, 'load', selected)
        return original(selected, frames)
    replace(stack, version.A.D, 'session_creator', create)


def main() -> int:
    """既存入口と所有終了を維持して、新しい二つの実codeを固定する。"""
    args = G.arguments()
    plan = G.read(args.plan)
    for module in (sys.modules[__name__], N):
        G.require(str(Path(module.__file__).resolve()) in plan['entry_pins'], 'a40_reset_entry_pin')
    previous = G.protect_runtime
    def protect(stack: Any, adapter: Any, main: Any, fixed: dict) -> None:
        previous(stack, adapter, main, fixed)
        install(stack, adapter, Path(args.output))
    G.protect_runtime = protect
    try:
        return R.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())

