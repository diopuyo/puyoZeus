"""原80prefix不変、後半25更新だけ次手なし入力へ揃える限定CPU入口。"""
from __future__ import annotations
import sys
from typing import Any
import proof_cpu_hidden as H
import basis_only_input as B

ORIGINAL_INSTALL = H.OLD.INPUT.install


def install(original: Any, reset: int, q: Any, stack: Any, pipe: Any,
            pixels: Any, types: Any, clock: Any) -> Any:
    assert H.OLD.INPUT.BASIS_ONLY, 'basis_only_entry_required'
    supplied = ORIGINAL_INSTALL(original, reset, q, stack, pipe, pixels, types, clock)
    output = H.OLD.DRIVER.P.R.Q.ROOT / sys.argv[1]
    stream = stack.enter_context((output / 'BASIS_ONLY_INPUT.jsonl').open('x', encoding='utf-8'))
    B.install(stack, pixels, types, reset, supplied['original'], stream)
    supplied.update(geometry_scenario='basis_only_no_next_hand',
                    geometry_before_transform_retained=True, original_geometry_is_effective=False)
    return supplied


if __name__ == '__main__':
    H.OLD.INPUT.install = install
    raise SystemExit(H.main())
