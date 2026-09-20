"""基準only専用finisherを原実loaderへ挿入。旧整数経路のfinishは保持する。"""
from __future__ import annotations
from pathlib import Path
import proof_cpu_basis as BASE
import probabilistic_finish as F
import factory_anchor as A


def main() -> int:
    continuation = BASE.H.OLD.DRIVER.P.R.R.R
    expected = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/run_continuation.py'
    assert Path(continuation.__file__).resolve() == expected
    assert Path(continuation.wrap_loader.__code__.co_filename).resolve() == expected
    continuation.wrap_loader = F.wrap(continuation.wrap_loader)
    BASE.H.Context = A.derived(BASE.H.Context)
    BASE.H.OLD.INPUT.install = BASE.install
    return BASE.H.main()


if __name__ == '__main__':
    raise SystemExit(main())
