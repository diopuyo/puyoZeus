"""合格済み58の原委譲に原配置成功観測だけを追加し、旧融合も再確認。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import completion as C

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'g2_private_suffix_fusion_2026-09-10_v1'


def loader(P: Any, *args: Any) -> Any:
    selected, kept = P.loader(*args), args[-1]
    def load(alias: str, path: Path, stack: Any) -> Any:
        module = selected(alias, path, stack)
        if alias == '_private_suffix_connection':
            original = module.install
            def installed(scope: Any, factory: Any, patch: Any, history: Any, basis: Any,
                          placement: Any, exits: Any, rows: Any) -> None:
                original(scope, factory, patch, history, basis, placement, exits, rows)
                C.install(scope, factory, kept['evidence'], basis, placement, patch)
            module.install = installed
        return module
    return load


def finish(P: Any, kept: Any, evidence: Any, calls: Any) -> None:
    P.finish(kept, evidence, calls)
    import fusion
    import samecall
    output = kept['state']['output']
    journal = fusion.read(output/'atomic_journal.jsonl')
    history = fusion.read(output/'directional_history.jsonl')
    result = C.verify(kept['factory'], evidence, journal, history, samecall.journal_step)
    import completion_checks
    P.write(output/'PRIVATE_COMPLETION_CHECKS.json', completion_checks.verify(
        kept['factory'], evidence, journal, history, samecall.journal_step))
    P.write(output/'PRIVATE_COMMIT_OBSERVER.json', dict(result=result,
        records=[asdict(v) for v in kept['factory'].controller.private_suffix_completion_rows]))
    from live_finalizer import audit
    with ExitStack() as stack:
        goals = fusion.goals(kept['runtime'], stack, history, fusion.read(output/'POSTCOMMIT_CONSUMER_ROWS.json'))
        checked = audit(goals, history, kept['factory'].controller.legal, output, factory=kept['factory'],
            parts=N(evidence=evidence, completion=C), firing_rows=[],
            conditional_rows=kept['state']['conditional_full_rows'])
        assert checked['runtime_finalization_allowed']
        P.write(output/'PRIVATE_LIVE_AUDIT_CPU.json', checked)


def main() -> int:
    with ExitStack() as stack:
        before = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), before)
        sys.path.insert(0, str(PARENT))
        import run_runtime as P
        assert Path(P.__file__).resolve() == PARENT/'run_runtime.py'
        delegated = FunctionType(P.main.__code__, dict(vars(P), ROOT=ROOT,
            loader=lambda *args: loader(P, *args), finish=lambda *args: finish(P, *args)))
        return delegated()


if __name__ == '__main__':
    raise SystemExit(main())
