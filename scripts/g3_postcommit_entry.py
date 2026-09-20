"""検収済みG3入口を再用し、公開後履歴と終了readerを同じ所有scopeに接続。"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from scripts import g3_context_entry as E
from scripts import g3_postcommit_binding as B
from scripts import g3_postcommit_rows as R
from scripts import g3_postcommit_finish as F

G = E.G
SOURCE = G.ROOT / 'data/verify/g2_history_publication_consumer_2026-09-09_v2/adapter.py'


def install(stack: Any, adapter: Any) -> None:
    """内側の元close登録後に設置し、終了検証より後までhookを保持する。"""
    whole, replace = adapter.A.A.A.V4.A, adapter.A.A.A.V4.replace_owned
    original, fired = whole.W.Bridge, []
    def bridge(collector: Any, state: dict, inner: Any, *args: Any, **kwargs: Any) -> Any:
        G.require(not fired, 'postcommit_duplicate_bridge')
        value = original(collector, state, inner, *args, **kwargs)
        consumer = state['postcommit_publication_consumer']
        module = sys.modules[type(consumer).__module__]
        rows = B.bind(stack, consumer, module, replace, source=SOURCE,
                      source_sha=B.SOURCE_SHA, verify_root=G.VERIFY)
        receipt = F.install(stack, rows, replace)
        fired.append(True)
        G.save(state['output'] / 'G3_POSTCOMMIT_START.json', receipt | dict(
            sidecar=str(rows.path), source_sha256=B.SOURCE_SHA, actual_bridge=True))
        return value
    replace(stack, whole.W, 'Bridge', bridge)


def main() -> int:
    """実走する追加codeを固定し、元例外・保存・解放の入口を維持する。"""
    plan = G.read(G.arguments().plan)
    for module in (sys.modules[__name__], B, R, F):
        G.require(str(Path(module.__file__).resolve()) in plan['entry_pins'], 'postcommit_entry_pin')
    previous = G.protect_runtime
    def protect(stack: Any, adapter: Any, main: Any, fixed: dict) -> None:
        previous(stack, adapter, main, fixed)
        install(stack, adapter)
    G.protect_runtime = protect
    try:
        return E.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())
