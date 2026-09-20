"""既存G3入口のcontext生成時だけ、保存済み履歴の保持を削減する。"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
from scripts import g3_a40_reset_entry as E
from scripts import g3_context_rows as C

G = E.G
SOURCE = G.ROOT / 'data/verify/g2_provisional_context_capture_2026-09-09_v1/observer.py'
SOURCE_SHA = '1248b3ba0328bb03e2d1039e5f0f293424770038ac2a311fe70a2347cd40cd71'


def bind_context(stack: Any, module: Any, replace: Any) -> None:
    """認証済み実moduleの生成/検査を同じ所有scopeで復元する。"""
    G.require(Path(module.__file__).resolve() == SOURCE and C.digest(SOURCE) == SOURCE_SHA,
              'context_retention_source')
    original = module.Recorder.__init__
    G.require(Path(original.__code__.co_filename).resolve() == SOURCE, 'context_retention_duplicate')
    records: list[Any] = []
    def closed() -> None:
        for rec in records:
            G.save(rec.output / 'G3_CONTEXT_RETENTION_END.json', dict(
                count=len(rec.rows), saved_bytes=rec.rows.offset, stream_closed=rec.stream.closed,
                retained_rows=int(rec.rows.latest is not None), source_sha256=SOURCE_SHA,
                constructor_restored=module.Recorder.__init__ is original, quality_gate_clear=False))
    stack.callback(closed)
    def initialize(rec: Any, *args: Any, **kwargs: Any) -> None:
        G.require(not records, 'context_retention_multiple_recorders')
        original(rec, *args, **kwargs)
        try:
            G.require(type(rec.rows) is list and not rec.rows and not rec.stream.closed,
                      'context_retention_install_late')
            G.require(rec.output.resolve().is_relative_to(G.VERIFY.resolve()), 'context_retention_storage')
            rec.rows = C.ContextRows(rec.output / module.SIDECAR)
            G.save(rec.output / 'G3_CONTEXT_RETENTION_START.json', dict(source_sha256=SOURCE_SHA,
                saved_path=str(rec.rows.path), max_row_bytes=C.MAX_ROW_BYTES,
                quality_gate_clear=False))
            records.append(rec)
        except BaseException:
            rec.stream.close()
            raise
    def verify(output: Path, *, expected_frames: list[int] | None = None) -> dict:
        return C.verify(module, output, expected_frames=expected_frames)
    replace(stack, module.Recorder, '__init__', initialize)
    replace(stack, module, 'sha', C.digest)
    replace(stack, module, 'verify', verify)


def install(stack: Any, adapter: Any) -> None:
    """元window.bindの返す実contextを、Sink生成前に一回だけ接続する。"""
    window, replace = adapter.OC.ORIGINAL, adapter.A.A.A.V4.replace_owned
    original, fired = window.bind, []
    def bind(owner_stack: Any, latest: Any, owner_replace: Any) -> dict:
        G.require(owner_stack is stack and owner_replace is replace and not fired,
                  'context_retention_bind_owner')
        values = original(owner_stack, latest, owner_replace)
        bind_context(stack, values['context'], replace)
        fired.append(True)
        return values
    replace(stack, window, 'bind', bind)


def main() -> int:
    """元A40入口の保存/終了/解放を再用し、追加codeの固定を必須にする。"""
    plan = G.read(G.arguments().plan)
    for module in (sys.modules[__name__], C):
        G.require(str(Path(module.__file__).resolve()) in plan['entry_pins'], 'context_retention_pin')
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
