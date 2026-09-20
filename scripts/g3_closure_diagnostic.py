"""closure.finish の sink 複合検査を、4つの部分条件へ分解して記録する診断接続。

v10は sink_stream_unclosed で落ちたが、4条件のANDなのでどれが偽か原票に残らない。
140分の実走を1回して「どれか分からない失敗」を得るのを避けるため、
部分条件・stream別のclosed状態・errorsを保存する。

既定は診断のみで継続 (tolerate)。診断runで残りの終端欠陥をまとめて洗い出すのが目的で、
これは品質合格ではない。原因が分かった後、tolerateを切って正規の合格を取り直す。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from scripts import g3_legacy_m1_window as E

G = E.G
FUNCTION = 'finish'
TARGET = 'data/verify/g2_history_publication_probe_runtime_2026-09-10_v13/closure.py'
TARGET_SHA = '0fb4586326f0ab166f1ce7f9bbebd3ef2b1468aabd65b30d08d4ee248db20381'
OLD_LINE = ("    K.require(sink.closed and not sink.errors and handles "
            "and all(v.closed for v in handles.values()), 'sink_stream_unclosed')")
NEW_LINE = "    G3_SINK(sink, handles, state)"
REPORT = 'G3_CLOSURE_DIAGNOSTIC.json'
TOLERATE = False
SIDECAR = 'G3_POSTCOMMIT_ROWS.jsonl'
WALK_NAMES = ('stream', 'base_sink', 'rows', 'writer', 'sink', 'ledger_stream')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_sidecar(state: Any) -> list[dict]:
    """G3が足した副次streamを、G2の閉鎖契約が要求する境界で閉じる。

    stream_handles は state から到達できる開いたstreamを自動で拾う。
    G3のpostcommit sidecarは巻き戻しでしか閉じないため、closure.finish の時点で
    未閉鎖として検出されていた。閉じ忘れではなく順序の問題なので、契約を緩めず順序を直す。
    SavedRows.close は多重呼出しに耐えるので、後段の原closeはそのまま残す。
    """
    closed, visited = [], set()

    def visit(value: Any, depth: int) -> None:
        if id(value) in visited or depth > 4:
            return
        visited.add(id(value))
        stream = getattr(value, 'stream', None)
        name = str(getattr(stream, 'name', '')) if stream is not None else ''
        if name.endswith(SIDECAR) and callable(getattr(value, 'close', None)):
            before = bool(getattr(stream, 'closed', False))
            try:
                value.close()
                closed.append(dict(owner=type(value).__name__, path=name, was_closed=before,
                                   now_closed=bool(getattr(stream, 'closed', False)), error=None))
            except Exception as error:  # noqa: BLE001
                closed.append(dict(owner=type(value).__name__, path=name, was_closed=before,
                                   now_closed=bool(getattr(stream, 'closed', False)), error=repr(error)))
            return
        for attribute in WALK_NAMES:
            if hasattr(value, attribute):
                visit(getattr(value, attribute), depth + 1)

    if isinstance(state, dict):
        for value in state.values():
            visit(value, 0)
    return closed


def observer(record: list) -> Any:
    """4つの部分条件を個別に測り、原票へ残す。件数0を成功にしない。"""
    def observe(sink: Any, handles: Any, state: Any) -> None:
        sidecar = close_sidecar(state)
        closed = bool(getattr(sink, 'closed', False))
        errors = getattr(sink, 'errors', None)
        mapping = dict(handles) if handles else {}
        per_handle = {}
        for key, value in mapping.items():
            per_handle[str(key)] = bool(getattr(value, 'closed', False))
        item = dict(
            sink_closed=closed,
            sink_errors=None if errors is None else [repr(x) for x in errors] if hasattr(errors, '__iter__')
            else repr(errors),
            sink_errors_empty=not errors,
            handles_present=bool(mapping),
            handles_count=len(mapping),
            handles_all_closed=all(per_handle.values()) if per_handle else False,
            unclosed_handles=sorted(k for k, v in per_handle.items() if not v),
            per_handle=per_handle,
            sink_path=str(getattr(sink, 'path', '')),
            sidecar_closed_here=sidecar,
            sidecar_close_count=len(sidecar),
            original_require_satisfied=bool(closed and not errors and mapping
                                            and all(per_handle.values())),
            tolerated=TOLERATE, quality_gate_clear=False)
        record.append(item)
        try:
            output = Path(state['output']) if isinstance(state, dict) and 'output' in state else None
            if output is not None:
                (output / REPORT).write_text(
                    json.dumps(dict(calls=record, tolerated=TOLERATE, quality_gate_clear=False),
                               ensure_ascii=False, indent=1), encoding='utf-8')
        except OSError as error:
            item['report_error'] = repr(error)
        if not TOLERATE:
            G.require(item['original_require_satisfied'], 'sink_stream_unclosed_diagnosed')
    return observe


def rebuilt(module: Any, record: list) -> Any:
    """元finishのsourceからsink検査の1行だけを差し替え、元namespaceで再構築する。"""
    path = G.ROOT / TARGET
    G.require(digest(path) == TARGET_SHA, 'closure_diagnostic_source_sha')
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION]
    G.require(len(nodes) == 1, 'closure_diagnostic_function_count')
    lines = text.splitlines()
    block = '\n'.join(lines[nodes[0].lineno - 1:nodes[0].end_lineno])
    G.require(block.count(OLD_LINE) == 1, 'closure_diagnostic_require_line')
    namespace = dict(vars(module)) | {'G3_SINK': observer(record)}
    exec(compile(block.replace(OLD_LINE, NEW_LINE), str(path), 'exec'), namespace)
    return namespace[FUNCTION]


def install(stack: Any, replace: Any) -> dict:
    """実loadedのclosure moduleを差し替える。"""
    receipts: list[dict] = []
    record: list = []
    done: set[int] = set()

    def apply(module: Any) -> None:
        file = getattr(module, '__file__', None)
        if not file or id(module) in done or not hasattr(module, FUNCTION):
            return
        if not Path(file).resolve().as_posix().endswith(TARGET):
            return
        done.add(id(module))
        replace(stack, module, FUNCTION, rebuilt(module, record))
        receipts.append(dict(file=TARGET, module=module.__name__, actual_loaded=True))

    for module in list(sys.modules.values()):
        apply(module)
    original_spec = importlib.util.spec_from_file_location

    def spec_from_file_location(name: Any, location: Any = None, *args: Any, **kwargs: Any) -> Any:
        spec = original_spec(name, location, *args, **kwargs)
        if spec is None or getattr(spec, 'loader', None) is None or location is None:
            return spec
        if not Path(location).resolve().as_posix().endswith(TARGET):
            return spec
        previous = spec.loader.exec_module

        def exec_module(module: Any) -> None:
            previous(module)
            apply(module)
        spec.loader.exec_module = exec_module
        return spec
    replace(stack, importlib.util, 'spec_from_file_location', spec_from_file_location)
    receipt = dict(receipts=receipts, tolerate=TOLERATE, diagnostic_not_quality_pass=True,
                   quality_gate_clear=False)

    def summarize() -> dict:
        base = {key: value for key, value in receipt.items() if key != 'summarize'}
        return dict(base, replacements=len(receipts), call_count=len(record), calls=list(record),
                    unsatisfied_calls=[x for x in record if not x['original_require_satisfied']])
    receipt['summarize'] = summarize
    return receipt


def main() -> int:
    """M1接続の前段に設置する。認識codeと原G2ファイルは変更しない。"""
    args = G.arguments()
    output = Path(args.output)
    plan = G.read(args.plan)
    G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'closure_diagnostic_entry_pin')
    previous = G.protect_runtime
    state: dict = {}

    def protect(stack: Any, adapter: Any, main_module: Any, fixed: dict) -> None:
        previous(stack, adapter, main_module, fixed)
        state.update(install(stack, adapter.A.A.A.V4.replace_owned))

        def saved() -> None:
            G.save(output / 'G3_CLOSURE_DIAGNOSTIC_RECEIPT.json', state['summarize']())
        stack.callback(saved)
    G.protect_runtime = protect
    try:
        return E.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())
