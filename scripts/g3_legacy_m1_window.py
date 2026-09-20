"""legacy M1互換判定の終端要求を、全呼出を記録したうえで既存の unsupported 経路へ流す。

W-G3-M1-OBSERVATION-INCOMPLETE の限定接続。原G2ファイルは変更しない。
v8は observation_incomplete で落ちたが、保存された最終stateでは同条件が成立していた。
どの呼出で落ちたかを確定するため、呼出ごとに state/END/pending を原票へ残す。
条件を満たさない呼出は、原コードが既に持つ incomplete_coverage の unsupported 経路へ流す。
これは回避策であり根治ではない。早すぎる close が原因なら記録から判明する。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from scripts import g3_start_qualification_effect as E

G = E.G
FUNCTION = 'finished'
TARGET = 'data/verify/g2_legacy_m1_compatibility_2026-09-14_v1/compatibility.py'
OLD_LINE = "    require(state.last == schedule.END and state.pending is None, 'observation_incomplete')"
NEW_LINE = "    G3_OBSERVE(schedule, state, output)"
LOG = 'G3_M1_FINISHED_CALLS.jsonl'
TARGET_SHA = '91f417b7611b2b147bad7c97a0f12c2489ae9026bcf976b95e66c34f8571a950'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observer(record: list) -> Any:
    """原requireの位置で呼出事実を残す。満たさない呼出も止めず、原unsupported経路へ渡す。"""
    def observe(schedule: Any, state: Any, output: Any) -> None:
        end = getattr(schedule, 'END', None)
        accepted = tuple(getattr(state, 'accepted', ()) or ())
        item = dict(last=getattr(state, 'last', None), end=end,
                    pending=getattr(state, 'pending', None), accepted=list(accepted),
                    earliest=list(getattr(schedule, 'EARLIEST', ()) or ()),
                    original_require_satisfied=bool(
                        getattr(state, 'last', None) == end and getattr(state, 'pending', None) is None),
                    output=str(output), quality_gate_clear=False)
        record.append(item)
        try:
            path = Path(output) / LOG
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        except OSError as error:
            item['log_error'] = repr(error)
    return observe


def rebuilt(module: Any, record: list) -> Any:
    """元finishedのsourceから終端requireの1行だけを差し替え、元namespaceで再構築する。"""
    path = G.ROOT / TARGET
    G.require(digest(path) == TARGET_SHA, 'm1_window_source_sha')
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION]
    G.require(len(nodes) == 1, 'm1_window_function_count')
    lines = text.splitlines()
    block = '\n'.join(lines[nodes[0].lineno - 1:nodes[0].end_lineno])
    G.require(block.count(OLD_LINE) == 1, 'm1_window_require_line')
    namespace = dict(vars(module)) | {'G3_OBSERVE': observer(record)}
    exec(compile(block.replace(OLD_LINE, NEW_LINE), str(path), 'exec'), namespace)
    return namespace[FUNCTION]


def install(stack: Any, replace: Any) -> dict:
    """実loadedのcompatibility moduleを差し替える。件数は受領票へ残す。"""
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
    receipt = dict(receipts=receipts, workaround_not_root_fix=True, quality_gate_clear=False)

    def summarize() -> dict:
        # 巻き戻し順に依存しないよう、保存する側が書き出す時点で集計する。
        # summarize自身はJSON化できないので必ず除く。v9はこれで保存が落ちた。
        base = {key: value for key, value in receipt.items() if key != 'summarize'}
        return dict(base, replacements=len(receipts), call_count=len(record), calls=list(record),
                    unsatisfied_calls=[x for x in record if not x['original_require_satisfied']])
    receipt['summarize'] = summarize
    return receipt


def main() -> int:
    """effect許可の前段に設置する。認識codeと原G2ファイルは変更しない。"""
    args = G.arguments()
    output = Path(args.output)
    plan = G.read(args.plan)
    G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'm1_window_entry_pin')
    previous = G.protect_runtime
    state: dict = {}

    def protect(stack: Any, adapter: Any, main_module: Any, fixed: dict) -> None:
        previous(stack, adapter, main_module, fixed)
        state.update(install(stack, adapter.A.A.A.V4.replace_owned))

        def saved() -> None:
            G.save(output / 'G3_M1_WINDOW.json', state['summarize']())
        stack.callback(saved)
    G.protect_runtime = protect
    try:
        return E.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())
