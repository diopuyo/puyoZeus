"""予測入力(conditional_current)の却下理由を分解記録するG3専用sidecar。

`conditional_current.make()` (`data/verify/g2_hidden_current_candidate_2026-09-10_v1/
conditional_current.py:110-120`) には理由を残さない `return None` が6箇所あり、
`witness.py` (`data/verify/g2_history_baseline_adoption_2026-09-10_v4/witness.py`) の
採用経路にも同様の無言returnが複数ある。0件のとき「窓が塞いでいた」のか
「その区間に条件が無かった」のか区別できない (.claude/rules/01-verification-ladder.md 原則5)。

本ファイルはast で `make` 関数だけを取り出し、6箇所それぞれへ理由コードの
記録呼出を1回ずつ挿入して再構築する (g3_start_qualification_effect.py と同じ型)。
witness.py側は関数まるごとの外側wrapで、到来/採用試行/採用の3点を数える
(内部ロジックの複製をしない)。

記録先は本ファイル専用のsidecar `G3_PREDICTION_INPUT_FUNNEL.jsonl` のみ。
`control.hidden_current_events` には絶対に追加しない
(`data/verify/g2_conditional_finalizer_world_2026-09-10_v1/world_api.py:26-27` の
`unknown_hidden_event` でrunごと落ちるため)。`world_api.py:25` の
`require(bool(events), ...)` は緩めない。凍結資産はディスク上無変更。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from scripts import g3_postcommit_entry as E

G = E.G
FUNCTION = 'make'
MARKER = '_G3_FUNNEL_RECORD'
TARGET_SHA = {
    'data/verify/g2_hidden_current_candidate_2026-09-10_v1/conditional_current.py':
        '8808a6dd1093895bda2c8b5d57d98580cef4cbed9af10d909d41c7dc48854adc',
}
WITNESS_SHA = {
    'data/verify/g2_history_baseline_adoption_2026-09-10_v4/witness.py':
        '703c4b24450d8299a1cb934c8ec06b710fbc206e4b634f5164643e309f34f3b4',
}
# 各要素は (make内の完全一致テキスト, 理由コード)。順序はmake内の出現順。
REASON_PATCHES = [
    ("    if not getattr(binding,'hidden_tail_consumed',False): return None",
     'R113_tail_not_consumed'),
    ("    if ('hidden_transition' not in call and getattr(binding,'hidden_current',None) is None\n"
     "        and getattr(binding,'hidden_anchor',None) is None): return None",
     'R115_no_anchor_or_transition'),
    ("    if result is None or result.state.value!='stable' or values['ctx'].state.value!='stable': return None",
     'R116_not_stable'),
    ("    if values['signals'].effect_gate_window_active is not False: return None",
     'R117_effect_gate_active'),
    ("    if not fresh(control,call,result): return None",
     'R118_not_fresh'),
    ("    if captured is None: return None",
     'R120_no_observation'),
]
REASON_CODES = tuple(reason for _, reason in REASON_PATCHES)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_for(module: Any, table: dict) -> str | None:
    """実loadされたmoduleが対象tableのどれかならその相対keyを返す。"""
    file = getattr(module, '__file__', None)
    if not file:
        return None
    posix = Path(file).resolve().as_posix()
    for key in table:
        if posix.endswith(key):
            return key
    return None


def patch_block(block: str) -> str:
    """make関数のsource blockへ、6箇所の無言returnそれぞれに理由記録を挿入する。

    件数不一致・二重適用は名前つきで落とす (推測で直さない、原則5/6)。
    """
    G.require(MARKER not in block, 'funnel_already_present')
    for old, reason in REASON_PATCHES:
        G.require(block.count(old) == 1, 'funnel_literal_count:' + reason)
        call = "%s(control,view,'%s'); return None" % (MARKER, reason)
        block = block.replace(old, old.replace('return None', call), 1)
    return block


def rebuild_make(module: Any, key: str, recorder: Any) -> Any:
    """原ファイルから`make`だけを取り出し、理由記録つきで元namespaceへ再構築する。"""
    path = G.ROOT / key
    G.require(digest(path) == TARGET_SHA[key], 'funnel_conditional_source_sha')
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION]
    G.require(len(nodes) == 1, 'funnel_function_count')
    lines = text.splitlines()
    block = patch_block('\n'.join(lines[nodes[0].lineno - 1:nodes[0].end_lineno]))
    namespace = dict(vars(module))
    namespace[MARKER] = recorder
    exec(compile(block, str(path), 'exec'), namespace)
    value = namespace[FUNCTION]
    G.require(value.__code__.co_filename == str(path), 'funnel_origin')
    return value


class FunnelSink:
    """G3専用sidecarへ1行ずつflushしながら書き、side別の母数を保持する。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = path.open('x', encoding='utf-8')
        self.counts: dict[str, dict[str, Any]] = {}

    def _side(self, side: Any) -> dict[str, Any]:
        return self.counts.setdefault(str(side), dict(n2=0, n3=0, n4=0, n7=0, n8=0, n7r={}))

    def _emit(self, kind: str, side: Any, **fields: Any) -> None:
        row = dict(kind=kind, side=str(side), **fields)
        self.stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        self.stream.flush()

    def note_n2(self, side: Any) -> None:
        self._side(side)['n2'] += 1

    def note_n3(self, side: Any) -> None:
        self._side(side)['n3'] += 1

    def note_n4(self, side: Any, frame: Any) -> None:
        self._side(side)['n4'] += 1
        self._emit('n4_adopted', side, frame=frame)

    def note_n7(self, side: Any, frame: Any) -> None:
        self._side(side)['n7'] += 1

    def note_reason(self, side: Any, frame: Any, reason: str) -> None:
        entry = self._side(side)
        entry['n7r'][reason] = entry['n7r'].get(reason, 0) + 1
        self._emit('make_rejected', side, frame=frame, reason=reason)

    def note_n8(self, side: Any, frame: Any) -> None:
        self._side(side)['n8'] += 1
        self._emit('make_succeeded', side, frame=frame)

    def status(self) -> dict[str, Any]:
        return dict(sides=self.counts, sidecar=str(self.path))

    def close(self) -> None:
        self.stream.close()


def reason_recorder(sink: FunnelSink) -> Any:
    """makeの内側から`_G3_FUNNEL_RECORD(control,view,reason)`として呼ばれる。"""
    def record(control: Any, view: Any, reason: str) -> None:
        sink.note_reason(view.scope[-1], view.frame, reason)
    return record


def wrap_make(original: Any, sink: FunnelSink) -> Any:
    """make呼出のN7(到達)とN8(made)だけを外側で数える。理由の内訳は内側の記録に任せる。"""
    def make(control: Any, call: Any, result: Any, provisional: Any) -> Any:
        view = call['view']
        side = view.scope[-1]
        sink.note_n7(side, view.frame)
        value = original(control, call, result, provisional)
        if value is not None:
            sink.note_n8(side, view.frame)
        return value
    return make


def wrap_caller(original: Any, sink: FunnelSink) -> Any:
    """Witness.caller: 到来(N2) = added tokens を伴う行がここまで届いた回数。"""
    def caller(self: Any, row: Any, frame: Any) -> None:
        sink.note_n2(row.get('side'))
        return original(self, row, frame)
    return caller


def wrap_creation_in_scope(original: Any, sink: FunnelSink) -> Any:
    """Witness.creation_in_scope: 採用試行(N3) = 窓判定に入った回数。"""
    def creation_in_scope(self: Any, row: Any) -> bool:
        sink.note_n3(row.get('side'))
        return original(self, row)
    return creation_in_scope


def wrap_capture(original: Any, sink: FunnelSink) -> Any:
    """Witness.capture: 採用(N4) = 呼出前後でslotsの実体が入れ替わった回数。

    内部の各requireを複製せず、observableな状態(dict identity)の変化だけを見る。
    """
    def capture(self: Any, row: Any, caller: Any) -> Any:
        side = row.get('side')
        before = self.slots.get(side)
        result = original(self, row, caller)
        after = self.slots.get(side)
        if after is not None and after is not before:
            sink.note_n4(side, row.get('frame_idx'))
        return result
    return capture


def install_witness(stack: Any, module: Any, key: str, replace: Any, sink: FunnelSink) -> dict:
    """witness.pyのWitnessクラスへ、到来/採用試行/採用の3点だけを外側から数える。"""
    G.require(digest(G.ROOT / key) == WITNESS_SHA[key], 'funnel_witness_source_sha')
    cls = module.Witness
    replace(stack, cls, 'caller', wrap_caller(cls.caller, sink))
    replace(stack, cls, 'creation_in_scope', wrap_creation_in_scope(cls.creation_in_scope, sink))
    replace(stack, cls, 'capture', wrap_capture(cls.capture, sink))
    return dict(file=key, module=module.__name__, kind='witness',
                patched=['caller', 'creation_in_scope', 'capture'])


def apply(module: Any, stack: Any, replace: Any, sink: FunnelSink, receipts: list, done: set) -> None:
    """実loadされたmoduleが対象2種のどちらかなら、そのタイミングで一度だけ差し替える。"""
    if id(module) in done:
        return
    key = key_for(module, TARGET_SHA)
    if key is not None and hasattr(module, FUNCTION):
        done.add(id(module))
        rebuilt = rebuild_make(module, key, reason_recorder(sink))
        replace(stack, module, FUNCTION, wrap_make(rebuilt, sink))
        receipts.append(dict(file=key, module=module.__name__, kind='conditional_current',
                              reasons=list(REASON_CODES)))
        return
    wkey = key_for(module, WITNESS_SHA)
    if wkey is not None and hasattr(module, 'Witness'):
        done.add(id(module))
        receipts.append(install_witness(stack, module, wkey, replace, sink))


def install(stack: Any, replace: Any, output: Path) -> dict:
    """実loadされた対象moduleを全て捉え、scope終了で母数票を保存する。件数0を成功に読み替えない。"""
    sink = FunnelSink(output / 'G3_PREDICTION_INPUT_FUNNEL.jsonl')
    receipts: list[dict] = []
    done: set[int] = set()
    for module in list(sys.modules.values()):
        apply(module, stack, replace, sink, receipts, done)
    original_spec = importlib.util.spec_from_file_location
    all_keys = set(TARGET_SHA) | set(WITNESS_SHA)

    def spec_from_file_location(name: Any, location: Any = None, *args: Any, **kwargs: Any) -> Any:
        spec = original_spec(name, location, *args, **kwargs)
        if spec is None or getattr(spec, 'loader', None) is None or location is None:
            return spec
        posix = Path(location).resolve().as_posix()
        if not any(posix.endswith(key) for key in all_keys):
            return spec
        previous = spec.loader.exec_module

        def exec_module(module: Any) -> None:
            previous(module)
            apply(module, stack, replace, sink, receipts, done)
        spec.loader.exec_module = exec_module
        return spec
    replace(stack, importlib.util, 'spec_from_file_location', spec_from_file_location)

    def ended() -> None:
        status = sink.status() | dict(receipts=receipts, targets_loaded=len(receipts),
                                       no_target_loaded=not receipts)
        G.save(output / 'G3_PREDICTION_INPUT_FUNNEL_STATUS.json', status)
        sink.close()
    stack.callback(ended)
    return dict(receipts=receipts)


def main() -> int:
    """既存postcommit入口の前段に設置する。認識codeと原G2ファイルは変更しない。"""
    args = G.arguments()
    output = Path(args.output)
    plan = G.read(args.plan)
    G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'funnel_entry_pin')
    previous = G.protect_runtime
    state: dict = {}

    def protect(stack: Any, adapter: Any, main_module: Any, fixed: dict) -> None:
        previous(stack, adapter, main_module, fixed)
        replace = adapter.A.A.A.V4.replace_owned
        state.update(install(stack, replace, output))
    G.protect_runtime = protect
    try:
        return E.main()
    finally:
        G.protect_runtime = previous


if __name__ == '__main__':
    raise SystemExit(main())
