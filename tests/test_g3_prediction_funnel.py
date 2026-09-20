"""G3予測入力の却下理由分解記録を、実G2ファイル+人工の呼出引数で検査する。

conditional_current.py/witness.py はdata/verify配下の凍結資産を読み取り専用で
loadするだけで、一切書き換えない。実走ランナー・実動画は使わない。
"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

import pytest

from scripts import g3_prediction_funnel as F

CONDITIONAL_KEY = 'data/verify/g2_hidden_current_candidate_2026-09-10_v1/conditional_current.py'
WITNESS_KEY = 'data/verify/g2_history_baseline_adoption_2026-09-10_v4/witness.py'


def _pop_target_modules() -> dict[str, Any]:
    """sys.modules上の対象2ファイル一致分を退避する(戻り値で復元できる形)。"""
    saved: dict[str, Any] = {}
    for name, module in list(sys.modules.items()):
        if F.key_for(module, F.TARGET_SHA) or F.key_for(module, F.WITNESS_SHA):
            saved[name] = sys.modules.pop(name)
    return saved


@pytest.fixture(autouse=True)
def _isolate_target_modules() -> Any:
    """他test(同一pytestセッション内の他ファイル含む)がsys.modulesへ残した実
    conditional_current.py/witness.py一致分を、本ファイルのtest実行中だけ退避する。

    F.install()はsys.modules全体を走査するため、他test由来の残留物を無視できないと
    「0件」の前提が崩れる(測定器自体の健全性を先に確認する、原則7相当)。
    """
    saved = _pop_target_modules()
    yield
    for extra in _pop_target_modules().values():
        del extra  # 本test中に新規loadした分は捨て、他test由来の分だけ戻す。
    sys.modules.update(saved)


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    old = getattr(owner, name)
    stack.callback(setattr, owner, name, old)
    setattr(owner, name, value)


def load_conditional(monkeypatch: Any, name: str) -> Any:
    """実conditional_current.pyを、兄弟依存2directoryをsys.pathへ足して読み込む。"""
    root = F.G.ROOT
    monkeypatch.syspath_prepend(str(root / 'data/verify/g2_hidden_two_hand_candidate_2026-09-10_v1'))
    monkeypatch.syspath_prepend(str(root / 'data/verify/g2_hidden_tail_candidate_2026-09-10_v1'))
    path = root / CONDITIONAL_KEY
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def load_witness(monkeypatch: Any, name: str) -> Any:
    path = F.G.ROOT / WITNESS_KEY
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def make_view(side: str = '1P', frame: int = 100) -> Any:
    return N(frame=frame, clock=1.0, scope=('src', 'run', 0, 1, 2, 0, side), refs=(), quiet=True)


def base_binding() -> Any:
    return N(owner=N(state=N()))


# ---- patch_block: 純粋文字列変換 (原則6: 分解記録を先に仕込む) ----

def test_patch_block_inserts_all_six_reason_codes() -> None:
    text = F.G.ROOT.joinpath(CONDITIONAL_KEY).read_text(encoding='utf-8')
    import ast
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make')
    lines = text.splitlines()
    block = '\n'.join(lines[node.lineno - 1:node.end_lineno])
    patched = F.patch_block(block)
    for reason in F.REASON_CODES:
        assert ("%s(control,view,'%s')" % (F.MARKER, reason)) in patched
    ast.parse(patched)  # 構文として壊れていないこと


def test_patch_block_rejects_double_apply() -> None:
    block = "def make():\n    if True: return None\n"
    once = block.replace('return None', "%s(); return None" % F.MARKER)
    with pytest.raises(ValueError, match='funnel_already_present'):
        F.patch_block(once)


def test_patch_block_rejects_missing_or_duplicate_literal() -> None:
    with pytest.raises(ValueError, match='funnel_literal_count:R113_tail_not_consumed'):
        F.patch_block("def make():\n    pass\n")


# ---- rebuild_make: 実ファイルへ適用し、6理由それぞれの単独発火を確認 ----

REASON_CASES = [
    dict(name='r113', reason='R113_tail_not_consumed',
         binding=dict(), call_extra={}, values={}, fresh=None, observed=None, result=None),
    dict(name='r115', reason='R115_no_anchor_or_transition',
         binding=dict(hidden_tail_consumed=True), call_extra={}, values={}, fresh=None, observed=None, result=None),
    dict(name='r116', reason='R116_not_stable',
         binding=dict(hidden_tail_consumed=True), call_extra=dict(hidden_transition=True), values={},
         fresh=None, observed=None, result=None),
    dict(name='r117', reason='R117_effect_gate_active',
         binding=dict(hidden_tail_consumed=True), call_extra=dict(hidden_transition=True),
         values=dict(ctx=N(state=N(value='stable')), signals=N(effect_gate_window_active=True)),
         fresh=None, observed=None, result=N(state=N(value='stable'))),
    dict(name='r118', reason='R118_not_fresh',
         binding=dict(hidden_tail_consumed=True), call_extra=dict(hidden_transition=True),
         values=dict(ctx=N(state=N(value='stable')), signals=N(effect_gate_window_active=False)),
         fresh=lambda control, call, result: False, observed=None, result=N(state=N(value='stable'))),
    dict(name='r120', reason='R120_no_observation',
         binding=dict(hidden_tail_consumed=True), call_extra=dict(hidden_transition=True),
         values=dict(ctx=N(state=N(value='stable')), signals=N(effect_gate_window_active=False)),
         fresh=lambda control, call, result: True, observed=lambda *a, **k: None,
         result=N(state=N(value='stable'))),
]


@pytest.mark.parametrize('case', REASON_CASES, ids=[c['name'] for c in REASON_CASES])
def test_each_reason_fires_exactly_once(monkeypatch: Any, case: dict) -> None:
    module = load_conditional(monkeypatch, '_funnel_' + case['name'])
    if case['fresh'] is not None:
        monkeypatch.setattr(module, 'fresh', case['fresh'])
    if case['observed'] is not None:
        monkeypatch.setattr(module, 'W', N(observed=case['observed']))
    recorded: list[tuple[Any, Any, str]] = []

    def recorder(control: Any, view: Any, reason: str) -> None:
        recorded.append((view.scope[-1], view.frame, reason))
    rebuilt = F.rebuild_make(module, CONDITIONAL_KEY, recorder)
    binding = base_binding()
    for key, value in case['binding'].items():
        setattr(binding, key, value)
    view = make_view(frame=222)
    call = dict(binding=binding, view=view, frame=N(f_locals=dict(case['values'])))
    call.update(case['call_extra'])
    result = rebuilt(N(), call, case['result'], None)
    assert result is None
    assert recorded == [('1P', 222, case['reason'])]


def test_rebuild_make_source_sha_guard(monkeypatch: Any) -> None:
    module = load_conditional(monkeypatch, '_funnel_sha')
    monkeypatch.setitem(F.TARGET_SHA, CONDITIONAL_KEY, '0' * 64)
    with pytest.raises(ValueError, match='funnel_conditional_source_sha'):
        F.rebuild_make(module, CONDITIONAL_KEY, lambda *a: None)


# ---- wrap_make: N7(到達)/N8(made)を外側だけで数える ----

def test_wrap_make_counts_attempts_and_success(tmp_path: Path) -> None:
    sink = F.FunnelSink(tmp_path / 'funnel.jsonl')
    calls: list[Any] = []

    def original(control: Any, call: Any, result: Any, provisional: Any) -> Any:
        calls.append(call)
        return call.get('outcome')
    wrapped = F.wrap_make(original, sink)
    view = make_view(side='2P', frame=10)
    wrapped(None, dict(view=view, outcome=None), None, None)
    wrapped(None, dict(view=view, outcome=('value', '2P')), None, None)
    sink.close()
    assert sink.counts['2P']['n7'] == 2
    assert sink.counts['2P']['n8'] == 1


# ---- witness側: N2/N3/N4を外側だけで数える (内部require複製なし) ----

def test_wrap_caller_and_creation_in_scope_count(tmp_path: Path) -> None:
    sink = F.FunnelSink(tmp_path / 'funnel.jsonl')
    seen: list[str] = []
    caller = F.wrap_caller(lambda self, row, frame: seen.append('caller'), sink)
    scope = F.wrap_creation_in_scope(lambda self, row: True, sink)
    row = dict(side='1P', frame_idx=5)
    caller(None, row, None)
    scope(None, row)
    sink.close()
    assert sink.counts['1P']['n2'] == 1 and sink.counts['1P']['n3'] == 1
    assert seen == ['caller']


def test_wrap_capture_counts_n4_only_on_new_slot(tmp_path: Path) -> None:
    sink = F.FunnelSink(tmp_path / 'funnel.jsonl')

    class Owner:
        def __init__(self) -> None:
            self.slots: dict[str, Any] = {}
    owner = Owner()

    def creates_slot(self: Any, row: Any, caller: Any) -> None:
        self.slots[row['side']] = dict(token='t1')

    def rejects(self: Any, row: Any, caller: Any) -> None:
        return None
    wrapped_create = F.wrap_capture(creates_slot, sink)
    wrapped_reject = F.wrap_capture(rejects, sink)
    wrapped_create(owner, dict(side='1P', frame_idx=7), None)
    wrapped_reject(owner, dict(side='2P', frame_idx=8), None)
    sink.close()
    assert sink.counts['1P']['n4'] == 1
    assert '2P' not in sink.counts or sink.counts['2P']['n4'] == 0


# ---- install/apply: 実2ファイルへの配線と、母数票の0件/未測定の区別 ----

def test_install_patches_both_real_targets_and_restores(monkeypatch: Any, tmp_path: Path) -> None:
    conditional = load_conditional(monkeypatch, '_funnel_install_conditional')
    witness = load_witness(monkeypatch, '_funnel_install_witness')
    original_make = conditional.make
    original_caller = witness.Witness.caller
    with ExitStack() as stack:
        result = F.install(stack, replace, tmp_path)
        # 件数そのものは環境依存 (同じファイルが別名でも load されうる)。
        # 実走では「実際に使われている実体」を取りこぼさないため全部を差し替えるのが
        # 正しい挙動なので、**件数を固定しない**。確かめるのは次の3つ。
        kinds = {item['kind'] for item in result['receipts']}
        assert kinds == {'conditional_current', 'witness'}, "両方の種類が差し替わること"
        assert conditional.make is not original_make, "本testが読んだ実体が差し替わること"
        assert witness.Witness.caller is not original_caller
        assert (tmp_path / 'G3_PREDICTION_INPUT_FUNNEL.jsonl').exists()
    assert conditional.make is original_make, "scope終了で必ず元へ戻ること"
    assert witness.Witness.caller is original_caller
    status = json.loads((tmp_path / 'G3_PREDICTION_INPUT_FUNNEL_STATUS.json').read_text(encoding='utf-8'))
    # 票は件数を持ち、受領票と自己整合していること (0件と未測定を区別するため)。
    assert status['targets_loaded'] == len(result['receipts'])
    assert status['targets_loaded'] >= 2 and status['no_target_loaded'] is False
    assert status['sides'] == {}  # 呼出0件。ただし未測定ではない(targets_loadedで判別)


def test_install_records_zero_targets_as_unmeasured(tmp_path: Path) -> None:
    """対象moduleが一つもloadされない場合、0件と未測定が票の別項目で区別できる。

    前提 (対象が1つもloadされていない) を、fixture の順序に頼らず
    **この test の中で自分で作る**。他testが sys.path を汚して別名で読み込んだ実体が
    残っていると前提が崩れ、測る対象がずれる (原則7: 測定器を本番と同じ条件にする)。
    """
    saved = _pop_target_modules()
    try:
        assert _pop_target_modules() == {}, "前提の作り方が効いていない"
        with ExitStack() as stack:
            result = F.install(stack, replace, tmp_path)
            assert result['receipts'] == []
        status = json.loads(
            (tmp_path / 'G3_PREDICTION_INPUT_FUNNEL_STATUS.json').read_text(encoding='utf-8'))
        assert status['targets_loaded'] == 0 and status['no_target_loaded'] is True
    finally:
        sys.modules.update(saved)
    assert status['sides'] == {}


def test_witness_source_sha_guard(monkeypatch: Any, tmp_path: Path) -> None:
    load_witness(monkeypatch, '_funnel_witness_sha')
    monkeypatch.setitem(F.WITNESS_SHA, WITNESS_KEY, '0' * 64)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='funnel_witness_source_sha'):
            F.install(stack, replace, tmp_path)
