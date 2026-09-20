"""source39候補の次の入口単位だけを検査し、後段対応を合格化しない。"""
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace as N
from typing import Any
import copy
import importlib.util
import sys

import pytest

from scripts import g3_video39_candidate_scope as S
from scripts import g3_video39_candidate_prepare as P


def common() -> Any:
    value = ModuleType('_source39_cpu_common')
    value.__file__ = str(S.G.COMMON_FILE)
    value.__dict__.update(FIRST=S.OLD_FIRST, END=S.OLD_ENDS[-1], FPS=S.OLD_FPS,
                          STRIDE=S.OLD_STRIDE, SOURCE=S.OLD_SOURCE_SHA, FRAMES=())
    exec(compile('def bounds():\n return dict(first_frame=FIRST,end_exclusive=END)\n',
                 str(S.G.COMMON_FILE), 'exec'), vars(value))
    return value


def test_scope_old_clock_and_identity_restored(monkeypatch: Any) -> None:
    module = common()
    before = dict(vars(module))
    ns = dict(FIRST_FRAME=S.OLD_FIRST, END_FRAME=S.OLD_ENDS[-1])
    exec('def begin_frame(): pass', ns)
    history = N(HistoryRecorder=N(begin_frame=ns['begin_frame']))
    main_ns = dict(K=module, S=N(K=module), Q=N(K=module), R=N(K=module))
    main = N(__globals__=main_ns)
    monkeypatch.setattr(S.G, 'SOURCE_SHA', 'source39_fixture')
    with ExitStack() as stack:
        receipts = S.bind_bounds(stack, main, history)
        assert module.FPS == 30 and module.STRIDE == 1 and len(module.FRAMES) == 11878
        assert module.SOURCE == 'source39_fixture' and ns['END_FRAME'] == 11878
        assert receipts[0]['old_fps'] == 60 and receipts[0]['old_stride'] == 2
    assert vars(module) == before and ns['FIRST_FRAME'] == S.OLD_FIRST
    module.FPS = 30
    with ExitStack() as stack, pytest.raises(ValueError, match='old_collection_scope'):
        S.bind_bounds(stack, main, history)


def test_receipt_uses_new_source_and_calibration(tmp_path: Path, monkeypatch: Any) -> None:
    old = dict(video_path='old_video38', input_and_code_sha256={'old_video38': S.OLD_SOURCE_SHA},
               source_id=S.OLD_SOURCE_SHA, original_collection_tokens=['old'])
    reference = dict(video_path='old_video38', prepare_receipt='old_prepare.json', prepare_sha256='reference_sha')
    contract = dict(source_video_id=S.B.SOURCE_NAME, source_video_path='data/frames/video_39.mp4',
                    source_video_sha256='video39_sha', time_base_numerator=1, time_base_denominator=30,
                    width=1920, height=1080, frame_count=94380,
                    processing_start_frame=0, processing_end_frame_exclusive=94380)
    prereg = dict(sources=[dict(source_video_id=S.B.SOURCE_NAME, source=contract,
        required_continuous_processing=dict(start_frame=0, end_frame_exclusive=11878,
                                            seek_between_windows=False))])
    def document(path: Path, sha: str) -> dict:
        if path == S.B.M.PREREG:
            return copy.deepcopy(prereg)
        return copy.deepcopy(reference if path == S.B.REFERENCE_PLAN else old)
    monkeypatch.setattr(S.B.M, 'document', document)
    monkeypatch.setattr(S.G, 'SOURCE', tmp_path / 'video39.mp4')
    monkeypatch.setattr(S.G, 'SOURCE_SHA', 'video39_sha')
    plan = dict(reference_prepare='old_prepare.json', reference_sha='reference_sha',
                source_id='sha256:video39_sha', runtime_pins={str(S.G.SOURCE): 'video39_sha'},
                tokens=['--score-region-calibration', str(S.B.CALIBRATION)],
                gt_scoring='NOT_SCORED_NO_CERTIFIED_GT', _sha256='new_plan')
    checks: list[dict] = []
    receipt, config = S.receipt_for(plan, tmp_path, N(assert_unchanged=checks.append))
    assert receipt['source_id'] == 'video39_sha' and receipt['video_path'] == str(S.G.SOURCE)
    assert receipt['expected_frame_count'] == 11878 and '11877' in receipt['initialization']
    assert config['collection_tokens'] == plan['tokens'] and checks
    assert receipt['downstream_source_clock_not_verified'] and not receipt['quality_gate_clear']
    assert old['source_id'] == S.OLD_SOURCE_SHA
    assert receipt['original_source']['source_video_sha256'] == 'video39_sha'
    assert receipt['original_source']['processing_end_frame_exclusive'] == 94380
    assert (receipt['original_source']['time_base_numerator'],
            receipt['original_source']['time_base_denominator']) == (1, 30)


def test_sampling_overrides_legacy_two(monkeypatch: Any, tmp_path: Path) -> None:
    collector = N(collect_lean=lambda *args, **kwargs: None)
    def main() -> None:
        collector.collect_lean(S.G.SOURCE, tmp_path/'unused', sample_interval_frames=2,
                               sample_interval_sec=0, normalize_fps_30=True)
    collector.main = main
    original = collector.collect_lean
    result = S.arguments(collector, dict(collection_tokens=[]))
    assert result['sample_interval_frames'] == 1 and result['max_sec'] == 11878/30
    assert collector.collect_lean is original


def test_gpu_path_is_explicitly_rejected(monkeypatch: Any) -> None:
    monkeypatch.setattr(P.G, 'arguments', lambda: N(cpu_prepare=False, arm='candidate'))
    with pytest.raises(ValueError, match='cpu_prepare_only'):
        P.main()


def test_real_bridge_stride_counterexample_and_restore(monkeypatch: Any) -> None:
    name = '_source39_original_whole'
    spec = importlib.util.spec_from_file_location(name, P.WHOLE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    marker = RuntimeError('reached_authentic_code_after_frames')
    def authentic(collector: Any) -> None:
        raise marker
    monkeypatch.setattr(module, 'authentic_code', authentic)
    args = (None, {}, None, None, None, (0, 1, 2), None)
    with pytest.raises(ValueError, match='frames'):
        module.Bridge(*args)
    adapter = N(A=N(A=N(A=N(V4=N(A=N(W=module))))))
    with ExitStack() as stack:
        P.whole_clock(stack, adapter)
        assert (module.FPS, module.STRIDE) == (30, 1)
        with pytest.raises(RuntimeError) as error:
            module.Bridge(*args)
        assert error.value is marker
    assert (module.FPS, module.STRIDE) == (60, 2)


def loaded_whole(monkeypatch: Any, name: str) -> Any:
    """元sourceを実loadし、実入口と同じmodule条件でwhole_clockへ渡す。"""
    spec = importlib.util.spec_from_file_location(name, P.WHOLE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_wrapped_bridge_resolves_to_original_class(monkeypatch: Any) -> None:
    """実runtimeがBridgeを包んでも、元classのglobals検査を通して時計を置換する。"""
    module = loaded_whole(monkeypatch, '_source39_wrapped_whole')
    original = module.Bridge
    def bridge(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'Bridge', bridge)
    adapter = N(A=N(A=N(A=N(V4=N(A=N(W=module))))))
    assert P.original_bridge(module) is original
    with ExitStack() as stack:
        P.whole_clock(stack, adapter)
        assert (module.FPS, module.STRIDE) == (30, 1)
        assert module.Bridge is bridge
    assert (module.FPS, module.STRIDE) == (S.OLD_FPS, S.OLD_STRIDE)


def test_bridge_without_original_owner_is_named_counterexample(monkeypatch: Any) -> None:
    """元classを保持しない差し替えは、名前付きG3失敗で拒否し時計を変えない。"""
    module = loaded_whole(monkeypatch, '_source39_orphan_whole')
    def bridge(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('not_called')
    monkeypatch.setattr(module, 'Bridge', bridge)
    adapter = N(A=N(A=N(A=N(V4=N(A=N(W=module))))))
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='source39_whole_bridge_owner'):
            P.whole_clock(stack, adapter)
    assert (module.FPS, module.STRIDE) == (S.OLD_FPS, S.OLD_STRIDE)


def test_protect_runs_whole_clock_before_previous(monkeypatch: Any) -> None:
    """実入口の実行順を固定する。previousがBridgeを包む前に元globalsを検査する。"""
    module = loaded_whole(monkeypatch, '_source39_order_whole')
    adapter = N(A=N(A=N(A=N(V4=N(A=N(W=module))))))
    order, captured = [], {}
    def previous(owner_stack: Any, owner_adapter: Any, main: Any, fixed: dict) -> None:
        order.append('previous')
        captured['bridge_is_class'] = isinstance(module.Bridge, type)
        captured['clock'] = (module.FPS, module.STRIDE)
        original = module.Bridge
        def bridge(*args: Any, **kwargs: Any) -> Any:
            return original(*args, **kwargs)
        module.Bridge = bridge
    def clock(owner_stack: Any, owner_adapter: Any) -> None:
        order.append('whole_clock')
        real(owner_stack, owner_adapter)
    real = P.whole_clock
    monkeypatch.setattr(P.G, 'protect_runtime', previous)
    monkeypatch.setattr(S, 'install_source_identity', lambda stack, adapter: {})
    monkeypatch.setattr(P, 'whole_clock', clock)
    holder: dict = {}
    monkeypatch.setattr(P.G, 'bind_bounds', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(P.B, 'bind', lambda stack: N(), raising=False)
    monkeypatch.setattr(P.G, 'capture_scope', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(P.G, 'receipt_for', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(P.B, 'checked_capture', S.B.checked_capture, raising=False)
    with ExitStack() as stack:
        P.install(stack)
        holder['protect'] = P.G.protect_runtime
        holder['protect'](stack, adapter, None, {})
    assert order == ['whole_clock', 'previous']
    assert captured['bridge_is_class'] is True
    assert captured['clock'] == (30, 1)
    assert (module.FPS, module.STRIDE) == (S.OLD_FPS, S.OLD_STRIDE)


def identity_module(tmp_path: Path, suffix: str, attribute: str, value: str) -> Any:
    module = ModuleType('_source39_identity_' + attribute.lower())
    path = tmp_path / suffix
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')
    module.__file__ = str(path)
    setattr(module, attribute, value)
    return module


def test_identity_retarget_replaces_only_checked_old_value(tmp_path: Path) -> None:
    """旧video38識別子だけを置換し、外側scope終了で元値へ戻す。"""
    suffix = 'g2_normal_completion_transaction_2026-09-09_v1/live_provider.py'
    old = 'sha256:' + S.OLD_SOURCE_SHA
    module = identity_module(tmp_path, suffix, 'SOURCE', old)
    def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
        stack.callback(setattr, owner, name, getattr(owner, name))
        setattr(owner, name, value)
    done: set = set()
    with ExitStack() as stack:
        receipts = S.retarget(stack, module, replace, done)
        assert len(receipts) == 1 and receipts[0]['attribute'] == 'SOURCE'
        assert module.SOURCE == 'sha256:' + S.G.SOURCE_SHA
        assert S.retarget(stack, module, replace, done) == []
    assert module.SOURCE == old


def test_identity_rejects_unexpected_old_value(tmp_path: Path) -> None:
    """想定外の旧値は名前付きG3失敗にし、黙って上書きしない。"""
    suffix = 'g2_palette_finish_proof_2026-09-09_v1/adapter.py'
    module = identity_module(tmp_path, suffix, 'SOURCE_ID', 'sha256:unexpected')
    def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
        setattr(owner, name, value)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='source39_identity_old_value'):
            S.retarget(stack, module, replace, set())
    assert module.SOURCE_ID == 'sha256:unexpected'
