"""206の仮説を元Native開始境界・失敗時保存契約で独立検査する。"""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import target_entry as T


def test_initial_call_cannot_reach_notice_capture() -> None:
    with ExitStack() as stack:
        T.A.configured(stack)
        import probe_native_merge
        owner = sys.modules[T.A.A.A.A.V4.OWNED_ALIAS]
        parts = owner.dependencies().modules()
        sys.path.insert(0, str(T.A.NOTICE))
        spec = importlib.util.spec_from_file_location('_notice_contract_input', T.A.NOTICE / 'test_settled_notice.py')
        F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(F)
        f = F.setup(parts)
        with pytest.raises(ValueError, match='native_frame_bound'):
            F.step(f, 35806, F.notice())
        assert not f.packets and not f.mode.native.seen_calls
        F.step(f, 35808, F.notice())
        assert len(f.packets) == 1


def test_original_failure_preserves_notices_without_success_review(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / 'failed'
    def write(path: Path, value: Any) -> None:
        with path.open('x') as stream:
            json.dump(value, stream)
    def forbidden(*args: Any) -> None:
        raise AssertionError('失敗runを成功検収してはならない')
    def fail_run(path: Path, extra: Any) -> None:
        path.mkdir()
        write(path / 'SECOND_SETTLED_NOTICES.jsonl', dict(saved_before_failure=True))
        raise ValueError('original_unrelated_step_failure')
    main = N(__globals__=dict(K=N(write=write), Q=N(seal=forbidden)))
    monkeypatch.setattr(T, 'selected_run', lambda selected: fail_run)
    monkeypatch.setitem(sys.modules, '_g2_second_notice_saved_v21', N(verify=forbidden))
    code = T.observe(main, output)
    entry = json.loads((output / 'ENTRY_RESULT.json').read_bytes())
    assert code == 1 and 'original_unrelated_step_failure' in entry['error']
    assert json.loads((output / 'SECOND_SETTLED_NOTICES.jsonl').read_bytes())['saved_before_failure']
    assert not (output / 'SECOND_SETTLED_NOTICE_REVIEW.json').exists()
