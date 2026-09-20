"""原session.configuredのcleanupを通す。依存設置/torch値は人工、GPU再生成は不要。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from types import CodeType, FunctionType, ModuleType, SimpleNamespace as N
from typing import Any, Iterator
import constructor_observation as C


def test_original_cleanup_removes_venv_torch_after_receipt(monkeypatch: Any) -> None:
    project = Path(__file__).resolve().parents[3]
    source = project / 'data/verify/g2_history_publication_probe_runtime_2026-09-10_v13/session.py'
    module = compile(source.read_bytes(), str(source), 'exec', dont_inherit=True)
    code = next(c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == 'configured')
    @contextmanager
    def installed() -> Iterator[Any]:
        yield {}
    namespace = dict(sys=sys, Path=Path, ExitStack=ExitStack, installed=installed,
                     REPEAT=N(load=lambda stack: None), K=N(PROJECT=project))
    configured = contextmanager(FunctionType(code, namespace))
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    with configured():
        torch = ModuleType('torch')
        torch.__file__ = str(project / 'venv/lib/python3.12/site-packages/torch/__init__.py')
        torch.cuda = N(is_initialized=lambda: True, get_device_name=lambda index: 'artificial-device')
        monkeypatch.setitem(sys.modules, 'torch', torch)
        receipt = C.gpu_receipt()
    assert 'torch' not in sys.modules
    assert receipt == dict(gpu=True, gpu_device='artificial-device', video_frame_inference=False)
