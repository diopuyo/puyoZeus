"""原load計装のdevice契約を原codeで確認。パラメータは人工でモデル生成ではない。"""
from __future__ import annotations
from pathlib import Path
from types import CellType, CodeType, FunctionType, SimpleNamespace as N
from typing import Any
import pytest

SOURCE = Path(__file__).resolve().parents[3] / 'scripts/diagnose_video38_accounting_history_v1.py'


def children(code: CodeType) -> list[CodeType]:
    result = [code]
    for value in code.co_consts:
        if isinstance(value, CodeType):
            result.extend(children(value))
    return result


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_original_device_guard(device: str) -> None:
    module = compile(SOURCE.read_bytes(), str(SOURCE), 'exec', dont_inherit=True)
    instrument = next(code for code in children(module) if code.co_name == 'instrument_pipeline')
    code = next(code for code in children(instrument) if code.co_name == 'load')
    model = N(parameters=lambda: iter([N(device=device)]))
    pipe = N(_reader=N(_classifier=N(_cnn=N(_model=model))))
    original = classmethod(lambda cls, **kwargs: pipe)
    rec = N(pipeline_receipt={})
    values = dict(original_load=original, rec=rec)
    function = FunctionType(code, dict(base=N(json_value=lambda value: value), frozen_modules=lambda: []),
                            closure=tuple(CellType(values[name]) for name in code.co_freevars))
    if device == 'cpu':
        with pytest.raises(RuntimeError, match='実CNNが指定CUDAではありません: cpu'):
            function(object, flag=True)
        assert rec.pipeline_receipt == {}
    else:
        assert function(object, flag=True) is pipe
        assert rec.pipeline_receipt['board_cnn_device'] == 'cuda:0'
