"""CNNの演算を変えず、ONNXの実行スレッドだけを切り替える候補。"""
from __future__ import annotations

import os
from typing import Any

from src.production_config import LIVE_ONNX_PARALLEL_DEFAULT

THREADS = 2
CACHE_SUFFIX = '|parallel2-no-spin'


def requested() -> bool:
    """配布設定から明示指定されたときだけ有効にする。"""
    return os.environ.get('PUYO_LIVE_ONNX_PARALLEL',
                          str(int(LIVE_ONNX_PARALLEL_DEFAULT))) == '1'


def configure(options: Any) -> None:
    """intraだけを2にし、二つのモデルの待機スピンによるCPU占有を防ぐ。"""
    options.intra_op_num_threads = THREADS
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
