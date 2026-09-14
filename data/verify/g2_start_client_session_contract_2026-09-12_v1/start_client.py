"""新開始資格workerの限定入口。原Clientの保存/例外/closeを保持する。"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import FunctionType
import parent_client_v2 as C

# 元consumerが参照するtransportを同じ実体として公開する。
T = C.T

ROOT = Path(__file__).resolve().parent
PINS: dict[str, str] = {"start_join.py":"aa66abc3581f3e4d45b9c69fc5bcb457f3849edef4b129c759f8a80facdbb647","start_qualification.py":"51b4e4c8ce978a49121681fcc13f3ab6cdb55afa9a2f9bdf7b029be8d548b2d1","start_worker.py":"4d8da1648574c1d754fa1b33a9992bd794564b2c748a9e42e91b57ae75e83e62"}


def worker_path() -> Path:
    C.T.guard()
    C.T.P.require(set(PINS) == {'start_worker.py', 'start_join.py', 'start_qualification.py'}, 'start_worker_pins')
    for name, expected in PINS.items():
        C.T.P.require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'start_worker_source')
    return ROOT / 'start_worker.py'


class Client(C.Client):
    __init__ = FunctionType(C.Client.__init__.__code__, dict(vars(C), worker_path=worker_path),
                            C.Client.__init__.__name__, C.Client.__init__.__defaults__,
                            C.Client.__init__.__closure__)
    __init__.__kwdefaults__ = C.Client.__init__.__kwdefaults__
