"""新開始資格workerの限定入口。原Clientの保存/例外/closeを保持する。"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import FunctionType
import parent_client_v2 as C

ROOT = Path(__file__).resolve().parent
PINS: dict[str, str] = {"start_worker.py":"ff5e5e70e80aa511a00069ac433fea834386c2b4ce01cd286340a8f4062ba13d","start_join.py":"a771271e2687865be9c9ad395e7d87b647938dcbd29d917fc220d9c7506ec0c6","start_qualification.py":"f23dcd60c70d8014a90585cd395127c5dd2cd1e288b508e79723249a5e24cbec"}


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
