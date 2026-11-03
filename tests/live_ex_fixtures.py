"""正式版5c7c6333の試験盤を、EX専用の型で作る。"""
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World
from src.phase_j._live_ex_frozen.src.prefire_ledger_v410 import Ledger

EMPTY = (0,) * 78


def context() -> Context:
    return Context(World(0., [EMPTY, EMPTY], Ledger()), 0., 0., 0., 0, 1,
                   (0., 0.), (None, None), ((), ()), (1, 2, 3, 4), (0, 0), (0, 0))
