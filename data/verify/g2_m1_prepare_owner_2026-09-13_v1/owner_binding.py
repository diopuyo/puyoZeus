"""実whole guardのL参照を、同sourceの私有所有loaderへ明示的に束縛する。"""
from pathlib import Path
import sys
from typing import Any

SOURCE = Path(__file__).resolve().parent.parent / 'g2_live_probability_context_2026-09-12_v1/loader.py'
ALIAS = '_g2_live_probability_owned_loader'


def install(stack: Any, whole: Any, owner: Any, main: Any, replace: Any) -> None:
    """原guardのcode/globalsを保ち、元module identityを同stackで復元する。"""
    guard = main.__globals__['collect'].__globals__['constructor_guard']
    if sys.modules.get(ALIAS) is not owner or Path(owner.__file__).resolve() != SOURCE:
        raise ValueError('prepare_private_owner')
    if guard.__globals__ is not vars(whole):
        raise ValueError('prepare_guard_globals')
    if Path(guard.__code__.co_filename).resolve() != SOURCE.parent / 'whole_live_adapter.py':
        raise ValueError('prepare_guard_code')
    previous = whole.L
    if previous is owner or Path(previous.__file__).resolve() != SOURCE:
        raise ValueError('prepare_original_loader')
    replace(stack, whole, 'L', owner)
    if guard.__globals__['L'] is not owner:
        raise ValueError('prepare_guard_owner_not_selected')
