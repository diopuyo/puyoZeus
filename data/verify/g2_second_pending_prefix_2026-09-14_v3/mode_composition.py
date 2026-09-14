"""原通知→prefix→原物理の協調MROをSession毎に組み立てる。"""
from pathlib import Path
from types import CodeType
from typing import Any
import settled_notice as NOTICE

SOURCE = Path(__file__).resolve().parent.parent/'g2_second_origin_postrun_2026-09-13_v1/settled_notice.py'
ALLOWED = frozenset(('__module__','B','__init__','capture_origin','__doc__'))


def verify_notice(selected: type, arrival: Any) -> None:
    require = arrival.B.require
    require(Path(NOTICE.__file__).resolve()==SOURCE, 'second_prefix_notice_source')
    require(selected.__bases__==(arrival.BASE.Mode,) and selected.B is arrival.B,
            'second_prefix_notice_base')
    require(set(vars(selected))<=ALLOWED, 'second_prefix_notice_extra_override')
    nested = [c for c in NOTICE.mode_class.__code__.co_consts
              if isinstance(c,CodeType) and c.co_name=='Mode']
    require(len(nested)==1, 'second_prefix_notice_factory_code')
    codes = {c.co_name:c for c in nested[0].co_consts if isinstance(c,CodeType)}
    for name in ('__init__','capture_origin'):
        method = vars(selected).get(name)
        require(getattr(method,'__code__',None) is codes[name]
                and method.__globals__ is vars(NOTICE), 'second_prefix_notice_method')


def compose(notice: type, arrival: Any, adapter: Any, engine: Any, serializer: Any) -> type:
    verify_notice(notice,arrival)
    prefix = adapter.mode_class(arrival.BASE.Mode,engine,serializer)
    class Combined(notice,prefix):
        pass
    arrival.B.require(Combined.__mro__[:4]==(Combined,notice,prefix,arrival.BASE.Mode),
                      'second_prefix_composed_mro')
    return Combined
