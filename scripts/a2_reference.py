"""c13ea95だけを読み込む参照再生。モデルの相対パスは元worktreeを使う。"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    """run_limitedが数えるモジュール起動を保ち、参照コードの読込み先を検証する。"""
    reference = Path(__file__).resolve().parents[1]/'logs/a2_safety/reference'
    sys.path.insert(0, str(reference))
    import scripts
    scripts.__path__ = [str(reference/'scripts')]
    from src import exchange_event_overlay, production_config
    assert Path(exchange_event_overlay.__file__).is_relative_to(reference)
    assert Path(production_config.__file__).is_relative_to(reference)
    print(f'参照コード: {exchange_event_overlay.__file__}', flush=True)
    runpy.run_module('scripts.replay_exchange_event_20260926', run_name='__main__')


if __name__ == '__main__':
    main()
