"""既存資産を変更せず、指定ソースだけ差し替える開発用の比較起動口。"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

TRACE_INTERVAL_SEC = 120


class SourceOverlay(importlib.abc.MetaPathFinder):
    """spawn子でも同じ明示モジュールを読み込む。配布起動には使用しない。"""

    def __init__(self, root: Path, names: list[str]) -> None:
        self.files = {name.removesuffix('.py').replace('/', '.'): root/name for name in names}

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        source = self.files.get(fullname)
        return importlib.util.spec_from_file_location(fullname, source) if source else None


# Windows spawnによる再実行でも、評価所有者と親が同じソースを使う。
if 'PUYO_OBS_SOURCE_ROOT' in os.environ:
    sys.path.insert(0, os.environ['PUYO_OBS_ASSET_APP'])
    sys.meta_path.insert(0, SourceOverlay(
        Path(os.environ['PUYO_OBS_SOURCE_ROOT']),
        json.loads(os.environ['PUYO_OBS_SOURCE_MODULES'])))

if 'PUYO_OBS_TRACE_DIRECTORY' in os.environ:
    import faulthandler
    print(f'STARTUP_TRACE pid={os.getpid()} parent={os.getppid()}', file=sys.stderr, flush=True)
    # 子はOS pipeへ書き、測定器のwriterがD:へまとめる。各stack行でSMBを待たない。
    faulthandler.dump_traceback_later(TRACE_INTERVAL_SEC, repeat=True, file=sys.stderr)


if __name__ == '__main__':
    from src.phase_j.launcher import main
    main()
