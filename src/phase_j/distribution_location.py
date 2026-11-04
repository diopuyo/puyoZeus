"""展開先が同期フォルダ内かを調べ、起動を妨げず注意する。"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Mapping

ONEDRIVE_VARIABLES = ('onedrive', 'onedriveconsumer', 'onedrivecommercial')


def is_onedrive_root(root: Path, environment: Mapping[str, str] | None = None) -> bool:
    """環境変数の境界とフォルダ名を使い、似た名前の別フォルダを除く。"""
    path = PureWindowsPath(str(root.resolve()) if os.name == 'nt' else str(root))
    variables = {key.casefold(): value for key, value in
                 (os.environ if environment is None else environment).items()}
    for key in ONEDRIVE_VARIABLES:
        value = variables.get(key, '').strip()
        sync_root = PureWindowsPath(str(Path(value).resolve()) if os.name == 'nt' else value)
        if value and path.is_relative_to(sync_root):
            return True
    return any(part.casefold() == 'onedrive' or part.casefold().startswith('onedrive - ')
               for part in path.parts)
