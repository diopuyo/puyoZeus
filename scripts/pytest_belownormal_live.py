"""pytestをBelowNormalで実行する (同梱Python用)。spawnした子が__main__を再実行して再帰起動しないよう守る。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import ctypes
from pathlib import Path
import sys

BELOW_NORMAL = 0x4000
SYSTEM_SITE = _public_dev_text('<home>\\AppData\\Local\\Python\\pythoncore-3.14-64\\Lib\\site-packages')  # pytest本体の置き場

if __name__ == '__main__':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    sys.path.append(SYSTEM_SITE)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pytest
    sys.exit(pytest.main(sys.argv[1:]))
