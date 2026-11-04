"""通常の同梱Pythonには組み込まない、署名付き入口専用の処理。"""
import os
from pathlib import Path
import sys

if Path(sys.executable).name in ('puyoZeus Live.exe', 'アンインストール.exe'):
    from signed_launcher import main
    os._exit(main(Path(sys.executable).parent, sys.argv[1:],
                  Path(sys.executable).name == 'アンインストール.exe'))
