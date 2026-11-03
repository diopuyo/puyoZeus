#!/usr/bin/python3
"""方式F検証用。std の巻き戻しを Zig の libunwind へリンクする。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[1])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import os
import sys

SKIP = frozenset(('-lgcc_eh', '-l:libpthread.a', '-lgcc', '-lmsvcrt', '-lmingwex', '-lmingw32'))


def main() -> None:
    args = [a[4:] if a.startswith('-Wl,') and a.endswith('.def') else a
            for a in sys.argv[1:] if a not in SKIP]
    # Zigは-Bdynamicの-lpython312をDLLに限定するためimport libraryを明示する。
    paths = [args[i+1] for i,a in enumerate(args[:-1]) if a == '-L']
    for i, arg in enumerate(args):
        if arg.startswith('-lpython'):
            name = arg[2:] + '.lib'
            args[i] = next((os.path.join(p,name) for p in paths
                            if os.path.isfile(os.path.join(p,name))), arg)
    python = os.environ.get('AMA_ZIG_PYTHON', _public_dev_text('<home>/zigvenv/bin/python'))
    os.execv(python, [python, '-m', 'ziglang', 'c++', '-target', 'x86_64-windows-gnu',
                     '-lc', '-lunwind', '-w', *args])


if __name__ == '__main__':
    main()
