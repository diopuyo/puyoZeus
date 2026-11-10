"""既存ama_zig_linkと同じGNUライブラリの置換をWindowsでも行う。"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

SKIP = frozenset(('-lgcc_eh', '-l:libpthread.a', '-lgcc', '-lmsvcrt', '-lmingwex', '-lmingw32'))


def link_arguments(arguments: list[str]) -> list[str]:
    args = [arg[4:] if arg.startswith('-Wl,') and arg.endswith('.def') else arg
            for arg in arguments if arg not in SKIP]
    paths = [Path(args[i + 1]) for i, arg in enumerate(args[:-1]) if arg == '-L']
    paths.extend(Path(arg[2:]) for arg in args if arg.startswith('-L') and len(arg) > 2)
    for index, arg in enumerate(args):
        if arg.startswith('-lpython'):
            name = arg[2:] + '.lib'
            args[index] = next((str(path / name) for path in paths if (path / name).is_file()), arg)
    return args


if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, '-m', 'ziglang', 'c++', '-target',
                                     'x86_64-windows-gnu', '-lc', '-lunwind', '-w',
                                     *link_arguments(sys.argv[1:])]))
