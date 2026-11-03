#!/usr/bin/python3
"""no_std単盤面DLL専用。Rust GNUリンク指定をZig 0.16へ渡す。"""
import os
import sys

SKIP = frozenset(('-lgcc_eh', '-l:libpthread.a', '-lgcc', '-lmsvcrt', '-lmingwex', '-lmingw32'))


def main() -> None:
    """例外を跨がないpanic=abortビルドに限定し、Zig提供CRTをリンクする。"""
    args = [arg[4:] if arg.startswith('-Wl,') and arg.endswith('.def') else arg
            for arg in sys.argv[1:] if arg not in SKIP]
    python = os.environ['LIVE_ZIG_PYTHON']
    os.execv(python, [python, '-m', 'ziglang', 'cc', '-target', 'x86_64-windows-gnu', '-lc', *args])


if __name__ == '__main__':
    main()
