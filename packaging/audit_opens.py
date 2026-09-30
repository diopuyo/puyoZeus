"""audit_sitecustomize が記録したログを集計し、app/ 配下で実際に読まれた非 .py 資産を列挙する。

    python packaging/audit_opens.py <ログのベース名> <app ディレクトリ>

出力: 読まれた資産 (app 相対) / app に在るのに読まれなかった資産 / app 外 (リポジトリ等) から
読まれた資産 (= 配布物へ入れ忘れの疑い)。母数 (記録 process 数・総 open 数) も併記する。
"""
from __future__ import annotations

from pathlib import Path
import sys

CODE_SUFFIXES = frozenset({'.py', '.pyc', '.pyd', '.dll', '.so'})


def load_opens(base: Path) -> tuple[set[Path], int]:
    """<base>.<pid> をすべて読み、(開かれたパス集合, process 数) を返す。"""
    opened: set[Path] = set()
    logs = sorted(base.parent.glob(base.name + '.*'))
    for log in logs:
        opened |= {Path(line) for line in log.read_text(encoding='utf-8').splitlines() if line}
    return opened, len(logs)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> None:
    base, app = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    opened, processes = load_opens(base)
    inside = sorted(p for p in opened if is_under(p.resolve(), app) and p.suffix not in CODE_SUFFIXES)
    print(f'# 記録 process {processes} / 総 open {len(opened)} / app 内の非コード資産 {len(inside)}')
    used = {p.resolve() for p in inside}
    for path in inside:
        print('USED   ', path.resolve().relative_to(app).as_posix())
    shipped = sorted(p for p in app.rglob('*') if p.is_file() and p.suffix not in CODE_SUFFIXES)
    for path in shipped:
        if path.resolve() not in used:
            print('UNUSED ', path.relative_to(app).as_posix())
    for path in sorted(opened):
        # open 試行は存在しないパスでも記録される。app 配下で存在しない資産 = 同梱漏れか、
        # 任意ファイルを探して無ければ黙って機能を切る実装 (fail-silent) の疑い。
        if is_under(path.resolve(), app) and not path.exists() and not path.name.endswith(('.pyc', '.pyd')) \
                and '__pycache__' not in path.parts and '.pyc.' not in path.name and path.suffix != '.py':
            print('MISSING', path.resolve().relative_to(app).as_posix())
    for path in sorted(opened):
        if not is_under(path.resolve(), app) and path.suffix not in CODE_SUFFIXES and path.exists():
            text = str(path)
            if 'site-packages' not in text and 'python' + '\\' not in text.lower() and 'Temp' not in text:
                print('OUTSIDE', text)


if __name__ == '__main__':
    main()
