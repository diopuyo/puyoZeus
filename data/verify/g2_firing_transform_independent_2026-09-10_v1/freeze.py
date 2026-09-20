"""所有する限定反例/検収票を排他固定する。"""
from __future__ import annotations
from pathlib import Path
import run_cpu as R

ROOT = Path(__file__).resolve().parent


def main() -> int:
    R.guards()
    R.style()
    files = {str(path.relative_to(ROOT)): R.sha(path) for path in sorted(ROOT.rglob('*'))
        if path.is_file() and path.name != 'FREEZE.json'}
    R.write(ROOT/'FREEZE.json', dict(files=files, count=len(files), full=False,
        unclosed=['factory_journal_controller_identity']))
    print(len(files), R.sha(ROOT/'FREEZE.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
