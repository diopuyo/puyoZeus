"""同じ標準ZIPへの起動補助を有効にし、保存原票の数値同値を再確認する。"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import verify_live_obs_20261020d as previous

original_run = previous.previous.guarded_run
BOOTSTRAP = (
    "import sys,runpy,zipimport;"
    f"sys.path[:0]=[{str(ROOT)!r},{str(ROOT/'packaging')!r}];"
    "import obs_stdlib_cache;"
    "assert getattr(zipimport._io,'_puyo_stdlib_cache',False);"
    "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
)


def guarded_run(command: list[str], environment: dict, log: object) -> int:
    """検収用C:既存Pythonにも配布と同じhookを適用し、子ごとに確認する。"""
    environment['PUYO_OBS_STDLIB_CACHE'] = '1'
    wrapped = [command[0], '-B', '-c', BOOTSTRAP, *command[1:]]
    return original_run(wrapped, environment, log)


if __name__ == '__main__':
    os.environ['PUYO_OBS_STDLIB_CACHE'] = '1'
    previous.previous.guarded_run = guarded_run
    previous.main()
