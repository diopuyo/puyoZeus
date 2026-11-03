"""部分再生を含む入力取得後に検証・全表生成を自動で完了させる。"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from scripts.gfe_phase_stage2_replay_20261002 import SET1, SET2
from scripts._diag_gfe_stage2_analyze import OUT

POLL_SECONDS = 10
MODULES = ('landing','verify','probability','report')


def main() -> None:
    """セッションが途切れても、許可された診断ディレクトリ内で完結する。"""
    os.nice(max(0,19-os.nice(0)))
    while not (all((OUT/'capture'/source/'DONE.json').exists() for source in (*SET1,*SET2))
               and (OUT/'capture_V3/c1/DONE.json').exists()):
        memory_guard()
        if 'Traceback' in (OUT/'capture_all.log').read_text():
            raise RuntimeError('入力取得が異常終了したため停止')
        time.sleep(POLL_SECONDS)
    for suffix in MODULES:
        memory_guard()
        with (OUT/f'final_{suffix}.log').open('w') as stream:
            subprocess.run([sys.executable, '-B', '-m', 'scripts._diag_gfe_stage2_'+suffix],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    save_json(OUT/'COMPLETE.json', dict(status='diagnostic_complete', modules=list(MODULES)))
    print('診断の全表生成完了',flush=True)


if __name__ == '__main__':
    main()
