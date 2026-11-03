"""既定OFFの5記録照合後、固定18記録のV3s/R1sを再生する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
import traceback

from scripts.gfe_phase_stage2b_train_20261002 import OUT, R1
from scripts.gfe_phase_stage2_replay_20261002 import SOURCES, EXISTING
from scripts.gfe_phase_stage1_20261002 import ROOT, save_json, memory_guard

WORKERS, POLL_SECONDS = 3, 15


def command(module: str, args: tuple, log: str) -> dict:
    """引数を配列で渡し、全処理を指定worktreeに閉じる。"""
    memory_guard()
    started = time.monotonic()
    with (OUT/log).open('w') as stream:
        subprocess.run([sys.executable, '-B', '-m', module, *args], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(module=module, args=args, seconds=time.monotonic()-started)


def replay(job: tuple) -> dict:
    """監査原票も含めて完了した再生だけを再利用する。"""
    mode, source = job
    directory = OUT/mode/'replay'/source
    if (directory/'GFE_USAGE.json').exists():
        done = json.loads((directory/'DONE.json').read_text())
        assert done['state'] == 'completed'
        return dict(args=job, reused=True, seconds=done['elapsed_seconds'])
    result = command('scripts.gfe_phase_stage2b_replay_20261002', job, f'{mode}_{source}.log')
    print(json.dumps(result), flush=True)
    return result


def batch(jobs: list) -> list:
    """最大3再生に限定し、失敗したら未開始の処理を中止する。"""
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(replay, job) for job in jobs]
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception:
                for other in pending:
                    other.cancel()
                raise
    return results


def main() -> None:
    """段2本番18再生は参照のみ、採点は全候補完了後に行う。"""
    os.nice(max(0, 19-os.nice(0)))
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        while not (OUT/'baseline/replay/review/GFE_USAGE.json').exists():
            if 'Traceback' in (OUT/'baseline_review.log').read_text():
                raise RuntimeError('既定OFF初回再生が失敗')
            time.sleep(POLL_SECONDS)
        timing = batch([('baseline', s) for s in EXISTING])
        while not (R1/'manifest.json').exists():
            if 'Traceback' in (OUT/'train.log').read_text():
                raise RuntimeError('R1学習が失敗')
            time.sleep(POLL_SECONDS)
        timing += batch([(m, s) for s in SOURCES for m in ('V3s', 'R1s')])
        timing.append(command('scripts.gfe_phase_stage2b_score_20261002', (), 'score.log'))
        save_json(OUT/'TIMING.json', dict(seconds=time.monotonic()-started, jobs=timing, workers=WORKERS))
    except Exception as error:
        save_json(OUT/'STOP.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
