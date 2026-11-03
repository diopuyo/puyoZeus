"""段1参考比較の完了後に段2を実行し、失敗時は理由を保存して停止する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
import traceback

from scripts.gfe_phase_stage2_replay_20261002 import OUT, SOURCES
from scripts.gfe_phase_stage2_train_20261002 import model_directory
from scripts.gfe_phase_stage1_20261002 import ROOT, save_json, memory_guard

WORKERS = 3
POLL_SECONDS = 15


def command(module: str, args: tuple[str, ...], log: str) -> dict:
    """実行内容を配列で渡し、完了コードと実時間を保存する。"""
    memory_guard()
    started = time.monotonic()
    with (OUT/log).open("w") as stream:
        subprocess.run([sys.executable, "-B", "-m", module, *args], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(module=module, args=args, seconds=time.monotonic()-started)


def replay(job: tuple[str, str]) -> dict:
    """記録単位で並列化し、完了票の無いものだけを実行する。"""
    mode, source = job
    done = OUT/mode/"replay"/source/"DONE.json"
    if done.exists():
        assert json.loads(done.read_text())["state"] == "completed"
        return dict(args=job, reused=True)
    result = command("scripts.gfe_phase_stage2_replay_20261002", job, f"{mode}_{source}.log")
    print(json.dumps(result), flush=True)
    return result


def execute() -> None:
    """Bの結果が完了してから最終学習→既定OFF照合→全再生→採点へ進む。"""
    prerequisite = ROOT/"logs/gfe_phase_reference_20261002/METRICS.json"
    while not prerequisite.exists():
        time.sleep(POLL_SECONDS)
    assert json.loads(prerequisite.read_text())["status"] == "COMPLETE"
    assert json.loads((ROOT/"logs/gfe_phase_reference_20261002/LEAK_AUDIT.json").read_text())["passed"]
    started = time.monotonic()
    trained = [model_directory(m)/"manifest.json" for m in ("V3", "V1b")]
    timing = []
    if not all(p.exists() for p in trained):
        timing.append(command("scripts.gfe_phase_stage2_train_20261002", (), "train.log"))
    timing.append(replay(("baseline", "review")))
    jobs = [(m, s) for s in SOURCES for m in ("baseline", "V3", "V1b") if (m, s) != ("baseline", "review")]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(replay, job) for job in jobs]
        for future in as_completed(pending):
            try:
                timing.append(future.result())
            except Exception:
                for other in pending:
                    other.cancel()
                raise
    timing.append(command("scripts.gfe_phase_stage2_score_20261002", (), "score.log"))
    save_json(OUT/"TIMING.json", dict(seconds=time.monotonic()-started, jobs=timing, workers=WORKERS))


def main() -> None:
    """停止理由を消さずに残す。構成変更や門の緩和はしない。"""
    os.nice(max(0, 19-os.nice(0)))
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        execute()
    except Exception as error:
        save_json(OUT/"STOP.json", dict(status="STOP", error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
