"""E8のテスト・OFF互換・最終レンダと再生／E7一致を保存する。"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts import run_e3_exchange_eval_20260926 as e3
from scripts import run_e4_exchange_eval_20260926 as e4
from scripts import run_e5_exchange_eval_20260926 as e5

OUT = e3.ROOT / "logs/e8"
PREVIOUS = e3.ROOT / "logs/e7"
RELATED = (
    "tests/test_exchange_event_*.py", "tests/test_e3_exchange_metrics.py",
    "tests/test_e4_exchange_*.py", "tests/test_e6_exchange_end.py",
    "tests/test_e6b_exchange_score.py", "tests/test_e8_exchange_finish.py",
)


def read(path: Path) -> dict | list:
    """保存済みの比較対象を読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def run_tests(related: bool) -> None:
    """全体テストはE2と同じ選択条件で、一つのpytestプロセスだけ実行する。"""
    name = "related_tests" if related else "full_tests"
    paths = sorted({str(p.relative_to(e3.ROOT)) for pattern in RELATED
                    for p in e3.ROOT.glob(pattern)}) if related else ["tests/"]
    args = [sys.executable, "-m", "pytest", *paths, "-q", "-p", "no:xdist"]
    if not related:
        args += ["-k", "not test_chain_commit_candidate_v1"]
    with (OUT / f"{name}.log").open("w") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
    (OUT / f"{name}.exit").write_text(str(result.returncode))
    if related:
        if result.returncode:
            raise RuntimeError("関連テスト失敗")
        return
    before = read(e3.ROOT / "logs/e2c/validation_summary.json")["before"]
    log = (OUT / f"{name}.log").read_text()
    after = {status: sorted({line[len(status) + 1:].split(" - ", 1)[0]
             for line in log.splitlines() if line.startswith(status + " tests/")})
             for status in ("FAILED", "ERROR")}
    changes = {status: dict(added=sorted(set(after[status]) - set(before[status])),
                           removed=sorted(set(before[status]) - set(after[status])))
               for status in after}
    counts = {key: int(value) for value, key in re.findall(
        r"(\d+) (failed|passed|skipped|deselected|errors)", log.splitlines()[-1])}
    e3.save_json(OUT / "test_id_diff.json", dict(counts=counts, diff=changes, after=after,
                 baseline_counts=before["counts"], summary=log.splitlines()[-1]))
    if "passed" not in counts or any(row["added"] for row in changes.values()):
        raise RuntimeError("全体テストの失敗／エラーIDが増加")


def verify_replay() -> None:
    """今回の記録を計装付きで再生し、全バイトとE7の全物差しを確認する。"""
    from scripts import e6b_exchange_reach_20260926 as reach
    from scripts import report_e7_exchange_20260927 as metrics
    from scripts.replay_exchange_event_20260926 import compare, replay
    from src.exchange_event_tracker import ExchangeEventTracker
    reach.instrument(ExchangeEventTracker)
    all_rows, videos, checks = [], [], []
    for source in e3.SOURCES:
        reach.STATS, _ = defaultdict(reach.empty_stats), reach.FRAMES.clear()
        rendered = OUT / "renders" / source / "on"
        replayed = OUT / "replay/renders" / source / "on"
        status = replay(rendered / "inputs.jsonl.gz", replayed)
        checks.append(dict(source=source, status=status, render_replay=compare(rendered, replayed),
                           e7_equivalence=compare(PREVIOUS / "renders" / source / "on", rendered)))
        rows = reach.event_rows(replayed, source)
        all_rows.extend(rows)
        videos.append(dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows)))
        e3.save_json(OUT / "replay_progress.json", checks)
    result = dict(stage="current", videos=videos, pooled=reach.summary(all_rows))
    assert result == read(PREVIOUS / "reach_summary.json"), "E7到達計測と不一致"
    e3.save_json(OUT / "reach_summary.json", result)
    e3.save_json(OUT / "reach_events.json", all_rows)
    metrics.OUT = OUT
    summaries, rows, events = [], [], []
    for source in e3.SOURCES:
        summary, video_rows, video_events = metrics.summarize(source)
        summaries.append(summary)
        rows.extend(video_rows)
        events.extend(video_events)
    pooled = metrics.pooled(summaries, rows, events)
    pooled["gates"] = metrics.gates(pooled, summaries)
    for name, value in (("summary.json", summaries), ("pooled.json", pooled)):
        clean = metrics.e3.finite_json(value)
        assert clean == read(PREVIOUS / "v2" / name), f"E7 v2 {name} 不一致"
        e3.save_json(OUT / "v2" / name, clean)
    for source in e3.SOURCES:
        assert read(OUT / "v2" / source / "events.json") == read(
            PREVIOUS / "v2" / source / "events.json"), "M2全件台帳不一致"
    e3.save_json(OUT / "verification.json", dict(videos=checks, all_v2_metrics_identical=True,
                 reach_identical=True, m2_event_rows_identical=True))


def render_all() -> None:
    """本番フラグを読む既存ランナーへ記録を追加し、最大3並列で描画する。"""
    e4.OUT = OUT
    e4.check_off()
    e3.OUT, e3.command = OUT, e5.record_command
    with ThreadPoolExecutor(max_workers=e3.WORKERS) as pool:
        results = list(pool.map(lambda source: e3.run_one(source, "on"), e3.SOURCES))
    e3.save_json(OUT / "render_summary.json", results)
    if any(row["state"] != "completed" for row in results):
        raise RuntimeError("最終レンダ失敗")
    verify_replay()


def main() -> None:
    """切断後も継続する起動元から、各工程の完了状態を残す。"""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(max(0, 19 - os.nice(0)))
    e3.save_json(OUT / "runner.json", dict(pid=os.getpid(), nice=os.nice(0)))
    run_tests(True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tests = pool.submit(run_tests, False)
        render_all()
        tests.result()
    e3.save_json(OUT / "validation_complete.json", dict(completed=True))


if __name__ == "__main__":
    main()
