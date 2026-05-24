"""logs/<phase>/_status.jsonl を集計して progress + ETA を返す stateless 関数群.

ヘルスチェック framework の Python 集計層。 bash 側 (= _lib_health.sh) が
jsonl で書き出した event を読み取り、 progress / ETA / failure を集計する。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def load_status(log_root: Path) -> list[dict[str, Any]]:
    """_status.jsonl を読み込み event list を返す。 不在なら空 list."""
    p = log_root / "_status.jsonl"
    if not p.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def summarize(events: list[dict[str, Any]], expected_steps: int) -> dict[str, Any]:
    """event list から progress / ETA / failure を集計."""
    step_done = [e for e in events if e.get("event") == "ok"]
    step_fail = [e for e in events if e.get("event") == "fail"]
    items_ok = [e for e in events if "item" in e and e.get("rc") == 0]
    items_ng = [e for e in events if "item" in e and e.get("rc") != 0]
    start_evt = next((e for e in events if e.get("event") == "start"), None)
    done_evt = next((e for e in events if e.get("event") == "done"), None)
    start_ts = float(start_evt["ts"]) if start_evt else time.time()
    elapsed = time.time() - start_ts
    if done_evt is not None:
        elapsed = float(done_evt.get("elapsed", elapsed))
    progress = len(step_done) / max(expected_steps, 1)
    eta_sec: float | None = None
    if 0 < progress < 1.0:
        eta_sec = elapsed * (1 - progress) / progress
    return {
        "progress": round(progress, 4),
        "elapsed_sec": round(elapsed, 1),
        "eta_sec": round(eta_sec, 1) if eta_sec is not None else None,
        "step_done": len(step_done),
        "step_fail": len(step_fail),
        "step_expected": expected_steps,
        "item_ok": len(items_ok),
        "item_ng": len(items_ng),
        "done": done_evt is not None,
        "total_rc": int(done_evt.get("total_rc", -1)) if done_evt else None,
    }


def health_report(log_root: Path, expected_steps: int) -> dict[str, Any]:
    """1 行 wrapper: load + summarize."""
    events = load_status(log_root)
    return summarize(events, expected_steps)


def is_stale(log_root: Path, max_idle_sec: int = 1800) -> bool:
    """直近 event から max_idle_sec 経過していれば stale (= ハング疑い).

    Phase L master script のような自律実行で進捗が止まったことを検出する。
    """
    events = load_status(log_root)
    if not events:
        return False
    last_ts = max(float(e.get("ts", 0)) for e in events)
    done_evt = next((e for e in events if e.get("event") == "done"), None)
    if done_evt is not None:
        return False  # 完了済みは stale でない
    return (time.time() - last_ts) > max_idle_sec
