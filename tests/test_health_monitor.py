"""health_monitor の単体テスト (= stateless 集計関数)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.health_monitor import (
    health_report,
    is_stale,
    load_status,
    summarize,
)


def _write_jsonl(p: Path, events: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def test_load_status_empty(tmp_path: Path) -> None:
    assert load_status(tmp_path) == []


def test_load_status_skip_invalid(tmp_path: Path) -> None:
    p = tmp_path / "_status.jsonl"
    p.write_text('{"event":"start","ts":100}\n\nINVALID\n{"step":"x","event":"ok","rc":0}\n')
    events = load_status(tmp_path)
    assert len(events) == 2
    assert events[0]["event"] == "start"


def test_summarize_progress(tmp_path: Path) -> None:
    events = [
        {"event": "start", "ts": 100.0},
        {"step": "cut", "event": "ok", "rc": 0, "dur": 10, "ts": 110.0},
        {"step": "seed", "event": "ok", "rc": 0, "dur": 20, "ts": 130.0},
    ]
    s = summarize(events, expected_steps=4)
    assert s["step_done"] == 2
    assert s["step_fail"] == 0
    assert s["progress"] == 0.5
    assert s["done"] is False


def test_summarize_with_done(tmp_path: Path) -> None:
    events = [
        {"event": "start", "ts": 100.0},
        {"step": "cut", "event": "ok", "rc": 0, "dur": 10, "ts": 110.0},
        {"event": "done", "total_rc": 0, "ts": 120.0, "elapsed": 20},
    ]
    s = summarize(events, expected_steps=1)
    assert s["done"] is True
    assert s["total_rc"] == 0
    assert s["elapsed_sec"] == 20


def test_summarize_failure(tmp_path: Path) -> None:
    events = [
        {"event": "start", "ts": 100.0},
        {"step": "cut", "event": "fail", "rc": 1, "dur": 5, "ts": 105.0},
    ]
    s = summarize(events, expected_steps=2)
    assert s["step_fail"] == 1
    assert s["step_done"] == 0


def test_summarize_items(tmp_path: Path) -> None:
    events = [
        {"event": "start", "ts": 100.0},
        {"step": "viz", "item": "v89m3", "rc": 0, "dur": 60, "ts": 160.0},
        {"step": "viz", "item": "v97m11", "rc": 1, "dur": 30, "ts": 190.0},
        {"step": "viz", "item": "v70m2", "rc": 0, "dur": 60, "ts": 250.0},
    ]
    s = summarize(events, expected_steps=1)
    assert s["item_ok"] == 2
    assert s["item_ng"] == 1


def test_health_report_wrapper(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "_status.jsonl", [
        {"event": "start", "ts": 100.0},
        {"step": "a", "event": "ok", "rc": 0, "dur": 5, "ts": 105.0},
    ])
    r = health_report(tmp_path, expected_steps=2)
    assert r["step_done"] == 1
    assert r["progress"] == 0.5


def test_is_stale_no_events(tmp_path: Path) -> None:
    assert is_stale(tmp_path) is False


def test_is_stale_after_done(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "_status.jsonl", [
        {"event": "start", "ts": 100.0},
        {"event": "done", "total_rc": 0, "ts": 200.0, "elapsed": 100},
    ])
    assert is_stale(tmp_path, max_idle_sec=1) is False


def test_is_stale_active(tmp_path: Path) -> None:
    import time
    now = time.time()
    _write_jsonl(tmp_path / "_status.jsonl", [
        {"event": "start", "ts": now - 5000},
        {"step": "x", "event": "ok", "rc": 0, "dur": 10, "ts": now - 3000},
    ])
    assert is_stale(tmp_path, max_idle_sec=1800) is True
    assert is_stale(tmp_path, max_idle_sec=5000) is False
