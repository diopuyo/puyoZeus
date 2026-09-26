"""E5記録を無改変再生し、S3根拠の撤回を入力時系列へ結び付ける。"""
from __future__ import annotations

from collections import Counter
import ast
import inspect
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e6/diagnosis")
CONTEXT: dict = {}
TRACE: list[dict] = []
ORIGINAL_UPDATE = ExchangeEventOverlay.update
ORIGINAL_END = ExchangeEventTracker.end
ORIGINAL_REVOKE = ExchangeEventTracker._revoke_end
BASELINE_REVISION = "8e48558"


def baseline_source(path: str) -> str:
    """Windows作成worktreeでもWSLからコミット内容だけを読めるようにする。"""
    root = Path(__file__).resolve().parents[1]
    pointer = root / ".git"
    command = ["git"]
    if pointer.is_file():
        directory = pointer.read_text().strip().removeprefix("gitdir: ").replace("\\", "/")
        if sys.platform != "win32" and len(directory) > 2 and directory[1] == ":":
            directory = "/mnt/" + directory[0].lower() + directory[2:]
        command.append("--git-dir=" + directory)
    return subprocess.check_output(command + ["show", f"{BASELINE_REVISION}:{path}"],
                                   cwd=root, text=True, encoding="utf-8")


def baseline_module(path: str, name: str) -> ModuleType:
    """作業ツリーを変更せず、確定済みE5コードを診断専用に読み込む。"""
    source = baseline_source(path)
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, path, "exec"), vars(module))
    return module


def load_baseline() -> None:
    """修正後にも同じE5根因分類を再現できるよう評価境界を固定する。"""
    import scripts.replay_exchange_event_20260926 as replay_module
    import scripts.visualize_advantage_overlay as vao
    global ExchangeEventOverlay, ExchangeEventTracker
    global ORIGINAL_UPDATE, ORIGINAL_END, ORIGINAL_REVOKE
    tracker_module = baseline_module("src/exchange_event_tracker.py", "e6_baseline_tracker")
    overlay_module = baseline_module("src/exchange_event_overlay.py", "e6_baseline_overlay")
    ExchangeEventTracker = tracker_module.ExchangeEventTracker
    overlay_module.ExchangeEventTracker = ExchangeEventTracker
    ExchangeEventOverlay = overlay_module.ExchangeEventOverlay
    replay_module.ExchangeEventOverlay = ExchangeEventOverlay
    ORIGINAL_UPDATE = ExchangeEventOverlay.update
    ORIGINAL_END, ORIGINAL_REVOKE = ExchangeEventTracker.end, ExchangeEventTracker._revoke_end
    source = baseline_source("scripts/visualize_advantage_overlay.py")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == "_ExchangeEventEndSignals")
    namespace = vars(vao).copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), "E5_end_signals", "exec"), namespace)
    vao._ExchangeEventEndSignals = namespace["_ExchangeEventEndSignals"]


def update(self: Any, *args: Any, **kwargs: Any) -> None:
    """当該フレームの観測だけを診断文脈へ複製する。"""
    result, snapshot, _, t_sec, game_idx, totals, scores, visible = args
    CONTEXT.clear()
    CONTEXT.update(t_sec=t_sec, game_idx=game_idx, sides=[])
    for idx, side in enumerate((result.p1, result.p2)):
        event = side.chain_event
        CONTEXT["sides"].append(dict(state=side.state.name, score=side.score,
            displayed_score=scores[idx], formula_visible=visible[idx], formula_total=totals[idx],
            next_pair=side.next_pair, slide=side.next_slide_motion,
            dropped=getattr(snapshot, f"total_dropped_to_p{idx + 1}"),
            event=vars(event) if event else None))
    ORIGINAL_UPDATE(self, *args, **kwargs)


def end(self: Any, side: str, t_sec: float, reason: str) -> None:
    """採用された終了信号に、その瞬間の入力を付ける。"""
    chain = self.latest_chain(side)
    if chain is not None and chain.end_signal_sec is None:
        TRACE.append(dict(kind="end", exchange_id=self.current.exchange_id,
                          chain_id=chain.chain_id, side=side, reason=reason, **CONTEXT))
    ORIGINAL_END(self, side, t_sec, reason)


def revoke(self: Any, chain: Any, t_sec: float) -> None:
    """撤回した呼出経路を記録し、元の状態遷移を保つ。"""
    callers = [frame.function for frame in inspect.stack()[1:5]]
    reason = ("chain_observation" if "_record_chain" in callers else
              "display_score_change" if "observe_score" in callers else "formula_visible")
    TRACE.append(dict(kind="revoke", exchange_id=self.current.exchange_id,
        chain_id=chain.chain_id, side=chain.side, reason=reason,
        end_sec=chain.end_signal_sec, **CONTEXT))
    ORIGINAL_REVOKE(self, chain, t_sec)


def classify(source: str, events: list[dict]) -> list[dict]:
    """M1(i)の全信号を列挙し、撃ち合い数と信号数を区別する。"""
    rows = []
    for event in events:
        s3 = [v["t_sec"] for v in event["values"] if v["source"] == "S3"]
        for chain in event["chains"]:
            for signal in chain["end_signals"]:
                revoked = signal.get("revoked_sec")
                affected = [t for t in s3 if revoked and signal["t_sec"] <= t < revoked]
                if not affected:
                    continue
                matching = [r for r in TRACE if r["exchange_id"] == event["exchange_id"]
                            and r["chain_id"] == chain["chain_id"]]
                start = next(r for r in matching if r["kind"] == "end"
                             and r["t_sec"] == signal["t_sec"])
                stop = next(r for r in matching if r["kind"] == "revoke" and r["t_sec"] == revoked)
                rows.append(dict(source=source, exchange_id=event["exchange_id"],
                    chain_id=chain["chain_id"], side=chain["side"], signal=signal,
                    duration=revoked - signal["t_sec"], s3=affected, start=start, stop=stop))
    return rows


def main() -> None:
    """E5出力の完全一致を確認してから分類を保存する。"""
    import json
    load_baseline()
    ExchangeEventOverlay.update = update
    ExchangeEventTracker.end = end
    ExchangeEventTracker._revoke_end = revoke
    rows = []
    for source in SOURCES:
        TRACE.clear()
        baseline = Path("logs/e5/renders") / source / "on"
        output = OUT / source
        replay(baseline / "inputs.jsonl.gz", output)
        compare(baseline, output)
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        rows.extend(classify(source, events))
        save_json(output / "trace.json", TRACE)
        print(source, "診断再生一致", flush=True)
    save_json(OUT / "cases.json", rows)
    counts = Counter((r["signal"]["reason"], r["stop"]["reason"]) for r in rows)
    save_json(OUT / "classification.json", {str(k): v for k, v in counts.items()})
    print(counts, flush=True)


if __name__ == "__main__":
    main()
