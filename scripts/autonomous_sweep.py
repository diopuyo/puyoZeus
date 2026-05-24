"""自律品質向上 sweep runner (= 2026-05-21 深夜 → 朝).

学習軸凍結中の推論側 parameter sweep を Python で完全自動化。 bash sub-shell
の quote 罠を排除し、 朝まで複数施策を順次評価。

評価軸: baseline_videos_v3 8 動画 + 強化アナリスト critical 集計
判定: baseline 比改善 ≥ 5 → ACCEPT, ≤ -5 → REJECT, 中間 → NEUTRAL
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import os as _os
# sweep 用に重要 4 動画に絞る (= 朝までの時間制約)。 baseline 集計は別 8 動画で実施済
SWEEP_VIDEOS = _os.environ.get(
    "SWEEP_VIDEOS",
    "v89m3,v97m11,v70m2,v95m15",  # cycle 46 で問題が起きやすかった代表 4
).split(",")
VIDEOS = SWEEP_VIDEOS  # autonomous_sweep の評価対象
DEFAULT_MODEL = "models/cnn_phase_b_large_v2.pt"
HSV_STATE = "data/per_video_hsv_ranges/_merged_default.json"
BASE_DIR = Path("data/baseline_videos_v3")
JUDGMENTS_FILE = Path("data/verify/autonomous/_judgments.jsonl")


@dataclass
class SweepStep:
    name: str
    file_path: str
    pattern: str         # regex で current 値を catch (1 group)
    new_value: str       # 置換新値 (= str)
    desc: str = ""


@dataclass
class Substitution:
    """1 file の正規表現置換 (apply / revert 可)."""
    file_path: Path
    original_text: str
    new_text: str
    pattern: str

    def apply(self) -> None:
        text = self.file_path.read_text(encoding="utf-8")
        text2 = re.sub(self.pattern, self.new_text, text, count=1)
        if text == text2:
            raise RuntimeError(f"pattern {self.pattern} did not match in {self.file_path}")
        self.file_path.write_text(text2, encoding="utf-8")

    def revert(self) -> None:
        self.file_path.write_text(self.original_text, encoding="utf-8")


def make_substitution(file_path: str, pattern: str, new_text: str) -> Substitution:
    p = Path(file_path)
    return Substitution(
        file_path=p, original_text=p.read_text(encoding="utf-8"),
        new_text=new_text, pattern=pattern,
    )


def eval_one_video(cycle_name: str, vkey: str, log_dir: Path, out_dir: Path) -> int | None:
    input_path = BASE_DIR / f"{vkey}_buf15s.mp4"
    if not input_path.exists():
        return None
    board_log = log_dir / f"viz_{vkey}.jsonl"
    report = out_dir / f"{vkey}.json"
    if report.exists():
        return _get_critical(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    viz_cmd = [
        "./venv/bin/python", "-m", "scripts.visualize_recognition",
        "--video", str(input_path),
        "--output", str(out_dir / f"{vkey}.mp4"),
        "--cnn-model", DEFAULT_MODEL,
        "--hsv-state", HSV_STATE,
        "--dump-board-log", str(board_log),
    ]
    with (log_dir / f"viz_{vkey}.log").open("w") as f:
        subprocess.run(viz_cmd, stdout=f, stderr=subprocess.STDOUT, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})
    if not board_log.exists():
        return None
    eval_cmd = [
        "./venv/bin/python", "-m", "scripts.evaluate_recognition",
        "--board-log", str(board_log),
        "--report-out", str(report),
    ]
    with (log_dir / f"eval_{vkey}.log").open("w") as f:
        subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})
    return _get_critical(report)


def _get_critical(report_path: Path) -> int | None:
    if not report_path.exists():
        return None
    try:
        d = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return int(d.get("summary", {}).get("critical", 0))


def _eval_one_viz_only(cycle_name: str, vkey: str, log_dir: Path, out_dir: Path) -> None:
    """viz だけ実行 (= 並列実行用)、 eval は別 phase で."""
    input_path = BASE_DIR / f"{vkey}_buf15s.mp4"
    if not input_path.exists():
        return
    board_log = log_dir / f"viz_{vkey}.jsonl"
    if board_log.exists() and board_log.stat().st_size > 100:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    viz_cmd = [
        "./venv/bin/python", "-m", "scripts.visualize_recognition",
        "--video", str(input_path),
        "--output", str(out_dir / f"{vkey}.mp4"),
        "--cnn-model", DEFAULT_MODEL,
        "--hsv-state", HSV_STATE,
        "--dump-board-log", str(board_log),
    ]
    with (log_dir / f"viz_{vkey}.log").open("w") as f:
        subprocess.run(viz_cmd, stdout=f, stderr=subprocess.STDOUT, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})


def _eval_one_eval_only(cycle_name: str, vkey: str, log_dir: Path, out_dir: Path) -> int | None:
    """eval だけ実行."""
    board_log = log_dir / f"viz_{vkey}.jsonl"
    report = out_dir / f"{vkey}.json"
    if report.exists():
        return _get_critical(report)
    if not board_log.exists():
        return None
    eval_cmd = [
        "./venv/bin/python", "-m", "scripts.evaluate_recognition",
        "--board-log", str(board_log),
        "--report-out", str(report),
    ]
    with (log_dir / f"eval_{vkey}.log").open("w") as f:
        subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})
    return _get_critical(report)


def evaluate_setup(cycle_name: str) -> dict:
    """N 動画 viz + 評価、 critical 集計を返す (= 並列 2 で viz 高速化)."""
    from concurrent.futures import ProcessPoolExecutor
    log_dir = Path(f"logs/autonomous/{cycle_name}")
    out_dir = Path(f"data/verify/autonomous/{cycle_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # viz 並列 2 (= GPU 競合下限で安全)
    with ProcessPoolExecutor(max_workers=2) as ex:
        list(ex.map(
            _eval_one_viz_only,
            [cycle_name] * len(VIDEOS),
            VIDEOS,
            [log_dir] * len(VIDEOS),
            [out_dir] * len(VIDEOS),
        ))
    # eval 並列 4 (= CPU only、 軽い)
    total = 0
    per = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(
            _eval_one_eval_only,
            [cycle_name] * len(VIDEOS),
            VIDEOS,
            [log_dir] * len(VIDEOS),
            [out_dir] * len(VIDEOS),
        ))
    for v, c in zip(VIDEOS, results):
        if c is None:
            continue
        per[v] = c
        total += c
    summary = {"cycle": cycle_name, "total_critical": total, "per_video": per, "ts": int(time.time())}
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def judge(cycle_name: str, before: int, after: int, threshold: int = 5) -> str:
    diff = after - before
    if diff <= -threshold:
        return "ACCEPT"
    if diff >= threshold:
        return "REJECT"
    return "NEUTRAL"


def log_judgment(record: dict) -> None:
    JUDGMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with JUDGMENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_sweep(steps: list[SweepStep], baseline_critical: int) -> None:
    for step in steps:
        print(f"\n=== {step.name}: {step.desc} ===")
        try:
            sub = make_substitution(step.file_path, step.pattern, step.new_value)
        except FileNotFoundError:
            print(f"[skip] file not found: {step.file_path}")
            continue
        try:
            sub.apply()
            summary = evaluate_setup(step.name)
            after = summary["total_critical"]
            verdict = judge(step.name, baseline_critical, after)
            rec = {
                "step": step.name, "desc": step.desc,
                "file": step.file_path, "pattern": step.pattern, "new_value": step.new_value,
                "before": baseline_critical, "after": after, "diff": after - baseline_critical,
                "verdict": verdict, "ts": int(time.time()),
            }
            log_judgment(rec)
            print(f"[{step.name}] {verdict} before={baseline_critical} after={after} diff={after - baseline_critical}")
        finally:
            sub.revert()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline-json", type=Path,
        default=Path("data/verify/baseline_v3_eval/_summary.json"),
    )
    p.add_argument("--sweep", choices=["all", "A", "B", "C", "D"], default="all")
    args = p.parse_args()

    if not args.baseline_json.exists():
        print(f"[FATAL] baseline not found: {args.baseline_json}")
        return 1
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    # baseline は 8 動画分集計 / sweep は 4 動画 subset → subset で再集計
    per_video = baseline.get("per_video", {})
    baseline_crit = sum(
        per_video.get(v, {}).get("critical", 0) for v in VIDEOS
    )
    print(f"[autonomous] baseline critical (subset {VIDEOS}) = {baseline_crit}")

    sweeps: dict[str, list[SweepStep]] = {
        "A": [  # HybridClassifier cnn_override_prob sweep (= 推論軸最重要)
            SweepStep("A1_cnn_override_050", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.50",
                      "cnn_override 0.70 → 0.50"),
            SweepStep("A1_cnn_override_055", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.55",
                      "cnn_override 0.70 → 0.55"),
            SweepStep("A1_cnn_override_060", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.60",
                      "cnn_override 0.70 → 0.60"),
            SweepStep("A1_cnn_override_065", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.65",
                      "cnn_override 0.70 → 0.65"),
            SweepStep("A1_cnn_override_075", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.75",
                      "cnn_override 0.70 → 0.75"),
            SweepStep("A1_cnn_override_080", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.80",
                      "cnn_override 0.70 → 0.80"),
            SweepStep("A1_cnn_override_085", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.85",
                      "cnn_override 0.70 → 0.85"),
            SweepStep("A1_cnn_override_090", "src/hybrid_classifier.py",
                      r"DEFAULT_CNN_OVERRIDE_PROB[^=]*=\s*[\d.]+", "DEFAULT_CNN_OVERRIDE_PROB: float = 0.90",
                      "cnn_override 0.70 → 0.90"),
        ],
        "C": [  # state machine ChainPhaseDetector tuning
            SweepStep("C1_chain_unknown_1", "src/state_detectors.py",
                      r"unknown_count\s*<\s*\d+", "unknown_count < 1",
                      "chain 4 連結 gate unknown_count 3 → 1 (= 厳格)"),
            SweepStep("C1_chain_unknown_2", "src/state_detectors.py",
                      r"unknown_count\s*<\s*\d+", "unknown_count < 2",
                      "chain 4 連結 gate unknown_count 3 → 2"),
            SweepStep("C1_chain_unknown_5", "src/state_detectors.py",
                      r"unknown_count\s*<\s*\d+", "unknown_count < 5",
                      "chain 4 連結 gate unknown_count 3 → 5"),
        ],
        "D": [  # recognition_evaluator metric threshold sweep
            SweepStep("D1_drop_threshold_1", "src/recognition_evaluator.py",
                      r"PUYO_COUNT_DROP_THRESHOLD_NORMAL[^=]*=\s*\d+", "PUYO_COUNT_DROP_THRESHOLD_NORMAL: int = 1",
                      "metric drop threshold 2 → 1 (= 厳格化)"),
            SweepStep("D1_drop_threshold_3", "src/recognition_evaluator.py",
                      r"PUYO_COUNT_DROP_THRESHOLD_NORMAL[^=]*=\s*\d+", "PUYO_COUNT_DROP_THRESHOLD_NORMAL: int = 3",
                      "metric drop threshold 2 → 3 (= 緩和)"),
            SweepStep("D2_sudden_drop_3", "src/recognition_evaluator.py",
                      r"SUDDEN_DROP_THRESHOLD[^=]*=\s*\d+", "SUDDEN_DROP_THRESHOLD: int = 3",
                      "sudden drop 5 → 3 (= 厳格)"),
            SweepStep("D2_sudden_drop_7", "src/recognition_evaluator.py",
                      r"SUDDEN_DROP_THRESHOLD[^=]*=\s*\d+", "SUDDEN_DROP_THRESHOLD: int = 7",
                      "sudden drop 5 → 7 (= 緩和)"),
            SweepStep("D3_chain_min_3", "src/recognition_evaluator.py",
                      r"CHAIN_STATE_MIN_FRAMES[^=]*=\s*\d+", "CHAIN_STATE_MIN_FRAMES: int = 3",
                      "chain min frames 5 → 3"),
            SweepStep("D3_chain_min_7", "src/recognition_evaluator.py",
                      r"CHAIN_STATE_MIN_FRAMES[^=]*=\s*\d+", "CHAIN_STATE_MIN_FRAMES: int = 7",
                      "chain min frames 5 → 7"),
            SweepStep("D4_chain_loss_2", "src/recognition_evaluator.py",
                      r"CHAIN_MIN_PUYO_LOSS[^=]*=\s*\d+", "CHAIN_MIN_PUYO_LOSS: int = 2",
                      "chain min puyo loss 4 → 2"),
            SweepStep("D4_chain_loss_6", "src/recognition_evaluator.py",
                      r"CHAIN_MIN_PUYO_LOSS[^=]*=\s*\d+", "CHAIN_MIN_PUYO_LOSS: int = 6",
                      "chain min puyo loss 4 → 6"),
        ],
    }

    if args.sweep == "all":
        order = ["A", "C", "D"]
    else:
        order = [args.sweep]

    all_steps = []
    for k in order:
        all_steps.extend(sweeps.get(k, []))
    print(f"[autonomous] {len(all_steps)} sweep steps")
    run_sweep(all_steps, baseline_crit)
    print("[autonomous] sweep complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
