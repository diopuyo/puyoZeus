"""修正D 効果測定(2026-07-24): _diag_false_event_source_2026-07-24.py の
enable_chain_formula_simulate_verify=True 構成版 (before/after 比較用)。

完全 read-only 診断。src/ は一切変更しない (config フラグを True にして
呼ぶだけ)。ロジックは _diag_false_event_source_2026-07-24.py と完全同一、
唯一の違いは RecognitionPipeline.load_default(...) に
enable_chain_formula_simulate_verify=True を渡す点のみ。

before (flag=False, 既定) の結果は
data/verify/diag_false_event_source_2026-07-24/summary.json に保存済み
(n_total=95, n_false=35, early_fire_synthetic: n_total=77, n_false=35)。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_false_event_source_2026-07-24_after.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

OUTPUT_DIR: Path = (
    PROJ_ROOT / "data" / "verify" / "diag_false_event_source_2026-07-24_after"
)

# before と同じ対象窓 (再現性のため揃える)。
TARGET_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("c62", 895.0, 65.0),
    ("c82", 960.0, 50.0),
    ("c11", 585.0, 60.0),
)


def _collect(video_stem: str, start_sec: float, max_sec: float) -> list[dict]:
    """1 動画・1 窓を処理し、新規 chain trigger 毎に発生源分類レコードを返す。"""
    video_path = PROJ_ROOT / "data" / "frames" / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}", file=sys.stderr)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        # 修正D: 機能D 疑似発火の起点盤面を ChainSimulator で事前検証する
        # (唯一 before との差分)。
        enable_chain_formula_simulate_verify=True,
    )
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(video_stem)
    sim = ChainSimulator()

    records: list[dict] = []
    last_trigger = {"1P": None, "2P": None}
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        for side, side_result in (("1P", result.p1), ("2P", result.p2)):
            ce = side_result.chain_event
            if ce is None:
                last_trigger[side] = None
                continue
            trig = float(ce.trigger_sec)
            if trig == last_trigger[side]:
                continue
            last_trigger[side] = trig
            # 発生源分類: total_erased==0 は機能B/D 疑似発火の目印
            # (VideoChainTracker 由来なら4連結消去の定義上 total_erased>=4)。
            source = "early_fire_synthetic" if ce.total_erased == 0 else "chain_tracker"
            try:
                sim_result = sim.simulate(ce.before_board)
                chain_count_resim = sim_result.chain_count
            except Exception:
                chain_count_resim = -1
            records.append({
                "video": video_stem, "side": side, "trigger_sec": trig,
                "source": source, "chain_count_event": int(ce.chain_count),
                "total_erased_event": int(ce.total_erased),
                "chain_count_resimulated": chain_count_resim,
                "is_false": chain_count_resim < 1,
            })
    cap.release()
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for stem, start_sec, max_sec in TARGET_WINDOWS:
        print(f"[collect] {stem} start={start_sec} max_sec={max_sec}")
        recs = _collect(stem, start_sec, max_sec)
        all_records.extend(recs)
        print(f"  -> {len(recs)} 件のトリガー")

    n_total = len(all_records)
    n_false = sum(1 for r in all_records if r["is_false"])
    by_source: dict[str, dict[str, int]] = {}
    for r in all_records:
        s = r["source"]
        by_source.setdefault(s, {"n_total": 0, "n_false": 0})
        by_source[s]["n_total"] += 1
        if r["is_false"]:
            by_source[s]["n_false"] += 1

    summary = {
        "n_total_triggers": n_total,
        "n_false_triggers": n_false,
        "false_rate": (n_false / n_total) if n_total else None,
        "by_source": by_source,
    }
    (OUTPUT_DIR / "records.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
