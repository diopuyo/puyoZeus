"""修正D(2026-07-24): 機能D 疑似発火 ChainSimulator 検証 の before/after 比較 viz。

完全 read-only 診断。src/ の挙動は変更しない
(既定 enable_chain_formula_simulate_verify=False で呼び出し、比較対象は
True のみ)。

対象: recognition_physics_review.TARGET_WINDOWS と同じ c62/c82/c11 窓。
各動画について検証 OFF (既定) / ON の 2 回 RecognitionPipeline を回し、
新規 chain trigger のタイムラインを並べて可視化する。
「どの疑似イベント (偽) が消えたか」「本物の連鎖 (真) が保持されているか」
を一目で確認できるようにする (feedback_recognition_regression_prevention:
過剰抑制で正当な早期発火を巻き込んでいないかの安全確認が最重要)。

Usage:
    PYTHONPATH=. ./venv/bin/python \
        scripts/_viz_chain_formula_simulate_verify_before_after_2026-07-24.py
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

from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

OUTPUT_DIR: Path = (
    PROJ_ROOT / "data" / "verify" / "chain_formula_simulate_verify_before_after_2026-07-24"
)

# recognition_physics_review.py / _diag_false_event_source_2026-07-24.py と
# 揃えた対象窓 (全3窓、真因診断の再現性のため)。
TARGET_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("c62", 895.0, 65.0),
    ("c82", 960.0, 50.0),
    ("c11", 585.0, 60.0),
)


def _collect_triggers(
    video_stem: str, start_sec: float, max_sec: float, simulate_verify: bool,
) -> list[dict]:
    """1 動画・1 窓を処理し、新規 chain trigger の一覧 (side別) を返す。"""
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
        enable_chain_formula_simulate_verify=simulate_verify,
    )
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(video_stem)
    sim = ChainSimulator()

    triggers: list[dict] = []
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
            source = "early_fire_synthetic" if ce.total_erased == 0 else "chain_tracker"
            try:
                chain_count_resim = sim.simulate(ce.before_board).chain_count
            except Exception:
                chain_count_resim = -1
            triggers.append({
                "side": side, "trigger_sec": trig, "source": source,
                "chain_count_event": int(ce.chain_count),
                "total_erased_event": int(ce.total_erased),
                "chain_count_resimulated": chain_count_resim,
                "is_false": chain_count_resim < 1,
            })
    cap.release()
    return triggers


def _write_timeline_png(
    before: list[dict], after: list[dict], video_stem: str, out_path: Path,
) -> None:
    """before(検証OFF) / after(検証ON) の trigger タイムラインを並べて描く。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    for ax, records, label in (
        (axes[0], before, "before (enable_chain_formula_simulate_verify=False, 既定)"),
        (axes[1], after, "after (enable_chain_formula_simulate_verify=True)"),
    ):
        for r in records:
            color = "red" if r["is_false"] else "green"
            marker = "x" if r["is_false"] else "o"
            y = 1.0 if r["side"] == "1P" else 0.0
            ax.scatter(r["trigger_sec"], y, color=color, marker=marker, s=60)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["2P", "1P"])
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(
            f"{label}  (総{len(records)}件、偽={sum(1 for r in records if r['is_false'])}件)",
        )
        ax.grid(axis="x", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="real (chain_count>=1)"),
        plt.Line2D([0], [0], marker="x", color="red", markersize=10, label="false (chain_count<1)"),
    ]
    axes[0].legend(handles=handles, loc="upper right")
    axes[-1].set_xlabel("time (sec)")
    fig.suptitle(f"{video_stem}: chain trigger before/after 機能D simulate_verify 比較")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summary: dict = {"videos": {}}
    for stem, start_sec, max_sec in TARGET_WINDOWS:
        print(f"[collect] {stem} before (simulate_verify=False)...")
        before = _collect_triggers(stem, start_sec, max_sec, simulate_verify=False)
        print(f"[collect] {stem} after (simulate_verify=True)...")
        after = _collect_triggers(stem, start_sec, max_sec, simulate_verify=True)

        n_before_false = sum(1 for r in before if r["is_false"])
        n_after_false = sum(1 for r in after if r["is_false"])
        n_before_real = len(before) - n_before_false
        n_after_real = len(after) - n_after_false
        all_summary["videos"][stem] = {
            "n_before_total": len(before), "n_before_false": n_before_false,
            "n_before_real": n_before_real,
            "n_after_total": len(after), "n_after_false": n_after_false,
            "n_after_real": n_after_real,
            "real_events_preserved": n_after_real >= n_before_real,
        }
        _write_timeline_png(
            before, after, stem, OUTPUT_DIR / f"timeline_{stem}.png",
        )
        (OUTPUT_DIR / f"triggers_{stem}_before.json").write_text(
            json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (OUTPUT_DIR / f"triggers_{stem}_after.json").write_text(
            json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(
            f"  {stem}: before 総{len(before)}(偽{n_before_false}/実{n_before_real}) "
            f"-> after 総{len(after)}(偽{n_after_false}/実{n_after_real})",
        )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
