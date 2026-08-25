"""偽連鎖イベント率 A/B (2026-08-24 コーダ、受け入れ条件4)。

_diag_false_event_source_2026-07-24.py (W7 真因計装) と同一の c62 窓で、
旧構成 (off) と根治フラグON (on) の新規 chain trigger を全件記録する。

真偽判定は 2 軸:
  (a) resim: before_board 再 simulate で連鎖ゼロか (2026-07-24 と同じ定義)。
      **注意**: この定義は凍結盤面を物差しにしており、根治②の実読発火は
      「凍結盤面に連鎖が無くても画面に連鎖がある」ケースを救うのが目的
      なので、新機構に対しては構造的に偽陽性を出す (測定器の限界)。
  (b) score支持: trigger 後 SUPPORT_WINDOW 秒以内に自 side score が
      +MIN_CHAIN_SCORE 以上増加したか (独立証拠、こちらを主指標とする)。

Usage:
    python scripts/_ab_false_event_2026-08-24.py off
    python scripts/_ab_false_event_2026-08-24.py on
出力: data/verify/formula_read_false_event_ab_2026-08-24/records_<mode>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import cv2  # noqa: E402

from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "off"
assert MODE in ("off", "on"), MODE

OUTPUT_DIR = PROJ_ROOT / "data/verify/formula_read_false_event_ab_2026-08-24"

# 2026-07-24 計装と同一窓 (c82/c11 はローカル不在のため c62 のみ + 追加窓)。
TARGET_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("c62", 895.0, 65.0),
    ("c62", 585.0, 65.0),
    ("c62", 300.0, 65.0),
)

# score 支持判定: 連鎖の最小得点 = 4個消し×10×max(1,ボーナス) = 40 点。
MIN_CHAIN_SCORE: int = 40
# 連鎖完了から score 表示が読める状態に戻るまでの猶予。連鎖 1 段 ≈1.4 秒
# (reference_chain_formula_per_step_2026-08-22) × 数段 + 表示復帰。
SUPPORT_WINDOW_SEC: float = 8.0

NEW_FLAGS = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
)


def _collect(video_stem: str, start_sec: float, max_sec: float) -> list[dict]:
    video_path = PROJ_ROOT / "data" / "frames" / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}", file=sys.stderr)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)

    kwargs: dict = dict(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if MODE == "on":
        kwargs.update(NEW_FLAGS)
    pipeline = RecognitionPipeline.load_default(**kwargs)
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(video_stem)
    sim = ChainSimulator()

    records: list[dict] = []
    score_timeline: list[tuple[float, str, int]] = []
    last_trigger: dict[str, float | None] = {"1P": None, "2P": None}
    last_scores: dict[str, int | None] = {"1P": None, "2P": None}
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        for side, tracker in (
            ("1P", pipeline._score_tracker_1p),
            ("2P", pipeline._score_tracker_2p),
        ):
            s = tracker.last_score if tracker is not None else None
            if s is not None and s != last_scores[side]:
                score_timeline.append((t_sec, side, int(s)))
                last_scores[side] = s
        for side, side_result in (("1P", result.p1), ("2P", result.p2)):
            ce = side_result.chain_event
            if ce is None:
                last_trigger[side] = None
                continue
            trig = float(ce.trigger_sec)
            if trig == last_trigger[side]:
                continue
            last_trigger[side] = trig
            try:
                resim = sim.simulate(ce.before_board).chain_count
            except Exception:
                resim = -1
            records.append({
                "video": video_stem, "side": side, "trigger_sec": trig,
                "t_seen": t_sec,
                "mechanism": ce.mechanism,
                "chain_count_event": int(ce.chain_count),
                "total_score_event": int(ce.total_score),
                "total_erased_event": int(ce.total_erased),
                "chain_count_resimulated": int(resim),
                "score_at_trigger": last_scores[side],
            })
    cap.release()
    # score 支持判定を後付けする
    for r in records:
        side = r["side"]
        base = r["score_at_trigger"]
        t0 = r["t_seen"]
        supported = None
        if base is not None:
            supported = False
            for t, s_side, s in score_timeline:
                if s_side != side or t < t0:
                    continue
                if t > t0 + SUPPORT_WINDOW_SEC:
                    break
                if s >= base + MIN_CHAIN_SCORE:
                    supported = True
                    break
        r["score_supported"] = supported
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for stem, start_sec, max_sec in TARGET_WINDOWS:
        print(f"[collect] {stem} start={start_sec} max_sec={max_sec}", flush=True)
        recs = _collect(stem, start_sec, max_sec)
        all_records.extend(recs)
        print(f"  -> {len(recs)} 件のトリガー", flush=True)

    n_total = len(all_records)
    n_resim_false = sum(
        1 for r in all_records if r["chain_count_resimulated"] < 1
    )
    n_unsupported = sum(
        1 for r in all_records if r["score_supported"] is False
    )
    by_mech: dict[str, dict[str, int]] = {}
    for r in all_records:
        m = str(r["mechanism"])
        d = by_mech.setdefault(
            m, {"n": 0, "resim_false": 0, "score_unsupported": 0},
        )
        d["n"] += 1
        if r["chain_count_resimulated"] < 1:
            d["resim_false"] += 1
        if r["score_supported"] is False:
            d["score_unsupported"] += 1
    summary = {
        "mode": MODE,
        "n_total_triggers": n_total,
        "n_resim_false": n_resim_false,
        "n_score_unsupported": n_unsupported,
        "by_mechanism": by_mech,
    }
    (OUTPUT_DIR / f"records_{MODE}.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUTPUT_DIR / f"summary_{MODE}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
