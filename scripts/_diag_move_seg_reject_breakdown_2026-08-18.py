"""1手区切りスケジューラ+持続的物理制約フィルタの棄却内訳計装 (2026-08-18)。

collect_boards_lean.py の本体コードは変更せず、_is_physics_violation_persistent /
_move_window_candidate_ok を monkeypatch してカウントするだけの計装スクリプト。
3動画 (36/52/c100、--start-sec 150) で実行し、
  - 物理制約フィルタの評価回数 / 棄却回数 (棄却率)
  - 棄却時の (video, side, frame_idx, signature) を記録 (空振り率の個別確認用)
を出力する。

実行:
    PYTHONPATH=. python scripts/_diag_move_seg_reject_breakdown_2026-08-18.py <video_id> <max_sec>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.collect_boards_lean as mod  # noqa: E402
from src.production_config import recognition_load_default_kwargs  # noqa: E402

OUT_DIR = Path("logs")

_counters = {
    "physics_evaluated": 0,
    "physics_rejected": 0,
    "window_candidate_checked": 0,
    "window_candidate_ok": 0,
}
_rejections: list[dict] = []

_orig_physics = mod._is_physics_violation_persistent
_orig_window_ok = mod._move_window_candidate_ok


def _patched_physics(state, board, sim):
    _counters["physics_evaluated"] += 1
    rejected = _orig_physics(state, board, sim)
    if rejected:
        _counters["physics_rejected"] += 1
        _rejections.append({
            "signature": sorted(list(state.prev_violation_signature or [])),
        })
    return rejected


def _patched_window_ok(state, frame_idx):
    _counters["window_candidate_checked"] += 1
    ok = _orig_window_ok(state, frame_idx)
    if ok:
        _counters["window_candidate_ok"] += 1
    return ok


def main() -> None:
    video_id = sys.argv[1]
    max_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    start_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
    use_move_segment = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
    suffix = "" if use_move_segment else "_eventdriven"

    mod._is_physics_violation_persistent = _patched_physics
    mod._move_window_candidate_ok = _patched_window_ok

    video_path = Path(f"data/frames/video_{video_id}.mp4")
    out_npz = Path(f"/tmp/_diag_reject_{video_id}{suffix}.npz")
    kwargs = dict(recognition_load_default_kwargs())
    n = mod.collect_lean(
        video_path, out_npz,
        start_sec=start_sec, max_sec=max_sec,
        capture_next=True,
        enable_chain_tracker=True,
        enable_move_segmented_recording=use_move_segment,
        enable_physics_persistence_filter=True,
        **kwargs,
    )
    report = {
        "video": video_id, "n_recorded": n,
        "move_segmented_recording": use_move_segment,
        "counters": _counters,
        "rejections_sample": _rejections[:30],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        OUT_DIR
        / f"_diag_move_seg_reject_breakdown_2026-08-18_{video_id}{suffix}.json"
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
