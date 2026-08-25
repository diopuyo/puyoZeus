"""新盤面収集方式の「隔離されたA/B」実例抽出 (2026-08-18、検収レビュア計装)。

_review_new_collect_before_after_2026-08-18.py は文字通りの
data/indicators_v2/boards_lean_phase_l_2026-08-11/ (旧本番データ、2026-08-11
時点のコード) と比較するため、境界ハードニング等 *この機能と無関係な*
2026-08-17/18 の別改善も差分に混ざる (交絡)。

本スクリプトは同一コード・同一フラグ (move-segmented/physics-persistence の
2フラグだけが違う) の baseline npz vs new_v2or npz を比較し、
「1手区切りスケジューラ+物理制約フィルタ」単体の効果だけを隔離して
実例を抽出する。

出力: data/verify/move_segment_physics_filter_2026-08-18/isolated_ab_frames/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.production_config import GHOST_CHAIN_RULE_ENABLED  # noqa: E402
from src.self_supervised.physical_consistency import check_gravity_rule  # noqa: E402

NEW_DIR = Path("data/verify/move_segment_physics_filter_2026-08-18")
FRAMES_DIR = Path("data/frames")
OUT_DIR = NEW_DIR / "isolated_ab_frames"
TARGET_VIDEOS: tuple[str, ...] = ("36", "52", "c100")
WIN_LO, WIN_HI = 150.0, 300.0
MATCH_TOL_SEC = 2.0
MAX_SAMPLES_PER_KIND = 4
TARGET_W, TARGET_H = 1920, 1080


def _region_for(side: str):
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def _load_rows(path: Path) -> list[dict]:
    d = dict(np.load(path, allow_pickle=True))
    t = d["t_sec"]
    m = (t >= WIN_LO) & (t < WIN_HI)
    idxs = np.nonzero(m)[0]
    return [
        {
            "side": str(d["side"][i]), "t_sec": float(t[i]),
            "frame_idx": int(d["frame_idx"][i]), "grid": d["grids"][i],
            "game_idx": int(d["game_idx"][i]),
        }
        for i in idxs
    ]


def _find_match(row: dict, others: list[dict]) -> bool:
    for o in others:
        if o["side"] != row["side"] or abs(o["t_sec"] - row["t_sec"]) > MATCH_TOL_SEC:
            continue
        if np.array_equal(o["grid"], row["grid"]):
            return True
    return False


def _physics_flags(grid: np.ndarray, sim: ChainSimulator) -> dict:
    board = Board.from_list(grid.tolist())
    has_erasable = len(sim.find_erasable_groups(board)) > 0
    gravity_valid, _ = check_gravity_rule(board)
    return {"has_erasable_group": has_erasable, "has_gravity_violation": (not gravity_valid)}


def _extract_frame_png(video_id: str, frame_idx: int, side: str, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(FRAMES_DIR / f"video_{video_id}.mp4"))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    if frame.shape[:2] != (TARGET_H, TARGET_W):
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    region = _region_for(side)
    crop = frame[region.y:region.y + region.height, region.x:region.x + region.width]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return True


def process_video(video_id: str, sim: ChainSimulator) -> dict:
    base_path = NEW_DIR / f"{video_id}_baseline.npz"
    new_path = NEW_DIR / f"{video_id}_new_v2or.npz"
    base_rows = _load_rows(base_path)
    new_rows = _load_rows(new_path)

    base_only = [r for r in base_rows if not _find_match(r, new_rows)]
    new_only = [r for r in new_rows if not _find_match(r, base_rows)]

    # 各行の物理フラグを1回だけ計算し、行オブジェクトの id で分類する
    # (dict に numpy 配列が入っているため `in`/`==` による比較は使えない)。
    base_only_flags = [_physics_flags(r["grid"], sim) for r in base_only]
    n_e = sum(1 for f in base_only_flags if f["has_erasable_group"])
    n_g = sum(1 for f in base_only_flags if f["has_gravity_violation"])
    n_clean = sum(
        1 for f in base_only_flags
        if not f["has_erasable_group"] and not f["has_gravity_violation"]
    )

    summary = {
        "n_baseline_in_window": len(base_rows), "n_new_v2or_in_window": len(new_rows),
        "n_baseline_only_dropped": len(base_only), "n_new_only_captured": len(new_only),
        "baseline_only_erasable_group": n_e, "baseline_only_gravity_violation": n_g,
        "baseline_only_clean_no_violation": n_clean,
    }

    dropped_violation = [
        (r, f) for r, f in zip(base_only, base_only_flags)
        if f["has_erasable_group"] or f["has_gravity_violation"]
    ]
    dropped_clean = [
        (r, f) for r, f in zip(base_only, base_only_flags)
        if not f["has_erasable_group"] and not f["has_gravity_violation"]
    ]

    samples: dict[str, list[dict]] = {"dropped_with_violation": [], "dropped_clean": [], "newly_captured": []}
    for kind, rows in (
        ("dropped_with_violation", dropped_violation),
        ("dropped_clean", dropped_clean),
        ("newly_captured", [(r, None) for r in new_only]),
    ):
        step = max(1, len(rows) // MAX_SAMPLES_PER_KIND) if rows else 1
        picked = rows[::step][:MAX_SAMPLES_PER_KIND]
        for r, f in picked:
            png_path = OUT_DIR / video_id / kind / (
                f"{r['side']}_t{r['t_sec']:.2f}_fi{r['frame_idx']}.png"
            )
            ok = _extract_frame_png(video_id, r["frame_idx"], r["side"], png_path)
            entry = {
                "side": r["side"], "t_sec": round(r["t_sec"], 2),
                "frame_idx": r["frame_idx"], "game_idx": r["game_idx"],
                "png": str(png_path) if ok else None,
            }
            if f is not None:
                entry["physics"] = f
            samples[kind].append(entry)

    summary["samples"] = samples
    return summary


def main() -> None:
    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    report: dict[str, dict] = {}
    for vid in TARGET_VIDEOS:
        print(f"[{vid}] 処理中...")
        report[vid] = process_video(vid, sim)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = NEW_DIR / "isolated_ab_report_2026-08-18.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in report.items()},
        ensure_ascii=False, indent=2,
    ))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
