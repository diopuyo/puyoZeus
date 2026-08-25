"""新盤面収集方式のビフォーアフター実例抽出 (2026-08-18、検収レビュア計装)。

比較:
  OLD = data/indicators_v2/boards_lean_phase_l_2026-08-11/{vid}.npz
        (実際に学習に使われていた旧方式データそのもの)
  NEW = data/verify/move_segment_physics_filter_2026-08-18/{vid}_new_v2or.npz
        (コーダエージェント生成の新方式データ、--start-sec 150 --max-sec 150、
        OR条件化後の最終構成。commit 97cc37fの実測値と行数完全一致を確認済み)

処理内容 (窓 [150, 300) 秒、side別):
  1. OLD の各行について、NEW 側に grid が完全一致する行が
     時刻 ±2.0 秒以内にあるかを探す。無ければ「新方式で棄却された
     候補 (old_only)」。
  2. NEW の各行について、OLD 側に grid が完全一致する行が
     時刻 ±2.0 秒以内にあるかを探す。無ければ「新方式で新たに
     拾えた候補 (new_only)」。
  3. old_only / new_only それぞれについて、実フレーム
     (data/frames/video_{vid}.mp4, frame_idx列) を切り出して
     PNG保存 (盤面ROI全体、cv2)。件数が多い場合は間引く (目視レビュー用サンプル)。
  4. old_only 行について、消去可能グループ残存/重力違反
     (=物理制約フィルタが狙う汚染) の有無を判定し、
     「本当に汚染だったか」の裏付け情報として記録する。

出力: data/verify/move_segment_physics_filter_2026-08-18/before_after_review/
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

OLD_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
NEW_DIR = Path("data/verify/move_segment_physics_filter_2026-08-18")
FRAMES_DIR = Path("data/frames")
OUT_DIR = NEW_DIR / "before_after_review"
TARGET_VIDEOS: tuple[str, ...] = ("36", "52", "c100")
WIN_LO, WIN_HI = 150.0, 300.0
MATCH_TOL_SEC = 2.0
MAX_SAMPLES_PER_KIND = 6

TARGET_W, TARGET_H = 1920, 1080


def _region_for(side: str):
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def _load_rows(path: Path, lo: float, hi: float) -> list[dict]:
    d = dict(np.load(path, allow_pickle=True))
    t = d["t_sec"]
    m = (t >= lo) & (t < hi)
    idxs = np.nonzero(m)[0]
    rows = []
    for i in idxs:
        rows.append({
            "side": str(d["side"][i]), "t_sec": float(t[i]),
            "frame_idx": int(d["frame_idx"][i]), "grid": d["grids"][i],
            "game_idx": int(d["game_idx"][i]),
        })
    return rows


def _find_match(row: dict, other_rows: list[dict]) -> bool:
    for o in other_rows:
        if o["side"] != row["side"]:
            continue
        if abs(o["t_sec"] - row["t_sec"]) > MATCH_TOL_SEC:
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
    old_path = OLD_DIR / f"{video_id}.npz"
    new_path = NEW_DIR / f"{video_id}_new_v2or.npz"
    if not old_path.exists() or not new_path.exists():
        return {"status": "missing_input", "old_exists": old_path.exists(), "new_exists": new_path.exists()}

    old_rows = _load_rows(old_path, WIN_LO, WIN_HI)
    new_rows = _load_rows(new_path, WIN_LO, WIN_HI)

    old_only = [r for r in old_rows if not _find_match(r, new_rows)]
    new_only = [r for r in new_rows if not _find_match(r, old_rows)]

    summary = {
        "n_old_in_window": len(old_rows), "n_new_in_window": len(new_rows),
        "n_old_only_rejected_candidates": len(old_only),
        "n_new_only_captured_candidates": len(new_only),
    }

    samples: dict[str, list[dict]] = {"old_only": [], "new_only": []}
    for kind, rows in (("old_only", old_only), ("new_only", new_only)):
        step = max(1, len(rows) // MAX_SAMPLES_PER_KIND) if rows else 1
        picked = rows[::step][:MAX_SAMPLES_PER_KIND]
        for r in picked:
            png_path = OUT_DIR / video_id / kind / (
                f"{r['side']}_t{r['t_sec']:.2f}_fi{r['frame_idx']}.png"
            )
            ok = _extract_frame_png(video_id, r["frame_idx"], r["side"], png_path)
            entry = {
                "side": r["side"], "t_sec": round(r["t_sec"], 2),
                "frame_idx": r["frame_idx"], "game_idx": r["game_idx"],
                "png": str(png_path) if ok else None,
            }
            if kind == "old_only":
                entry["physics"] = _physics_flags(r["grid"], sim)
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
    out_json = NEW_DIR / "before_after_report_review_2026-08-18.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
