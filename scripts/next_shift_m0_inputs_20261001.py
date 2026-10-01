"""M0 (T03) の学習入力を補正 queue で作り直す (2026-10-01)。

元手順: 56dc scripts/r2_grouped_cv_148_20260925.py の prepare_data / joined_metadata / pair_arrays。
同じ関数を写し、原票の queue 列だけを補正版に差し替える。
台の確認: 原票のままで作った入力が既存の inputs/<video>.npz と全配列バイト一致すること (動画ごとに記録)。
出力: <output>/inputs/<video>.npz、<output>/INPUTS.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.next_shift_common_20261001 import LEAN_ROOT, QUEUE_KEYS, read_raw
from src.advantage_m0_current_cnn_v1 import BOARD_VALUE_TO_CATEGORY

ORIGINAL = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
REFERENCE_INPUTS = Path("/mnt/c/Users/ryouj/.codex/worktrees/56dc/puyo_analyzer/logs/"
                        "r2_grouped_cv_148_20260925/inputs")
CSV_PATH = ORIGINAL / "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
META = ["video_id", "game_idx", "t_sec", "frame", "side", "won"]
QUEUES = Path("logs/next_shift_train/queues")
WORKERS = 7


def load_builder():
    """T03 が使った build_labeled_win_from_npz (56dc と ORIGINAL で同一ファイル) を読む。"""
    spec = importlib.util.spec_from_file_location("t03_builder", ORIGINAL / "scripts/build_labeled_win_from_npz.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joined_metadata(builder, meta: pd.DataFrame, raw: dict[str, np.ndarray]) -> pd.DataFrame:
    """T03 joined_metadata の写し。"""
    records = builder._build_meta_rows(*(raw[key] for key in
        ("video_id", "game_idx", "t_sec", "frame_idx", "side", "won")))
    source = pd.DataFrame(records)
    keys = ["video_id", "game_idx", "frame", "side"]
    own = meta.merge(source[keys + ["_source_grid_idx"]], on=keys, how="left", validate="one_to_one")
    assert own["_source_grid_idx"].notna().all()
    own["t_sec"] = own["t_sec"].astype(np.float32).astype(np.float64)
    groups, columns = ["video_id", "game_idx"], ["_source_grid_idx"]
    pieces = []
    for side, other in (("1P", "2P"), ("2P", "1P")):
        opponents = builder._grouped_side_frames(source, other, groups, columns)
        pieces.append(builder._asof_attach_opponent(own[own.side == side], opponents, groups, columns))
    result = pd.concat(pieces, ignore_index=True)
    assert len(result) == len(meta)
    return result


def pair_arrays(meta: pd.DataFrame, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """T03 pair_arrays の写し (queue は raw の QUEUE_KEYS 列を読む)。"""
    own = meta["_source_grid_idx"].to_numpy(dtype=np.int64)
    opposite = meta["_opp__source_grid_idx"].fillna(-1).to_numpy(dtype=np.int64)
    missing = opposite < 0
    opposite_safe = np.maximum(opposite, 0)
    grids = np.stack((raw["grids"][own], raw["grids"][opposite_safe]), axis=1)
    grids[missing, 1] = 10
    queue = np.stack([raw[key] for key in QUEUE_KEYS], axis=1)
    queues = np.stack((queue[own], queue[opposite_safe]), axis=1)
    queues[missing, 1] = 0
    swap = meta.side.to_numpy() == "2P"
    grids[swap] = grids[swap, ::-1]
    queues[swap] = queues[swap, ::-1]
    boards = np.empty_like(grids)
    for value, category in BOARD_VALUE_TO_CATEGORY.items():
        boards[grids == value] = category
    assert np.isin(grids, list(BOARD_VALUE_TO_CATEGORY)).all()
    queues = np.where((queues >= 1) & (queues <= 5), queues, 0).astype(np.int8)
    labels = meta.won.to_numpy(dtype=np.float32)
    labels = np.where(swap, 1.0 - labels, labels).astype(np.float32)
    games = meta.video_id.astype(str) + ":" + meta.game_idx.astype(str)
    minimum = meta.groupby("game_idx").t_sec.transform("min").to_numpy()
    maximum = meta.groupby("game_idx").t_sec.transform("max").to_numpy()
    progress = (meta.t_sec.to_numpy() - minimum) / np.maximum(maximum - minimum, 1e-12)
    return dict(boards=boards, queues=queues, labels=labels,
                source_groups=meta.video_id.to_numpy(dtype=str), game_keys=games.to_numpy(dtype=str),
                t_sec=meta.t_sec.to_numpy(), frame=meta.frame.to_numpy(),
                source_side=meta.side.to_numpy(dtype=str),
                phase=np.minimum((progress * 3).astype(np.int8), 2), opposite_missing=missing)


def with_queue(raw: dict[str, np.ndarray], queue: np.ndarray) -> dict[str, np.ndarray]:
    """queue 列だけ差し替えた原票の浅い複製。"""
    out = dict(raw)
    for i, key in enumerate(QUEUE_KEYS):
        out[key] = queue[:, i].astype(raw[key].dtype)
    return out


def video_job(job: tuple[str, pd.DataFrame, str, str]) -> dict:
    """1動画: 原票のままの再現照合と、補正版の保存。"""
    video, selected, variant, output = job
    builder = load_builder()
    raw = read_raw(LEAN_ROOT / (video.removeprefix("video_") + ".npz"))
    joined = joined_metadata(builder, selected.copy(), raw)
    reference = read_raw(REFERENCE_INPUTS / (video + ".npz"))
    rebuilt = pair_arrays(joined, raw)
    same = {k: bool(np.array_equal(rebuilt[k], reference[k])) for k in reference}
    with np.load(QUEUES / (video.removeprefix("video_") + ".npz")) as fixed:
        data = pair_arrays(joined, with_queue(raw, fixed["queue_" + variant]))
    changed = int((data["queues"] != reference["queues"]).any(axis=(1, 2)).sum())
    path = Path(output) / "inputs" / (video + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return dict(video=video, rows=len(data["labels"]), identical=all(same.values()),
                mismatched_keys=[k for k, v in same.items() if not v], changed_rows=changed)


def main() -> None:
    """144動画を並列に作り、再現照合と変更行数を INPUTS.json へ。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("C", "T"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = pd.read_csv(CSV_PATH, usecols=META).dropna(subset=["won"])
    videos = sorted(p.stem for p in REFERENCE_INPUTS.glob("video_*.npz"))
    jobs = [(v, meta[meta.video_id == v], args.variant, str(args.output)) for v in videos]
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(video_job, jobs))
    summary = dict(variant=args.variant, videos=len(results), rows=sum(r["rows"] for r in results),
                   identical_videos=sum(r["identical"] for r in results),
                   changed_rows=sum(r["changed_rows"] for r in results), per_video=results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "INPUTS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}), flush=True)


if __name__ == "__main__":
    main()
