"""認識強化統一測定 (2026-08-17): 持続誤認 (1次目標②) の検出。

## 目的
各誤りセルについて、収集npzの時系列で「同じ誤った値」が
PERSIST_FRAME_THRESHOLD (5フレーム、実効30fps) 以上持続していたかを判定する。
設置直後の反映遅延 (REFLECTION_DELAY_MAX_FRAMES=8フレーム以内、
feedback_placement_reflection_8frames) は別枠 (reflection_delay) として計上し、
「持続誤認」の件数には含めない。

## 測定器としての注意 (fail-silent警戒)
npz の STABLE snapshot は **直前と同一盤面なら間引き** される仕様
(scripts/collect_boards_lean.py `_should_emit`)。そのため
「持続時間」は npz の行数ではなく、**同一チャンク内で対象セルの値が
連続して同じだった区間の t_sec 差分** (フレーム換算は effective 30fps
= 秒数 * 30) で測る。区間がチャンク先頭/末尾に接する場合は
「chunk_boundary_censored」として真の持続時間が不明である旨を明記する
(過小評価の可能性を隠さない)。

reflection_delay 判定は以下の全てを満たす場合のみ (ヒューリスティック、
完全な設置イベント検出ではないことを明記):
    (a) 持続フレーム数 <= REFLECTION_DELAY_MAX_FRAMES (8)
    (b) 区間開始の直前セル値が EMPTY(0) (新規設置直後の反映待ちパターン)
    (c) 区間終了直後にセル値が正解値へ自己修復している

使い方:
    python scripts/_diag_persistent_misread_2026-08-17.py --tag a
    python scripts/_diag_persistent_misread_2026-08-17.py --tag b
    python scripts/_diag_persistent_misread_2026-08-17.py --all
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_sc = importlib.import_module("scripts._score_yardstick_v2_2026-08-14")

YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
SCORING_DIR = YARDSTICK_DIR / "scoring_ablation"
OUT_DIR = _ROOT / "data" / "verify" / "recognition_unified_2026-08-17"

EFFECTIVE_FPS: float = 30.0  # collect側の normalize_fps_30 と同じ意味論
PERSIST_FRAME_THRESHOLD: float = 5.0  # 1次目標②の閾値
REFLECTION_DELAY_MAX_FRAMES: float = 8.0  # feedback_placement_reflection_8frames

# 構成タグ -> npz ディレクトリ (収集済み実体)
NPZ_DIRS: dict[str, Path] = {
    "a": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w13p2_2026-08-17",
    "b": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_r2_2026-08-17",
    "c": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w10guard_2026-08-17",
    "d": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_r2w10_2026-08-17",
}


def _load_chunk_series(npz_path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """1チャンクnpz内を (video_id, side) -> frame_idx昇順の行リスト に整理する。"""
    d = np.load(npz_path, allow_pickle=True)
    n = len(d["frame_idx"])
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for i in range(n):
        vid = str(d["video_id"][i]).removeprefix("_hold_").removeprefix("video_")
        side = str(d["side"][i])
        by_key.setdefault((vid, side), []).append({
            "frame_idx": int(d["frame_idx"][i]),
            "t_sec": float(d["t_sec"][i]),
            "grid": d["grids"][i],
        })
    for rows in by_key.values():
        rows.sort(key=lambda r: r["frame_idx"])
    return by_key


def _find_run(rows: list[dict[str, Any]], anchor_idx: int, r: int, c: int) -> dict[str, Any]:
    """anchor_idx 行のセル(r,c)値と同じ値が連続する区間を前後に伸ばして返す。"""
    val = int(rows[anchor_idx]["grid"][r, c])
    lo = anchor_idx
    while lo - 1 >= 0 and int(rows[lo - 1]["grid"][r, c]) == val:
        lo -= 1
    hi = anchor_idx
    while hi + 1 < len(rows) and int(rows[hi + 1]["grid"][r, c]) == val:
        hi += 1
    duration_sec = rows[hi]["t_sec"] - rows[lo]["t_sec"]
    frames_equiv = duration_sec * EFFECTIVE_FPS
    prev_val = int(rows[lo - 1]["grid"][r, c]) if lo - 1 >= 0 else None
    next_val = int(rows[hi + 1]["grid"][r, c]) if hi + 1 < len(rows) else None
    boundary_censored = (lo == 0) or (hi == len(rows) - 1)
    return {
        "value": val, "lo_idx": lo, "hi_idx": hi,
        "frames_equiv": frames_equiv, "duration_sec": duration_sec,
        "prev_val": prev_val, "next_val": next_val,
        "boundary_censored": boundary_censored,
        "t_start": rows[lo]["t_sec"], "t_end": rows[hi]["t_sec"],
    }


def analyze_tag(tag: str) -> dict[str, Any]:
    """構成tagのscore_{tag}.jsonから誤りセルを抽出し、持続判定する。"""
    score_path = SCORING_DIR / f"score_{tag}.json"
    if not score_path.exists():
        raise FileNotFoundError(f"score_{tag}.json が無い (fail-silent禁止で要報告)")
    rows = json.loads(score_path.read_text(encoding="utf-8"))

    npz_dir = NPZ_DIRS[tag]
    # チャンクファイルごとの (video,side)->series をキャッシュ
    chunk_cache: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}

    persistent: list[dict[str, Any]] = []
    reflection_delay: list[dict[str, Any]] = []
    n_wrong_cells = 0
    n_wrong_cells_analyzable = 0

    for row in rows:
        if row["match_method"] == "miss":
            continue
        npz_name = row.get("npz")
        frame_idx = row.get("frame_idx")
        if npz_name is None or frame_idx is None:
            continue  # 旧score_a.json流用分は npz/frame_idx が無いためスキップ
        wrong_cells = [c for c in row.get("cells", []) if not c["is_correct"]]
        if not wrong_cells:
            continue
        if npz_name not in chunk_cache:
            chunk_cache[npz_name] = _load_chunk_series(npz_dir / npz_name)
        series = chunk_cache[npz_name]
        key = (row["video_id"], row["side"])
        chunk_rows = series.get(key)
        if not chunk_rows:
            continue
        anchor_idx = next(
            (i for i, r in enumerate(chunk_rows) if r["frame_idx"] == frame_idx), None,
        )
        if anchor_idx is None:
            continue
        for cell in wrong_cells:
            n_wrong_cells += 1
            r, c = cell["r"], cell["c"]
            run = _find_run(chunk_rows, anchor_idx, r, c)
            n_wrong_cells_analyzable += 1
            entry = {
                "sheet_id": row["sheet_id"], "video_id": row["video_id"], "side": row["side"],
                "r": r, "c": c, "wrong_value": cell["pred"], "correct_value": cell["correct"],
                "frames_equiv": round(run["frames_equiv"], 2),
                "duration_sec": round(run["duration_sec"], 4),
                "boundary_censored": run["boundary_censored"],
                "prev_val": run["prev_val"], "next_val": run["next_val"],
            }
            is_reflection = (
                run["frames_equiv"] <= REFLECTION_DELAY_MAX_FRAMES
                and run["prev_val"] == 0
                and run["next_val"] == cell["correct"]
            )
            if is_reflection:
                reflection_delay.append(entry)
                continue
            if run["frames_equiv"] >= PERSIST_FRAME_THRESHOLD:
                persistent.append(entry)

    return {
        "tag": tag,
        "n_wrong_cells_total": n_wrong_cells,
        "n_wrong_cells_analyzable": n_wrong_cells_analyzable,
        "n_persistent": len(persistent),
        "n_reflection_delay": len(reflection_delay),
        "persistent_cells": persistent,
        "reflection_delay_cells": reflection_delay,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["a", "b", "c", "d"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    tags = ["a", "b", "c", "d"] if args.all else ([args.tag] if args.tag else [])
    if not tags:
        ap.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        result = analyze_tag(tag)
        out_path = OUT_DIR / f"persistent_misread_{tag}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{tag}] 誤りセル総数={result['n_wrong_cells_total']} "
            f"解析可={result['n_wrong_cells_analyzable']} "
            f"持続誤認(>=5f)={result['n_persistent']} "
            f"反映遅延別枠(<=8f)={result['n_reflection_delay']}"
        )
        if result["n_persistent"]:
            print("  持続誤認セル一覧:")
            for e in result["persistent_cells"]:
                cens = " [chunk境界censored]" if e["boundary_censored"] else ""
                print(
                    f"    {e['sheet_id']} r{e['r']}c{e['c']}: "
                    f"誤り値={e['wrong_value']} 正解={e['correct_value']} "
                    f"持続={e['frames_equiv']}フレーム相当"
                    f"({e['duration_sec']}秒){cens}"
                )
        print(f"[{tag}] -> {out_path}")


if __name__ == "__main__":
    main()
