"""
セル認識精度と解像度・fps・色調(代理指標)の関係を分析する診断スクリプト。

読み取り・軽量集計のみ。フレームのデコード・認識再実行は禁止(cv2 プロパティ読みのみ許可)。
既存ファイルは一切変更しない (read-only diagnostic)。

対象データ:
  1. 19動画分のセル正解率 (data/verify/cell_accuracy_recheck_2026-07-26/*.json の per_video)
  2. その19動画の実ファイルから cv2 プロパティで解像度・fps を実測
  3. 66動画分 (c10-c81) の npz (data/indicators_v2/boards_lean_fixed/*.npz) から
     正解ラベル不要の代理指標 (COLOR_UNKNOWN率・物理あり得ない色遷移=ちらつき率) を計算
  4. data/verify/video_difficulty_2026-07-29.csv (解像度・fps・score系メトリクス) と突合

出力: JSON 1本 (scripts/_diag_cell_recognition_difficulty_2026-07-29_out.json) + 標準出力サマリ。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 定数 (マジックナンバー禁止規約に従い定義)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CELL_ACC_DIR = REPO_ROOT / "data" / "verify" / "cell_accuracy_recheck_2026-07-26"
WITH51_JSON = CELL_ACC_DIR / "with51_flags_2026-07-26.json"
CURRENT_DEFAULT_JSON = CELL_ACC_DIR / "current_default_2026-07-26.json"
VIDEO_DIFFICULTY_CSV = REPO_ROOT / "data" / "verify" / "video_difficulty_2026-07-29.csv"
BOARDS_LEAN_FIXED_DIR = REPO_ROOT / "data" / "indicators_v2" / "boards_lean_fixed"
OUT_JSON = REPO_ROOT / "scripts" / "_diag_cell_recognition_difficulty_2026-07-29_out.json"

COLOR_EMPTY = 0
COLOR_UNKNOWN = 10
COLOR_OJAMA = 9
# 実在色 (お邪魔含む・空/UNKNOWN除く) の集合。直接色→別色への遷移はゲームルール上あり得ない
# (消去は必ず空セルを経由し、設置は空セルにのみ起こる)。
REAL_COLOR_MIN = 1
REAL_COLOR_MAX = 9

# 19動画の per_video キー -> 実ファイルパス (評価用クリップ、削除されていない前提)
EVAL_VIDEO_FILE_MAP: dict[str, str] = {
    "v29_match2_156s": "data/evaluation_videos/v29_match2_156s.mp4",
    "v29m2_buf15s": "data/holdout_videos/v29m2_buf15s.mp4",
    "v30_5min_90s": "data/holdout_videos/v30_5min_90s.mp4",
    "v30_match11_89s": "data/holdout_videos/v30_match11_89s.mp4",
    "v30_match11_buf15s": "data/holdout_videos/v30_match11_buf15s.mp4",
    "v40_match7_125s": "data/evaluation_videos/v40_match7_125s.mp4",
    "v40m7_buf15s": "data/holdout_videos/v40m7_buf15s.mp4",
    "v51_match2_97s": "data/evaluation_videos/v51_match2_97s.mp4",
    "v51m2_buf15s": "data/holdout_videos/v51m2_buf15s.mp4",
    "v57_match2_100s": "data/evaluation_videos/v57_match2_100s.mp4",
    "v57m2_buf15s": "data/holdout_videos/v57m2_buf15s.mp4",
    "v70_match2_113s": "data/evaluation_videos/v70_match2_113s.mp4",
    "v70m2_buf15s": "data/holdout_videos/v70m2_buf15s.mp4",
    "v89_match3_95s": "data/evaluation_videos/v89_match3_95s.mp4",
    "v89m3_buf15s": "data/holdout_videos/v89m3_buf15s.mp4",
    "v95_match15_99s": "data/evaluation_videos/v95_match15_99s.mp4",
    "v95m15_buf15s": "data/holdout_videos/v95m15_buf15s.mp4",
    "v97_match11_96s": "data/evaluation_videos/v97_match11_96s.mp4",
    "v97_match11_buf15s": "data/holdout_videos/v97m11_buf15s.mp4",
}


def load_per_video_acc(path: Path) -> dict[str, dict[str, Any]]:
    """cell_accuracy_recheck の per_video を読む (加工なし)。"""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["per_video"]


def measure_video_properties(rel_path: str) -> dict[str, Any] | None:
    """cv2 プロパティのみで解像度・fps・総フレーム数を実測する (デコードしない)。"""
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        return None
    cap = cv2.VideoCapture(str(full_path))
    if not cap.isOpened():
        cap.release()
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "fourcc": fourcc,
    }


def load_video_difficulty_csv(path: Path) -> dict[str, dict[str, Any]]:
    """66動画分の解像度・fps・score系メトリクスを読む。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["video_stem"]: row for row in rows}


def compute_color_unknown_rate(grids: np.ndarray) -> tuple[int, int, float]:
    """COLOR_UNKNOWN(=10) セルの出現率を計算する。"""
    total = int(grids.size)
    unknown = int(np.count_nonzero(grids == COLOR_UNKNOWN))
    rate = unknown / total if total > 0 else 0.0
    return unknown, total, rate


def _is_impossible_direct_color_swap(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    連続する確定盤面間で「実在色 A -> 実在色 B (A != B)」への直接遷移を検出する。

    ゲームルール上、設置済みぷよの色が空セルを経由せず別の色に変わることはない
    (消去は必ず空セル化を経由し、新規設置は空セルにのみ起こるため)。
    このため直接遷移はセル認識の「ちらつき」= 誤読の代理指標として扱える。
    """
    prev_is_real = (prev >= REAL_COLOR_MIN) & (prev <= REAL_COLOR_MAX)
    curr_is_real = (curr >= REAL_COLOR_MIN) & (curr <= REAL_COLOR_MAX)
    both_real = prev_is_real & curr_is_real
    changed = prev != curr
    return both_real & changed


def compute_flicker_rate(
    grids: np.ndarray, side: np.ndarray, game_idx: np.ndarray, t_sec: np.ndarray
) -> tuple[int, int, float]:
    """
    同一 side・同一試合内で時系列順に隣接する確定盤面間の「あり得ない直接色遷移」率を計算する。

    grids: (N, 13, 6) の確定盤面スナップショット列 (試合・side混在、時系列非保証)
    side, game_idx, t_sec: 各スナップショットのメタ情報 (grids と同じ長さ)
    """
    n = grids.shape[0]
    flicker_count = 0
    pair_count = 0
    # (side, game_idx) の組ごとに t_sec でソートして隣接ペアを走査する
    keys: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (str(side[i]), int(game_idx[i]))
        keys.setdefault(key, []).append(i)
    for indices in keys.values():
        indices.sort(key=lambda idx: float(t_sec[idx]))
        for a, b in zip(indices[:-1], indices[1:]):
            prev_grid = grids[a]
            curr_grid = grids[b]
            impossible = _is_impossible_direct_color_swap(prev_grid, curr_grid)
            flicker_count += int(np.count_nonzero(impossible))
            pair_count += int(prev_grid.size)
    rate = flicker_count / pair_count if pair_count > 0 else 0.0
    return flicker_count, pair_count, rate


def compute_npz_proxy_metrics(npz_path: Path) -> dict[str, Any]:
    """1動画分の npz から COLOR_UNKNOWN 率・ちらつき率を計算する。"""
    data = np.load(npz_path, allow_pickle=True)
    grids = data["grids"]
    side = data["side"]
    game_idx = data["game_idx"]
    t_sec = data["t_sec"]
    unknown_count, total_cells, unknown_rate = compute_color_unknown_rate(grids)
    flicker_count, pair_cells, flicker_rate = compute_flicker_rate(grids, side, game_idx, t_sec)
    return {
        "n_snapshots": int(grids.shape[0]),
        "total_cells": total_cells,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "flicker_count": flicker_count,
        "flicker_pair_cells": pair_cells,
        "flicker_rate": flicker_rate,
    }


def build_report() -> dict[str, Any]:
    """全セクションを組み立てて返す。"""
    report: dict[str, Any] = {}

    # --- セクション1: 19動画のセル正解率 (with51_flags / current_default) ---
    with51 = load_per_video_acc(WITH51_JSON)
    current_default = load_per_video_acc(CURRENT_DEFAULT_JSON)
    per_video_acc = {}
    for name in with51.keys():
        per_video_acc[name] = {
            "with51_acc": with51[name]["acc"],
            "with51_total_cells": with51[name]["total_cells"],
            "current_default_acc": current_default[name]["acc"],
            "current_default_total_cells": current_default[name]["total_cells"],
            "acc_delta_51_minus_default": with51[name]["acc"] - current_default[name]["acc"],
        }
    report["per_video_acc_19"] = per_video_acc

    # --- セクション2: 19動画の実測解像度・fps ---
    per_video_props = {}
    for name, rel_path in EVAL_VIDEO_FILE_MAP.items():
        props = measure_video_properties(rel_path)
        per_video_props[name] = props if props is not None else {"error": "file_not_found", "path": rel_path}
    report["per_video_measured_props_19"] = per_video_props

    # --- セクション3: 66動画の代理指標 (npz から計算) ---
    difficulty_rows = load_video_difficulty_csv(VIDEO_DIFFICULTY_CSV)
    per_video_proxy = {}
    for stem in sorted(difficulty_rows.keys(), key=lambda s: int(s[1:])):
        npz_path = BOARDS_LEAN_FIXED_DIR / f"{stem}.npz"
        if not npz_path.exists():
            per_video_proxy[stem] = {"error": "npz_not_found"}
            continue
        metrics = compute_npz_proxy_metrics(npz_path)
        csv_row = difficulty_rows[stem]
        metrics["width"] = int(csv_row["width"]) if csv_row["width"] else None
        metrics["height"] = int(csv_row["height"]) if csv_row["height"] else None
        metrics["fps"] = float(csv_row["fps"]) if csv_row["fps"] else None
        metrics["n_score_events"] = int(csv_row["n_score_events"]) if csv_row["n_score_events"] else 0
        metrics["inconsistent_ratio"] = (
            float(csv_row["inconsistent_ratio"]) if csv_row["inconsistent_ratio"] else None
        )
        metrics["auc_全体"] = float(csv_row["全体"]) if csv_row["全体"] else None
        metrics["auc_中盤"] = float(csv_row["中盤"]) if csv_row["中盤"] else None
        per_video_proxy[stem] = metrics
    report["per_video_proxy_66"] = per_video_proxy

    return report


def print_summary(report: dict[str, Any]) -> None:
    """標準出力に要約表を出す。"""
    print("=" * 100)
    print("セクション1: 19動画 セル正解率 (with51_flags vs current_default)")
    print("=" * 100)
    rows = sorted(
        report["per_video_acc_19"].items(), key=lambda kv: kv[1]["with51_acc"]
    )
    for name, v in rows:
        print(
            f"{name:24s} with51={v['with51_acc']:.5f}  current_default={v['current_default_acc']:.5f}"
            f"  delta={v['acc_delta_51_minus_default']:+.5f}  n_cells={v['with51_total_cells']}"
        )

    print()
    print("=" * 100)
    print("セクション2: 19動画 実測 解像度・fps")
    print("=" * 100)
    for name, props in report["per_video_measured_props_19"].items():
        print(f"{name:24s} {props}")

    print()
    print("=" * 100)
    print("セクション3: 66動画 代理指標 (COLOR_UNKNOWN率 / ちらつき率) ワースト10 (unknown_rate順)")
    print("=" * 100)
    proxy = {k: v for k, v in report["per_video_proxy_66"].items() if "error" not in v}
    worst_unknown = sorted(proxy.items(), key=lambda kv: -kv[1]["unknown_rate"])[:10]
    for name, v in worst_unknown:
        print(
            f"{name:8s} unknown_rate={v['unknown_rate']:.6f} flicker_rate={v['flicker_rate']:.6f}"
            f" {v['width']}x{v['height']} fps={v['fps']} n_score_events={v['n_score_events']}"
        )

    print()
    print("ワースト10 (flicker_rate順)")
    worst_flicker = sorted(proxy.items(), key=lambda kv: -kv[1]["flicker_rate"])[:10]
    for name, v in worst_flicker:
        print(
            f"{name:8s} flicker_rate={v['flicker_rate']:.6f} unknown_rate={v['unknown_rate']:.6f}"
            f" {v['width']}x{v['height']} fps={v['fps']} n_score_events={v['n_score_events']}"
        )


def main() -> None:
    report = build_report()
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print()
    print(f"詳細JSON出力: {OUT_JSON}")


if __name__ == "__main__":
    main()
