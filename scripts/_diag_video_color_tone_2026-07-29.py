"""動画の色調 (HSV) 実測 → 認識品質指標との相関を調べる軽量診断スクリプト (2026-07-29)。

user質問「色調や解像度など動画によって違うが、認識精度で厳しいものはあるか」の
「色調は未測定」だった穴を埋める。 各動画あたり **数フレームだけ** デコードし、
盤面領域の HSV 統計・色別 hue 実測値を取る (通し処理・認識再実行は禁止)。

対象:
  A. 66動画 (c10-c81, combined66) — 試合区間は
     data/verify/winners_panel_diff_gated_2026-07-26/video_{stem}.json から選ぶ。
     色別 hue は data/indicators_v2/boards_lean_fixed/{stem}.npz の確定盤面
     (既存認識結果、再計算しない) を「正解ラベル」として使い、対応する生フレーム
     から HSV を実測する (認識のやり直しではなく、フレームと既存ラベルの対応から
     HSV 値を再計測するだけ)。
  B. 19動画 (v系、cell_accuracy_recheck 対象の評価クリップ) — 単一試合に
     切り出し済クリップなので試合区間 JSON は不要、クリップ全体から等間隔サンプル。
     盤面領域 HSV のみ (per-cell 色別 hue はラベル (grids) が手元にないため断念、
     理由をレポートに明記する)。

出力: JSON 1本 (scripts/_diag_video_color_tone_2026-07-29_out.json) + 標準出力サマリ。

読み取り専用・軽量処理: 各動画あたり最大 N_SAMPLE_FRAMES フレームのみデコード。
動画の通し再生・CNN/HSV 認識の再実行は行わない。既存ファイルは一切変更しない。

使い方 (nice -n 19 で逐次実行、他タスクを妨げない):
    nice -n 19 env PYTHONPATH=. ./venv/bin/python \
        scripts/_diag_video_color_tone_2026-07-29.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.board import (  # noqa: E402
    COLOR_BLUE, COLOR_GREEN, COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_COLOR_RANGES, DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion,
)
from src.online_hsv_calibrator import H_CIRCULAR_SPLIT_THRESHOLD  # noqa: E402

# ---------------------------------------------------------------------------
# 定数 (マジックナンバー禁止規約に従い定義)
# ---------------------------------------------------------------------------
FRAMES_DIR = REPO_ROOT / "data" / "frames"
WINNERS_PANEL_DIR = (
    REPO_ROOT / "data" / "verify" / "winners_panel_diff_gated_2026-07-26"
)
NPZ_DIR = REPO_ROOT / "data" / "indicators_v2" / "boards_lean_fixed"
CELL_ACC_DIR = REPO_ROOT / "data" / "verify" / "cell_accuracy_recheck_2026-07-26"
WITH51_JSON = CELL_ACC_DIR / "with51_flags_2026-07-26.json"
CURRENT_DEFAULT_JSON = CELL_ACC_DIR / "current_default_2026-07-26.json"
FLICKER_JSON = REPO_ROOT / "scripts" / "_diag_cell_recognition_difficulty_2026-07-29_out.json"
VIDEO_DIFFICULTY_CSV = REPO_ROOT / "data" / "verify" / "video_difficulty_2026-07-29.csv"
OUT_JSON = REPO_ROOT / "scripts" / "_diag_video_color_tone_2026-07-29_out.json"

# 認識pipelineの慣例 (scripts/extract_per_video_hsv_ranges.py:61-62) に合わせ
# 1920x1080 にリサイズしてから盤面領域を切り出す
TARGET_W: int = 1920
TARGET_H: int = 1080

# 66動画 (c10-c81): scripts/_diag_video_difficulty_2026-07-29.py:48-56 の
# VIDEO_STEMS と同一 (ハイフン付きファイル名は import 不可のため複製、更新時は両方直すこと)
C_SERIES_STEMS: tuple[str, ...] = (
    "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19",
    "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c28", "c29",
    "c34", "c35", "c36", "c37", "c40", "c41", "c42", "c43", "c44", "c45",
    "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55",
    "c56", "c57", "c58", "c59", "c60", "c61", "c62", "c63", "c64", "c65",
    "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75",
    "c76", "c77", "c78", "c79", "c80", "c81",
)

# 19動画 (v系評価クリップ): scripts/_diag_cell_recognition_difficulty_2026-07-29.py
# の EVAL_VIDEO_FILE_MAP と同一
V_SERIES_FILE_MAP: dict[str, str] = {
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

# サンプリング関連
N_SAMPLE_FRAMES: int = 5
MARGIN_SEC_MATCH: int = 3  # 試合区間の前後を避けるマージン (秒、c系)
MARGIN_FRACTION_SHORT: float = 0.15  # 短尺クリップ(v系)用マージン比率
MIN_GAME_DURATION_SEC: float = 20.0  # c系の対象ゲーム最小長 (短すぎる試合は除外)
MAX_CELL_SYNC_DIFF_SEC: float = 2.0  # npz確定盤面とサンプルフレームの許容時刻差

# 色別 hue 実測対象色 (お邪魔・空・不明は対象外)
TARGET_COLORS: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)
COLOR_LABELS_JA: dict[int, str] = {
    COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫",
}


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------
def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    """指定秒にシークして1フレームだけ読む (それ以外はデコードしない)。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _resize_to_target(frame: np.ndarray) -> np.ndarray:
    """認識pipelineの慣例に合わせ 1920x1080 にリサイズする。"""
    if frame.shape[:2] == (TARGET_H, TARGET_W):
        return frame
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def _crop_region(frame: np.ndarray, region: BoardRegion) -> np.ndarray:
    """盤面領域 (可視12行) を切り出す。"""
    return frame[region.y:region.y + region.height, region.x:region.x + region.width]


def _hue_circular_distance(a: float, b: float) -> float:
    """OpenCV Hue (0-180、円環) の距離。"""
    d = abs(a - b)
    return min(d, 180.0 - d)


def _robust_hue_median_std(vals: list[float], color: int) -> tuple[float, float]:
    """色別 hue の median/std を計算する。

    RED は H=0 と H=180 の両端に分布 (循環折り返し) するため、単純 median を
    取ると多数派クラスタが H=90 (=緑!) 付近に潰れて崩壊する
    (src/online_hsv_calibrator.py の _circular_h_range と同一の既知バグパターン)。
    RED のみ多数派クラスタ (H<90 か H>=90 の多い方) を選んでから median/std を取る。
    """
    arr = np.array(vals)
    if color != COLOR_RED or arr.max() - arr.min() < H_CIRCULAR_SPLIT_THRESHOLD:
        return float(np.median(arr)), float(np.std(arr))
    low = arr[arr < H_CIRCULAR_SPLIT_THRESHOLD]
    high = arr[arr >= H_CIRCULAR_SPLIT_THRESHOLD]
    majority = low if len(low) >= len(high) else high
    return float(np.median(majority)), float(np.std(majority))


def _distance_to_default_range(h_value: float, color: int) -> tuple[bool, float]:
    """既定 DEFAULT_COLOR_RANGES に対する hue の逸脱度を計算する。

    Returns:
        (within: 既定レンジ内か, min_dist_deg: 最も近いレンジ境界までの円環距離(度、内なら0))
    """
    ranges = DEFAULT_COLOR_RANGES.get(color, [])
    best = float("inf")
    for r in ranges:
        if r.h_min <= h_value <= r.h_max:
            return True, 0.0
        d = min(_hue_circular_distance(h_value, r.h_min), _hue_circular_distance(h_value, r.h_max))
        best = min(best, d)
    return False, (0.0 if best == float("inf") else best)


def _pearson(x: list[float], y: list[float]) -> tuple[float | None, int]:
    """欠損を除いた Pearson 相関係数。n<3 または分散0なら None。"""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and not np.isnan(a) and not np.isnan(b)]
    n = len(pairs)
    if n < 3:
        return None, n
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    if xa.std() == 0 or ya.std() == 0:
        return None, n
    return float(np.corrcoef(xa, ya)[0, 1]), n


# ---------------------------------------------------------------------------
# セクションA: 66動画 (c系)
# ---------------------------------------------------------------------------
def _load_games_json(stem: str) -> list[dict[str, Any]]:
    path = WINNERS_PANEL_DIR / f"video_{stem}.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("games", [])


def _select_c_series_window(games: list[dict[str, Any]], coverage_limit: float) -> dict[str, Any]:
    """npz が実際にカバーする時間 (20分打ち切り等) 内で「中央付近の試合」を選ぶ。"""
    covered = [
        g for g in games
        if g.get("end_sec") is not None and g["end_sec"] <= coverage_limit
        and (g["end_sec"] - g["start_sec"]) >= MIN_GAME_DURATION_SEC
    ]
    note = "ok"
    if not covered:
        covered = [g for g in games if g.get("start_sec", 1e18) < coverage_limit]
        note = "fallback_any_duration"
    if not covered:
        return {
            "start_sec": 0.0, "end_sec": min(60.0, coverage_limit),
            "game_abs_idx": None, "note": "fallback_no_json_coverage",
        }
    chosen = covered[len(covered) // 2]
    return {
        "start_sec": float(chosen["start_sec"]), "end_sec": float(min(chosen["end_sec"], coverage_limit)),
        "game_abs_idx": chosen.get("game_abs_idx"), "note": note,
    }


def _sample_times(start: float, end: float, n: int, margin: float) -> list[float]:
    """[start+margin, end-margin] に n 点等間隔配置。短すぎる場合はマージン縮小。"""
    duration = end - start
    m = margin if duration > margin * 2 else max(0.0, duration * 0.1)
    lo, hi = start + m, end - m
    if hi <= lo:
        return [start + duration / 2.0]
    return list(np.linspace(lo, hi, n))


def _nearest_npz_row(t_sec_arr: np.ndarray, side_mask: np.ndarray, target_t: float) -> int | None:
    """同一 side 内で target_t に最も近い行の index (npz全体でのindex) を返す。"""
    idxs = np.nonzero(side_mask)[0]
    if idxs.size == 0:
        return None
    diffs = np.abs(t_sec_arr[idxs] - target_t)
    best_local = int(np.argmin(diffs))
    if diffs[best_local] > MAX_CELL_SYNC_DIFF_SEC:
        return None
    return int(idxs[best_local])


def _collect_cell_hues(
    frame: np.ndarray, region: BoardRegion, grid: np.ndarray,
) -> dict[int, list[float]]:
    """1フレーム分、grid のラベルに従い各セルの hue median を色別に集める。"""
    out: dict[int, list[float]] = {c: [] for c in TARGET_COLORS}
    h, w = frame.shape[:2]
    for row in range(1, grid.shape[0]):  # row=0 は隠し段 (画面外)
        for col in range(grid.shape[1]):
            color = int(grid[row, col])
            if color not in TARGET_COLORS:
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            out[color].append(float(np.median(hsv[:, :, 0])))
    return out


def _region_hsv_pixels(frame: np.ndarray, region: BoardRegion) -> np.ndarray:
    """盤面領域全体を HSV に変換して (N,3) 配列で返す。"""
    crop = _crop_region(frame, region)
    if crop.size == 0:
        return np.zeros((0, 3))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3)


def _process_c_series_video(stem: str) -> dict[str, Any]:
    """1動画分の色調計測 (盤面領域HSV + 色別hue)。"""
    video_path = FRAMES_DIR / f"video_{stem}.mp4"
    npz_path = NPZ_DIR / f"{stem}.npz"
    if not video_path.exists() or not npz_path.exists():
        return {"error": "missing_file", "video_exists": video_path.exists(),
                "npz_exists": npz_path.exists()}

    data = np.load(npz_path, allow_pickle=True)
    t_sec = data["t_sec"]
    side = data["side"]
    grids = data["grids"]
    coverage_limit = float(t_sec.max()) if t_sec.size else 0.0

    games = _load_games_json(stem)
    window = _select_c_series_window(games, coverage_limit)
    times = _sample_times(window["start_sec"], window["end_sec"], N_SAMPLE_FRAMES, MARGIN_SEC_MATCH)

    cap = cv2.VideoCapture(str(video_path))
    all_pixels: list[np.ndarray] = []
    hue_by_color: dict[int, list[float]] = {c: [] for c in TARGET_COLORS}
    n_frames_read = 0
    n_cell_sync_ok = 0
    n_cell_sync_total = 0
    for t in times:
        frame = _read_frame_at(cap, t)
        if frame is None:
            continue
        frame = _resize_to_target(frame)
        n_frames_read += 1
        for side_name, region in (("1P", DEFAULT_P1_REGION), ("2P", DEFAULT_P2_REGION)):
            all_pixels.append(_region_hsv_pixels(frame, region))
            n_cell_sync_total += 1
            side_mask = side == side_name
            row_idx = _nearest_npz_row(t_sec, side_mask, t)
            if row_idx is None:
                continue
            n_cell_sync_ok += 1
            cell_hues = _collect_cell_hues(frame, region, grids[row_idx])
            for c, vals in cell_hues.items():
                hue_by_color[c].extend(vals)
    cap.release()

    pooled = np.concatenate(all_pixels, axis=0) if all_pixels else np.zeros((0, 3))
    region_stats = {
        "h_mean": float(pooled[:, 0].mean()) if pooled.size else None,
        "h_std": float(pooled[:, 0].std()) if pooled.size else None,
        "s_mean": float(pooled[:, 1].mean()) if pooled.size else None,
        "s_std": float(pooled[:, 1].std()) if pooled.size else None,
        "v_mean": float(pooled[:, 2].mean()) if pooled.size else None,
        "v_std": float(pooled[:, 2].std()) if pooled.size else None,
        "n_pixels": int(pooled.shape[0]),
    }
    color_stats: dict[str, Any] = {}
    for c in TARGET_COLORS:
        vals = hue_by_color[c]
        within, dist, median_h, std_h = (None, None, None, None)
        if vals:
            median_h, std_h = _robust_hue_median_std(vals, c)
            within, dist = _distance_to_default_range(median_h, c)
        color_stats[COLOR_LABELS_JA[c]] = {
            "n": len(vals),
            "h_median": median_h,
            "h_std": std_h,
            "within_default_range": within,
            "dist_to_default_range_deg": dist,
        }

    return {
        "coverage_limit_sec": coverage_limit,
        "window": window,
        "n_frames_read": n_frames_read,
        "n_frames_target": len(times),
        "n_cell_sync_ok": n_cell_sync_ok,
        "n_cell_sync_total": n_cell_sync_total,
        "region_hsv": region_stats,
        "color_hue": color_stats,
    }


# ---------------------------------------------------------------------------
# セクションB: 19動画 (v系評価クリップ)
# ---------------------------------------------------------------------------
def _process_v_series_video(rel_path: str) -> dict[str, Any]:
    """1クリップ分の盤面領域HSV計測 (色別hueはラベル不在のため断念)。"""
    video_path = REPO_ROOT / rel_path
    if not video_path.exists():
        return {"error": "file_not_found", "path": rel_path}
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = n_total / fps if fps > 0 else 0.0
    margin = duration * MARGIN_FRACTION_SHORT
    times = _sample_times(0.0, duration, N_SAMPLE_FRAMES, margin)

    all_pixels: list[np.ndarray] = []
    n_frames_read = 0
    for t in times:
        frame = _read_frame_at(cap, t)
        if frame is None:
            continue
        frame = _resize_to_target(frame)
        n_frames_read += 1
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            all_pixels.append(_region_hsv_pixels(frame, region))
    cap.release()

    pooled = np.concatenate(all_pixels, axis=0) if all_pixels else np.zeros((0, 3))
    return {
        "duration_sec": float(duration),
        "n_frames_read": n_frames_read,
        "n_frames_target": len(times),
        "region_hsv": {
            "h_mean": float(pooled[:, 0].mean()) if pooled.size else None,
            "h_std": float(pooled[:, 0].std()) if pooled.size else None,
            "s_mean": float(pooled[:, 1].mean()) if pooled.size else None,
            "s_std": float(pooled[:, 1].std()) if pooled.size else None,
            "v_mean": float(pooled[:, 2].mean()) if pooled.size else None,
            "v_std": float(pooled[:, 2].std()) if pooled.size else None,
            "n_pixels": int(pooled.shape[0]),
        },
    }


# ---------------------------------------------------------------------------
# 相関・レポート
# ---------------------------------------------------------------------------
def _load_flicker_metrics() -> dict[str, dict[str, Any]]:
    if not FLICKER_JSON.exists():
        return {}
    with FLICKER_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_video_proxy_66", {})


def _load_baseline_reset_metrics() -> dict[str, dict[str, Any]]:
    if not VIDEO_DIFFICULTY_CSV.exists():
        return {}
    import csv as _csv
    with VIDEO_DIFFICULTY_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.DictReader(f))
    return {r["video_stem"]: r for r in rows}


def _load_cell_acc_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    with WITH51_JSON.open(encoding="utf-8") as f:
        with51 = json.load(f)["per_video"]
    with CURRENT_DEFAULT_JSON.open(encoding="utf-8") as f:
        default = json.load(f)["per_video"]
    return with51, default


def _report_correlations(c_results: dict[str, Any], v_results: dict[str, Any]) -> dict[str, Any]:
    """色調指標と既存認識品質指標の相関 (層別・プールなし)。"""
    flicker = _load_flicker_metrics()
    reset_metrics = _load_baseline_reset_metrics()
    with51_acc, default_acc = _load_cell_acc_metrics()

    out: dict[str, Any] = {"c_series": {}, "v_series": {}}

    metric_keys = ["h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std"]
    c_stems = [s for s in c_results if "error" not in c_results[s]]
    for mk in metric_keys:
        xs = [c_results[s]["region_hsv"][mk] for s in c_stems]
        ys_flicker = [flicker.get(s, {}).get("flicker_rate") for s in c_stems]
        ys_reset = [
            float(reset_metrics[s]["baseline_reset_lines"]) if s in reset_metrics else None
            for s in c_stems
        ]
        r1, n1 = _pearson(xs, ys_flicker)
        r2, n2 = _pearson(xs, ys_reset)
        out["c_series"][mk] = {
            "corr_vs_flicker_rate": r1, "n_flicker": n1,
            "corr_vs_baseline_reset_lines": r2, "n_reset": n2,
        }

    v_names = [s for s in v_results if "error" not in v_results[s]]
    for mk in metric_keys:
        xs = [v_results[s]["region_hsv"][mk] for s in v_names]
        ys_acc_default = [default_acc.get(s, {}).get("acc") for s in v_names]
        ys_acc_with51 = [with51_acc.get(s, {}).get("acc") for s in v_names]
        r1, n1 = _pearson(xs, ys_acc_default)
        r2, n2 = _pearson(xs, ys_acc_with51)
        out["v_series"][mk] = {
            "corr_vs_current_default_acc": r1, "n_default": n1,
            "corr_vs_with51_acc": r2, "n_with51": n2,
        }
    return out


def main() -> None:
    print(f"=== セクションA: 66動画 (c系) 色調実測 開始 (n={len(C_SERIES_STEMS)}) ===")
    c_results: dict[str, Any] = {}
    for i, stem in enumerate(C_SERIES_STEMS, start=1):
        c_results[stem] = _process_c_series_video(stem)
        print(f"[{i}/{len(C_SERIES_STEMS)}] {stem} done")

    print(f"\n=== セクションB: 19動画 (v系評価クリップ) 色調実測 開始 (n={len(V_SERIES_FILE_MAP)}) ===")
    v_results: dict[str, Any] = {}
    for i, (name, rel_path) in enumerate(V_SERIES_FILE_MAP.items(), start=1):
        v_results[name] = _process_v_series_video(rel_path)
        print(f"[{i}/{len(V_SERIES_FILE_MAP)}] {name} done")

    print("\n=== 相関計算 ===")
    corr = _report_correlations(c_results, v_results)

    report = {"c_series": c_results, "v_series": v_results, "correlations": corr}
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {OUT_JSON}")

    print("\n=== c系 相関サマリ (色調 vs ちらつき率 / baseline_reset行数) ===")
    for mk, v in corr["c_series"].items():
        print(f"{mk:8s} vs flicker_rate: r={v['corr_vs_flicker_rate']} (n={v['n_flicker']})"
              f"  vs baseline_reset_lines: r={v['corr_vs_baseline_reset_lines']} (n={v['n_reset']})")

    print("\n=== v系 相関サマリ (色調 vs セル正解率) ===")
    for mk, v in corr["v_series"].items():
        print(f"{mk:8s} vs current_default_acc: r={v['corr_vs_current_default_acc']} (n={v['n_default']})"
              f"  vs with51_acc: r={v['corr_vs_with51_acc']} (n={v['n_with51']})")


if __name__ == "__main__":
    main()
