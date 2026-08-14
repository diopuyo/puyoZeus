"""物差しラベル (52盤面3744セル) の自動再アンカリング (タスク#5、2026-08-14)。

## 背景 (測定器事故)
人手ラベル (`board_labels_2026-07-31` v1/v3) は絶対フレーム番号で元動画に
紐づくが、元動画はストレージポリシーで削除済み。YouTubeから再DLした動画
(WSL `~/frames/video_cXX.mp4`) は再エンコード世代差で fps/解像度が変わり
(実測: c10/c17/c20/c21 は 1280x720、c16/c18 は 30fps、他は 60fps 系)、
同じ frame_idx が同じ画面を指さない。これに気付かず突合すると、実際は
無関係な2つの盤面を比較してしまい、認識の良し悪しと無関係な偽の誤りセルが
発生する (=測定器事故)。

## 方式 (循環なし)
認識パイプラインは一切使わない。人手ラベルPNGの左半分 (実画面をそのまま
切り出したもの、`_build_board_label_sheets_2026-07-31.py` の `_crop_board`
出力) を「参照画像」として、再DL動画の対応時刻 ±SEARCH_WINDOW_SEC 秒を
生フレームでスキャンし、ピクセル一致度 (NCC = 正規化相互相関) が最大の
フレームを「再アンカリングされた真の時刻」とする。

参照画像のジオメトリは実測で確定させた (推測禁止):
  - PNG は [実画面クロップ(拡大1.6x) | 24px黒ギャップ | 認識結果グリッド] の
    横結合。ギャップは列平均が厳密に0のピクセル列として自動検出できる。
  - 実画面クロップは `DEFAULT_P1_REGION`/`DEFAULT_P2_REGION`
    (x=282/1258, y=160, w=384, h=720、`src/image_reader.py`) をそのまま
    1.6倍拡大したもの (614x1152 == 384x1152/1.6 ... 実測 614/384=1.600,
    1152/720=1.600で確定)。

参照時刻 (再DL動画側でどこを中心に探すか) は、ラベルの元 frame_idx を
`scripts/_measure_yardstick_v4_2026-08-05.py` の `_reconstruct_correct_grid`
で正解グリッドと一緒に返る `t_sec` を使う。これは 2026-07-30 に**元動画が
健在だった時点**で収集された baseline npz
(`data/indicators_v2/boards_lean_allframes_ref_2026-07-30`) の値であり、
削除前の元動画タイムラインに正しく紐づいている (この値自体は再DL動画の
drift の影響を受けない)。

## 出力
`data/verify/yardstick_reanchor_2026-08-14/reanchor_table.tsv`:
  video, side, frame_idx, source_dir, label_t_original, best_t_redl,
  drift_sec, best_score, score_at_zero_drift, decision, reason

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._reanchor_yardstick_labels_2026-08-14
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

# ファイル名にハイフンを含むため動的import (既存規約、_measure_yardstick_v4_2026-08-05.py 準拠)
_MY = importlib.import_module("scripts._measure_yardstick_v4_2026-08-05")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

# 再DL動画側の探索窓 (ラベルの元 t_sec を中心に ±この秒数)。タスク#5指示準拠。
SEARCH_WINDOW_SEC: float = 10.0

# ラベルPNG生成時の実画面クロップ拡大率。_build_board_label_sheets_2026-07-31.py
# の定数 CROP_SCALE=2.0 は生成後に変更された可能性があるため、実測値
# (614/384=1.600, 1152/720=1.600、004_video_c12_1P_f80803.png で確認) を正とする。
LABEL_CROP_SCALE: float = 1.6

# NCC (cv2.TM_CCOEFF_NORMED) の判定閾値。
# 高: 「明確な一致」とみなし再アンカリング成功として採用する。
NCC_CONFIDENT_THRESHOLD: float = 0.85
# 低: 探索窓内の最良スコアがこれ未満なら「アンカー不能」として除外する
# (中間帯 [NCC_UNANCHORABLE_THRESHOLD, NCC_CONFIDENT_THRESHOLD) も安全側で不能扱い、
# fail-silent 回避のため grayzone を許容しない)。
NCC_UNANCHORABLE_THRESHOLD: float = 0.85

# 標準ジオメトリ (認識パイプラインが前提とする解像度)
STD_WIDTH: int = 1920
STD_HEIGHT: int = 1080

VIDEO_DIR_WSL: Path = Path.home() / "frames"
LABEL_PNG_DIRS: "dict[str, Path]" = {d.name: d for d in _MY.LABEL_DIRS}

OUT_DIR: Path = Path("data/verify/yardstick_reanchor_2026-08-14")
OUT_TABLE: Path = OUT_DIR / "reanchor_table.tsv"


# =============================================================================
# データ構造
# =============================================================================


@dataclass(frozen=True)
class ReanchorResult:
    """1ラベル分の再アンカリング結果。"""

    video_stem: str
    side: str
    frame_idx: int
    source_dir: str
    label_t_original: float
    best_t_redl: float
    drift_sec: float
    best_score: float
    score_at_zero_drift: float
    decision: str  # "anchored" / "unanchorable"
    reason: str


# =============================================================================
# 1. 参照画像の抽出 (ラベルPNG左半分 → ネイティブ解像度グレースケール)
# =============================================================================


def find_label_png(video_stem: str, side: str, frame_idx: int, source_dir: str) -> "Path | None":
    """labels.tsv の1行に対応するPNGファイルを探す。"""
    label_dir = LABEL_PNG_DIRS.get(source_dir)
    if label_dir is None:
        return None
    pattern = f"*_video_{video_stem}_{side}_f{frame_idx}.png"
    matches = sorted(label_dir.glob(pattern))
    return matches[0] if matches else None


def extract_reference_crop(png_path: Path) -> np.ndarray:
    """ラベルPNGの左半分 (実画面切り出し) をネイティブ解像度・グレースケールで返す。

    PNG は [実画面クロップ|24px黒ギャップ|認識グリッド] の横結合。ギャップは
    列平均が厳密に0のピクセル列として検出する (推測でなく実データから決定)。
    """
    img = cv2.imread(str(png_path))
    if img is None:
        raise ValueError(f"読込失敗: {png_path}")
    col_means = img.astype(np.float64).mean(axis=(0, 2))
    zero_cols = np.where(col_means == 0.0)[0]
    if len(zero_cols) == 0:
        raise ValueError(f"ギャップ列検出失敗 (想定と異なる画像形式): {png_path}")
    gap_start = int(zero_cols[0])
    left = img[:, :gap_start]
    native_w = max(1, round(left.shape[1] / LABEL_CROP_SCALE))
    native_h = max(1, round(left.shape[0] / LABEL_CROP_SCALE))
    native = cv2.resize(left, (native_w, native_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(native, cv2.COLOR_BGR2GRAY)


def region_for_side(side: str) -> "tuple[int, int, int, int]":
    """side に対応する盤面ROI (x, y, w, h) を返す (標準1920x1080前提)。"""
    reg = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    return reg.x, reg.y, reg.width, reg.height


# =============================================================================
# 2. 動画スキャン (ピクセル照合、認識は使わない)
# =============================================================================


def video_props(path: Path) -> "tuple[float, int, int, float]":
    """(fps, width, height, duration_sec) を返す。"""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n / fps if fps > 0 else 0.0
    return fps, w, h, duration


def _crop_gray(frame: np.ndarray, region: "tuple[int, int, int, int]") -> np.ndarray:
    """標準解像度へ正規化してから ROI を切り出し、グレースケール化する。"""
    if frame.shape[:2] != (STD_HEIGHT, STD_WIDTH):
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
    x, y, w, h = region
    crop = frame[y : y + h, x : x + w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def scan_video_window(
    video_path: Path, region: "tuple[int, int, int, int]", ref_gray: np.ndarray,
    center_t: float, window_sec: float,
) -> "tuple[float, float, float]":
    """center_t ± window_sec を生フレームでスキャンし、

    (best_t, best_score, score_at_zero_drift) を返す。
    score は cv2.TM_CCOEFF_NORMED (NCC、範囲おおよそ [-1, 1])。
    """
    fps, _w, _h, duration = video_props(video_path)
    if fps <= 0:
        return center_t, -2.0, float("nan")
    t0 = max(0.0, center_t - window_sec)
    t1 = min(duration, center_t + window_sec)
    n_frames = max(1, int(round((t1 - t0) * fps)))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(round(t0 * fps)))
    best_score = -2.0
    best_t = t0
    score_zero = float("nan")
    half_frame_sec = 0.5 / fps
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t_actual = t0 + i / fps
        gray = _crop_gray(frame, region)
        if gray.shape != ref_gray.shape:
            gray = cv2.resize(gray, (ref_gray.shape[1], ref_gray.shape[0]))
        res = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0])
        if score > best_score:
            best_score = score
            best_t = t_actual
        if abs(t_actual - center_t) < half_frame_sec:
            score_zero = score
    cap.release()
    return best_t, best_score, score_zero


# =============================================================================
# 3. 1ラベル分の再アンカリング
# =============================================================================


def reanchor_one(row: "_MY.YardstickRow") -> ReanchorResult:
    """1ラベル行を再アンカリングする (失敗は明示、黙って落とさない)。"""
    rec = _MY._reconstruct_correct_grid(row, _MY.BASELINE_NPZ_DIR)
    if rec is None:
        return ReanchorResult(
            row.video_stem, row.side, row.frame_idx, row.source_dir,
            float("nan"), float("nan"), float("nan"), 0.0, float("nan"),
            "unanchorable", "baseline npz に該当frame_idxが無い (正解再構成失敗)",
        )
    _correct_grid, label_t = rec
    png = find_label_png(row.video_stem, row.side, row.frame_idx, row.source_dir)
    if png is None:
        return ReanchorResult(
            row.video_stem, row.side, row.frame_idx, row.source_dir,
            label_t, float("nan"), float("nan"), 0.0, float("nan"),
            "unanchorable", "ラベルPNGが見つからない",
        )
    video_path = VIDEO_DIR_WSL / f"video_{row.video_stem}.mp4"
    if not video_path.exists():
        return ReanchorResult(
            row.video_stem, row.side, row.frame_idx, row.source_dir,
            label_t, float("nan"), float("nan"), 0.0, float("nan"),
            "unanchorable", f"再DL動画が無い: {video_path}",
        )
    ref_gray = extract_reference_crop(png)
    region = region_for_side(row.side)
    best_t, best_score, score_zero = scan_video_window(
        video_path, region, ref_gray, label_t, SEARCH_WINDOW_SEC,
    )
    drift = best_t - label_t
    if best_score >= NCC_CONFIDENT_THRESHOLD:
        decision, reason = "anchored", ""
    else:
        decision = "unanchorable"
        reason = f"探索窓内の最良NCC={best_score:.3f} < 閾値{NCC_CONFIDENT_THRESHOLD}"
    return ReanchorResult(
        row.video_stem, row.side, row.frame_idx, row.source_dir,
        label_t, best_t, drift, best_score, score_zero, decision, reason,
    )


# =============================================================================
# 4. main
# =============================================================================


def main() -> None:
    rows = _MY.load_yardstick_rows()
    print(f"[1/2] labels.tsv読込: {len(rows)}盤面 (v1+v3統合)")
    results: list[ReanchorResult] = []
    for i, row in enumerate(rows):
        r = reanchor_one(row)
        results.append(r)
        print(
            f"  [{i + 1}/{len(rows)}] {r.video_stem} {r.side} f{r.frame_idx} "
            f"[{r.source_dir}]: drift={r.drift_sec:+.3f}s score={r.best_score:.3f} "
            f"→ {r.decision} {r.reason}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "video", "side", "frame_idx", "source_dir", "label_t_original",
            "best_t_redl", "drift_sec", "best_score", "score_at_zero_drift",
            "decision", "reason",
        ])
        for r in results:
            w.writerow([
                r.video_stem, r.side, r.frame_idx, r.source_dir,
                f"{r.label_t_original:.3f}", f"{r.best_t_redl:.3f}",
                f"{r.drift_sec:+.3f}", f"{r.best_score:.4f}",
                f"{r.score_at_zero_drift:.4f}", r.decision, r.reason,
            ])
    n_anchored = sum(1 for r in results if r.decision == "anchored")
    n_unanchorable = len(results) - n_anchored
    print(f"[2/2] 出力: {OUT_TABLE}")
    print(f"アンカー成功: {n_anchored}/{len(results)} / アンカー不能: {n_unanchorable}/{len(results)}")
    if n_unanchorable:
        print("アンカー不能一覧:")
        for r in results:
            if r.decision == "unanchorable":
                print(f"  {r.video_stem} {r.side} f{r.frame_idx} [{r.source_dir}]: {r.reason}")


if __name__ == "__main__":
    main()
