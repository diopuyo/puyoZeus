"""ユーザがラベル付けしたセルから予告お邪魔ぷよのテンプレを生成する。

入力:
    data/verify/ojama_labels.tsv   (frame_idx, side, cell_idx, label)
    data/verify/ojama_label_index.tsv  (frame_idx, t_sec, video, match, side, cell_idx)
    data/training/score_series_cache.json  (動画パス特定用)

処理:
    各ラベル付きセル画像を集計 → 種類ごとに 中央 36×36 を平均化 → テンプレ保存

出力:
    models/ui_templates/ojama/<class>.png  (rock, moon, small, large, star, crown)
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.ojama_warning import (
    BOARD_WIDTH,
    CELL_COUNT,
    CELL_WIDTH,
    ICON_SAMPLE_HALF,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_HEIGHT,
    WARNING_TOP_Y,
)

# 複数バージョンのラベル+インデックスを順番に読み込む (frame_idx は重複可だが
# (label_set, frame_idx, side, cell_idx) で一意になるよう内部で扱う)
LABEL_SETS: list[tuple[Path, Path]] = [
    (Path("data/verify/ojama_labels.tsv"),
     Path("data/verify/ojama_label_index.tsv")),
    (Path("data/verify/ojama_labels_v2.tsv"),
     Path("data/verify/ojama_label_index_v2.tsv")),
]
TEMPLATE_DIR = Path("models/ui_templates/ojama")
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)
SAMPLE_SAVE_DIR = Path("data/verify/ojama_label_samples")  # 確認用

# 中央 36×36 でテンプレ化 (ICON_SAMPLE_HALF=18)
TEMPLATE_HALF: int = ICON_SAMPLE_HALF  # 18
# テンプレートサイズ (合致するように)
TEMPLATE_HEIGHT: int = 2 * TEMPLATE_HALF  # 36
TEMPLATE_WIDTH: int = 2 * TEMPLATE_HALF   # 36

VALID_LABELS = ("small", "large", "rock", "star", "moon", "crown")

# ユーザラベル → 実装側 ICON_* テンプレファイル名のマッピング
LABEL_TO_FILENAME: dict[str, str] = {
    "small": "small",
    "large": "line",       # 大ぷよ (= 6 個)
    "rock": "rock",
    "star": "big_crown",   # 星ぷよ (= 180 個)
    "moon": "moon",
    "crown": "crown",
}


def get_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != EXPECTED_FRAME_SHAPE:
        frame = cv2.resize(
            frame,
            (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def extract_center_patch(
    frame: np.ndarray, side: str, cell_idx: int
) -> np.ndarray:
    """セル中央 36×36 を切り出す。"""
    base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
    cell_x1 = base_x + cell_idx * CELL_WIDTH
    cell_cx = cell_x1 + CELL_WIDTH // 2
    cell_cy = WARNING_TOP_Y + WARNING_HEIGHT // 2
    return frame[
        cell_cy - TEMPLATE_HALF: cell_cy + TEMPLATE_HALF,
        cell_cx - TEMPLATE_HALF: cell_cx + TEMPLATE_HALF,
    ].copy()


def main() -> int:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 複数ラベルセットを読み込み (set_id, frame_idx, side, cell_idx) で識別
    bucket: dict[str, list[np.ndarray]] = defaultdict(list)
    frame_cache: dict[tuple[str, float], np.ndarray | None] = {}
    skipped = 0
    total_labels = 0

    for set_id, (labels_path, index_path) in enumerate(LABEL_SETS):
        if not labels_path.is_file() or not index_path.is_file():
            print(f"  [skip set {set_id}] {labels_path} or {index_path} 不在")
            continue
        idx: dict[tuple[int, str, int], tuple[float, str]] = {}
        with open(index_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (int(r["frame_idx"]), r["side"], int(r["cell_idx"]))
                idx[key] = (float(r["t_sec"]), r["video"])
        labels: list[tuple[int, str, int, str]] = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                labels.append(
                    (int(r["frame_idx"]), r["side"],
                     int(r["cell_idx"]), r["label"])
                )
        print(f"  [set {set_id}] index={len(idx)} labels={len(labels)}")
        total_labels += len(labels)

        for frame_idx, side, cell_idx, label in labels:
            if label not in VALID_LABELS:
                continue
            info = idx.get((frame_idx, side, cell_idx))
            if info is None:
                skipped += 1
                continue
            t_sec, vid = info
            cache_key = (vid, t_sec)
            if cache_key not in frame_cache:
                video_path = Path(f"data/frames/{vid}.mp4")
                frame_cache[cache_key] = get_frame(video_path, t_sec)
            frame = frame_cache[cache_key]
            if frame is None:
                skipped += 1
                continue
            patch = extract_center_patch(frame, side, cell_idx)
            if patch.shape[:2] != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
                skipped += 1
                continue
            bucket[label].append(patch)
            sample_path = (SAMPLE_SAVE_DIR /
                           f"set{set_id}_{label}_F{frame_idx}_"
                           f"{side}_S{cell_idx}.png")
            cv2.imwrite(str(sample_path), patch)

    print(f"total_labels: {total_labels}, skipped: {skipped}")
    print()

    # 既存テンプレファイルをクリア (旧 K=1 の上書き or rename を残さないため)
    for fname in TEMPLATE_DIR.glob("*.png"):
        fname.unlink()

    # 平均化テンプレ保存 (K=1 でシンプル、外れ値あればクラスタリングで除外)
    K_CLUSTERS: int = 1
    for label, patches in sorted(bucket.items()):
        if not patches:
            print(f"  [{label}] サンプルなし")
            continue
        fname_base = LABEL_TO_FILENAME.get(label, label)
        n = len(patches)
        # サンプル数が K 未満なら単一平均
        if n < K_CLUSTERS * 2:
            avg = np.mean(np.stack([p.astype(np.float32) for p in patches]),
                          axis=0)
            avg_u8 = np.clip(avg, 0, 255).astype(np.uint8)
            out_path = TEMPLATE_DIR / f"{fname_base}.png"
            cv2.imwrite(str(out_path), avg_u8)
            print(f"  [{label}→{fname_base}] {n} samples (single) → {out_path}")
            continue
        # 各サンプルを 1D ベクトルにして KMeans
        flat = np.stack([p.astype(np.float32).flatten() for p in patches])
        # シンプル KMeans (cv2.kmeans)
        compactness, labels_arr, centers = cv2.kmeans(
            flat,
            K_CLUSTERS,
            None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0),
            attempts=5,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        labels_arr = labels_arr.flatten()
        # 外れ値クラスタ破棄: 全体の 25% 未満 or 3 サンプル未満のクラスタは破棄
        MIN_CLUSTER_RATIO: float = 0.25
        MIN_CLUSTER_SAMPLES: int = 3
        min_size = max(MIN_CLUSTER_SAMPLES, int(n * MIN_CLUSTER_RATIO))
        kept_clusters: list[tuple[int, list]] = []
        for k in range(K_CLUSTERS):
            mask = labels_arr == k
            cluster_patches = [p for p, m in zip(patches, mask) if m]
            if len(cluster_patches) >= min_size:
                kept_clusters.append((k, cluster_patches))
        # クラスタが 1 つしか残らない or 全て大きい → suffix 付け直し
        if len(kept_clusters) == 0:
            # フォールバック: 単一平均
            avg = np.mean(np.stack([p.astype(np.float32) for p in patches]),
                          axis=0)
            avg_u8 = np.clip(avg, 0, 255).astype(np.uint8)
            out_path = TEMPLATE_DIR / f"{fname_base}.png"
            cv2.imwrite(str(out_path), avg_u8)
            print(f"  [{label}→{fname_base}] {n} samples (fallback) → {out_path}")
            continue
        for new_k, (orig_k, cluster_patches) in enumerate(kept_clusters):
            avg = np.mean(
                np.stack([p.astype(np.float32) for p in cluster_patches]),
                axis=0,
            )
            avg_u8 = np.clip(avg, 0, 255).astype(np.uint8)
            suffix = "" if new_k == 0 else f"_{new_k + 1}"
            out_path = TEMPLATE_DIR / f"{fname_base}{suffix}.png"
            cv2.imwrite(str(out_path), avg_u8)
            print(f"  [{label}→{fname_base}{suffix}] "
                  f"{len(cluster_patches)} samples "
                  f"(orig cluster {orig_k}, kept) → {out_path}")
        if len(kept_clusters) < K_CLUSTERS:
            dropped = K_CLUSTERS - len(kept_clusters)
            print(f"    (外れ値 {dropped} クラスタを破棄)")

    print(f"\nテンプレ出力先: {TEMPLATE_DIR}")
    print(f"個別サンプル: {SAMPLE_SAVE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
