"""
パッチ抽出パイプライン

動画から均一間隔でフレームをサンプリングし、キャリブレーション済み
盤面座標で各セルのパッチを切り出して HSV 分類器で擬似ラベル化する。

学習データセット (パッチ + ラベル) を npz で保存し、
MlpPatchClassifier / CnnPatchClassifier の学習に使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion, ColorClassifier

# ============================
# 定数定義
# ============================

# サンプリング間隔 (秒)
DEFAULT_SAMPLE_INTERVAL_SEC: float = 5.0

# パッチの正規化サイズ (CNN 入力と合わせる)
DEFAULT_PATCH_SIZE: int = 32

# クラス重み付けサンプリング時、空セルの保持上限比率
EMPTY_CLASS_CAP_RATIO: float = 0.35

# 保存フォーマット識別子
PATCH_DATASET_FORMAT: str = "puyo_patches_v1"

# side (プレイヤー側) コード
SIDE_UNKNOWN: int = 0  # 旧 npz ロード時や side 不明パッチ
SIDE_1P: int = 1
SIDE_2P: int = 2


# ============================
# データクラス
# ============================


@dataclass
class ExtractionStats:
    """抽出処理の統計情報。"""
    frames_sampled: int = 0
    patches_total: int = 0
    per_class_count: dict[int, int] = field(default_factory=dict)


@dataclass
class PatchDataset:
    """
    パッチ + ラベルのコレクション。

    Attributes:
        patches: shape=(N, H, W, 3) uint8 の BGR パッチ配列。
        labels: shape=(N,) int の色コード配列。
        stats: 抽出統計。
        sides: shape=(N,) int8 のプレイヤー側メタ配列。
               0=unknown, 1=1P, 2=2P。
               後方互換のため省略可 (その場合 zeros で埋める)。
    """
    patches: np.ndarray
    labels: np.ndarray
    stats: ExtractionStats = field(default_factory=ExtractionStats)
    sides: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.patches) != len(self.labels):
            raise ValueError(
                f"patches/labels 件数不一致: "
                f"{len(self.patches)} vs {len(self.labels)}"
            )
        # sides 未指定 or 長さ不一致なら全 unknown で埋める (後方互換)
        if self.sides is None:
            self.sides = np.zeros(len(self.labels), dtype=np.int8)
        elif len(self.sides) != len(self.labels):
            raise ValueError(
                f"sides 件数不一致: {len(self.sides)} vs {len(self.labels)}"
            )
        else:
            # dtype 正規化
            self.sides = np.asarray(self.sides, dtype=np.int8)

    # ============================
    # 永続化
    # ============================

    def save(self, path: Path) -> None:
        """npz で保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format=np.array([PATCH_DATASET_FORMAT]),
            patches=self.patches,
            labels=self.labels,
            sides=self.sides,
            frames_sampled=np.array([self.stats.frames_sampled]),
        )

    @classmethod
    def load(cls, path: Path) -> "PatchDataset":
        """
        npz から復元する。

        旧 npz (sides キー無し) は全 unknown (0) で埋めて読み込む。
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path)
        fmt = str(data["format"][0]) if "format" in data else ""
        if fmt != PATCH_DATASET_FORMAT:
            raise ValueError(f"フォーマット不一致: {fmt}")
        stats = ExtractionStats(
            frames_sampled=int(data.get(
                "frames_sampled", np.array([0]),
            )[0]),
        )
        patches = data["patches"]
        labels = data["labels"]
        # sides は旧 npz には無い → zeros で後方互換
        if "sides" in data.files:
            sides = np.asarray(data["sides"], dtype=np.int8)
        else:
            sides = np.zeros(len(labels), dtype=np.int8)
        ds = cls(
            patches=patches,
            labels=labels,
            stats=stats,
            sides=sides,
        )
        # per_class_count を再計算
        ds.stats.patches_total = len(ds.labels)
        unique, counts = np.unique(ds.labels, return_counts=True)
        ds.stats.per_class_count = {
            int(k): int(v) for k, v in zip(unique, counts)
        }
        return ds


# ============================
# PatchExtractor
# ============================


class PatchExtractor:
    """
    動画から学習用パッチを抽出するクラス。

    Usage:
        config = CalibratedConfig.load("models/calibration_video01.json")
        extractor = PatchExtractor(config=config)
        dataset = extractor.extract_from_video("data/frames/video_01.mp4")
        dataset.save("data/training/video01_patches.npz")
    """

    def __init__(
        self,
        config: CalibratedConfig,
        classifier: ColorClassifier | None = None,
        patch_size: int = DEFAULT_PATCH_SIZE,
    ) -> None:
        """
        Args:
            config: キャリブレーション設定 (座標・色閾値)。
            classifier: 擬似ラベル生成に使う HSV 分類器。None ならデフォルト。
            patch_size: 保存時のパッチ辺長 (px)。
        """
        self._config = config
        self._classifier = classifier or ColorClassifier(
            color_ranges=config.color_ranges or None,
        )
        self._patch_size = patch_size

    # ============================
    # 公開メソッド
    # ============================

    def extract_from_video(
        self,
        video_path: Path | str,
        sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
        max_frames: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PatchDataset:
        """
        動画全体からパッチを抽出する。

        Args:
            video_path: 動画パス。
            sample_interval_sec: サンプリング間隔 (秒)。
            max_frames: 最大抽出フレーム数 (デバッグ用、None なら制限なし)。
            progress_callback: 進捗コールバック。

        Returns:
            PatchDataset: 抽出結果。

        Raises:
            FileNotFoundError / RuntimeError: 動画が開けない場合。
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画なし: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        try:
            return self._run_extraction(
                cap, sample_interval_sec, max_frames, progress_callback,
            )
        finally:
            cap.release()

    def extract_from_frame(
        self, frame: np.ndarray,
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        単一フレームから 2 盤面 × 可視セルのパッチとラベルを返す。

        隠し段 (row < HIDDEN_ROWS) は画面外なので抽出しない。
        抽出数は: 2 盤面 × VISIBLE_ROWS(=12) × BOARD_COLS(=6) = 144 セル。

        Args:
            frame: BGR フレーム。

        Returns:
            tuple[list[np.ndarray], list[int]]: (パッチリスト, ラベルリスト)。

        Notes:
            side 情報が必要な場合は extract_from_frame_with_sides を使う。
            後方互換のため本メソッドの返り値シグネチャは変更していない。
        """
        patches, labels, _sides = self.extract_from_frame_with_sides(frame)
        return patches, labels

    def extract_from_frame_with_sides(
        self, frame: np.ndarray,
    ) -> tuple[list[np.ndarray], list[int], list[int]]:
        """
        単一フレームから (パッチ, ラベル, side) の 3 要素 tuple を返す。

        side は各パッチが 1P/2P どちらの盤面から取られたかを示す
        (SIDE_1P=1 / SIDE_2P=2)。抽出順は 1P → 2P 固定。

        Args:
            frame: BGR フレーム。

        Returns:
            tuple[list[np.ndarray], list[int], list[int]]:
                (パッチリスト, ラベルリスト, side リスト)。
        """
        patches: list[np.ndarray] = []
        labels: list[int] = []
        sides: list[int] = []
        region_side_pairs = (
            (self._config.p1_region, SIDE_1P),
            (self._config.p2_region, SIDE_2P),
        )
        for region, side_code in region_side_pairs:
            for row in range(HIDDEN_ROWS, BOARD_ROWS):
                for col in range(BOARD_COLS):
                    patch = self._crop_cell(frame, region, row, col)
                    if patch.size == 0:
                        continue
                    label = self._classifier.classify(patch)
                    patches.append(self._resize_patch(patch))
                    labels.append(label)
                    sides.append(side_code)
        return patches, labels, sides

    # ============================
    # 内部メソッド
    # ============================

    def _run_extraction(
        self,
        cap: cv2.VideoCapture,
        interval_sec: float,
        max_frames: int | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> PatchDataset:
        """動画 capture から抽出処理を実行する。"""
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_frames = max(1, int(round(fps * interval_sec)))

        all_patches: list[np.ndarray] = []
        all_labels: list[int] = []
        all_sides: list[int] = []
        frames_done = 0
        sample_count = 0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step_frames == 0:
                patches, labels, sides = self.extract_from_frame_with_sides(
                    frame,
                )
                all_patches.extend(patches)
                all_labels.extend(labels)
                all_sides.extend(sides)
                sample_count += 1
                if progress_callback is not None:
                    progress_callback({
                        "frame_index": frame_index,
                        "sample_count": sample_count,
                        "total_frames": total_frames,
                        "patches_so_far": len(all_patches),
                    })
                if max_frames is not None and sample_count >= max_frames:
                    break
            frame_index += 1
            frames_done += 1

        if not all_patches:
            patches_arr = np.zeros(
                (0, self._patch_size, self._patch_size, 3), dtype=np.uint8,
            )
            labels_arr = np.zeros(0, dtype=np.int64)
            sides_arr = np.zeros(0, dtype=np.int8)
        else:
            patches_arr = np.stack(all_patches)
            labels_arr = np.array(all_labels, dtype=np.int64)
            sides_arr = np.array(all_sides, dtype=np.int8)

        stats = ExtractionStats(
            frames_sampled=sample_count,
            patches_total=len(all_labels),
            per_class_count={
                int(k): int(v)
                for k, v in zip(*np.unique(labels_arr, return_counts=True))
            },
        )
        return PatchDataset(
            patches=patches_arr, labels=labels_arr, stats=stats,
            sides=sides_arr,
        )

    @staticmethod
    def _crop_cell(
        frame: np.ndarray, region: BoardRegion, row: int, col: int,
    ) -> np.ndarray:
        """セル位置のパッチ (フル cell サイズ) を切り出す。"""
        h, w = frame.shape[:2]
        cx, cy = region.cell_center(row, col)
        half_w = int(region.cell_width / 2)
        half_h = int(region.cell_height / 2)
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w)
        y1 = max(0, cy - half_h)
        y2 = min(h, cy + half_h)
        return frame[y1:y2, x1:x2]

    def _resize_patch(self, patch: np.ndarray) -> np.ndarray:
        """パッチを patch_size に正規化する。"""
        return cv2.resize(
            patch,
            (self._patch_size, self._patch_size),
            interpolation=cv2.INTER_AREA,
        )


# ============================
# バランス調整
# ============================


def balance_dataset(
    dataset: PatchDataset,
    empty_ratio_cap: float = EMPTY_CLASS_CAP_RATIO,
    color_balance_factor: float = 2.0,
    seed: int = 0,
    stratify_by_side: bool = False,
) -> PatchDataset:
    """
    クラス偏り (空セル過多 + 色間不均衡) をサブサンプリングで緩和する。

    1. 空セルが全体の empty_ratio_cap を超えないように削減する。
    2. 色クラス間で最大クラスが最小クラスの color_balance_factor 倍以内に
       なるようダウンサンプリングする (少数クラスの学習効果を改善)。
    3. stratify_by_side=True の場合、空セル削減および色クラス均等化を
       side×label のバケット単位で行う。side=0 (unknown) のパッチは
       1 個のバケットにまとめ、旧 npz と新 npz の混在でも壊れないようにする。

    Args:
        dataset: 元データセット。
        empty_ratio_cap: 空セルの最大割合。
        color_balance_factor: 色クラス間の最大倍率 (2.0 = 最大が最小の2倍まで)。
        seed: 乱数シード。
        stratify_by_side: True で side×label 層別バランス。
                         False (デフォルト) は従来動作。

    Returns:
        PatchDataset: バランス調整後の新データセット。
    """
    if stratify_by_side:
        return _balance_by_side(
            dataset, empty_ratio_cap, color_balance_factor, seed,
        )
    return _balance_global(
        dataset, empty_ratio_cap, color_balance_factor, seed,
    )


def _balance_global(
    dataset: PatchDataset,
    empty_ratio_cap: float,
    color_balance_factor: float,
    seed: int,
) -> PatchDataset:
    """全体を一括バランスする (従来挙動)。"""
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    patches = dataset.patches
    sides = dataset.sides

    # Step 1: 空セルキャップ
    empty_idx = np.where(labels == COLOR_EMPTY)[0]
    non_empty_idx = np.where(labels != COLOR_EMPTY)[0]

    max_empty = int(
        len(non_empty_idx) * empty_ratio_cap / (1.0 - empty_ratio_cap)
    )
    if len(empty_idx) > max_empty:
        kept_empty = rng.choice(empty_idx, size=max_empty, replace=False)
    else:
        kept_empty = empty_idx

    # Step 2: 色クラス間バランス (空以外)
    color_indices: dict[int, np.ndarray] = {}
    for idx in non_empty_idx:
        lbl = int(labels[idx])
        if lbl not in color_indices:
            color_indices[lbl] = []
        color_indices[lbl].append(idx)
    for k in color_indices:
        color_indices[k] = np.array(color_indices[k])

    if color_indices:
        min_count = min(len(v) for v in color_indices.values())
        max_allowed = int(min_count * color_balance_factor)
        balanced_non_empty = []
        for lbl, idxs in color_indices.items():
            if len(idxs) > max_allowed:
                idxs = rng.choice(idxs, size=max_allowed, replace=False)
            balanced_non_empty.append(idxs)
        keep_idx = np.concatenate([kept_empty, *balanced_non_empty])
    else:
        keep_idx = kept_empty

    rng.shuffle(keep_idx)
    new_patches = patches[keep_idx]
    new_labels = labels[keep_idx]
    new_sides = sides[keep_idx] if sides is not None else None

    stats = ExtractionStats(
        frames_sampled=dataset.stats.frames_sampled,
        patches_total=len(new_labels),
        per_class_count={
            int(k): int(v)
            for k, v in zip(*np.unique(new_labels, return_counts=True))
        },
    )
    return PatchDataset(
        patches=new_patches, labels=new_labels, stats=stats,
        sides=new_sides,
    )


def _balance_by_side(
    dataset: PatchDataset,
    empty_ratio_cap: float,
    color_balance_factor: float,
    seed: int,
) -> PatchDataset:
    """
    side×label で層別バランスする。

    - side=0 (unknown) は 1 つのバケットにまとめる (旧 npz 互換)。
    - 各 side について独立に「空セル上限」「色クラス均等化」を適用し、
      最後に結合する。
    """
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    sides = dataset.sides
    if sides is None:
        sides = np.zeros(len(labels), dtype=np.int8)

    # side ごとのインデックスグループ
    side_groups: dict[int, np.ndarray] = {}
    for s in np.unique(sides):
        side_groups[int(s)] = np.where(sides == s)[0]

    per_side_keep: list[np.ndarray] = []
    for side_val, group_idx in side_groups.items():
        sub_labels = labels[group_idx]

        # Step 1: 空セルキャップ (side 内)
        local_empty = np.where(sub_labels == COLOR_EMPTY)[0]
        local_non_empty = np.where(sub_labels != COLOR_EMPTY)[0]
        max_empty = int(
            len(local_non_empty) * empty_ratio_cap
            / (1.0 - empty_ratio_cap)
        )
        if len(local_empty) > max_empty:
            kept_empty_local = rng.choice(
                local_empty, size=max_empty, replace=False,
            )
        else:
            kept_empty_local = local_empty

        # Step 2: 色クラス均等化 (side 内)
        color_bucket: dict[int, list[int]] = {}
        for li in local_non_empty:
            lbl = int(sub_labels[li])
            color_bucket.setdefault(lbl, []).append(int(li))

        if color_bucket:
            min_count = min(len(v) for v in color_bucket.values())
            max_allowed = int(min_count * color_balance_factor)
            balanced_local = []
            for _lbl, idxs_list in color_bucket.items():
                idxs_arr = np.array(idxs_list)
                if len(idxs_arr) > max_allowed:
                    idxs_arr = rng.choice(
                        idxs_arr, size=max_allowed, replace=False,
                    )
                balanced_local.append(idxs_arr)
            keep_local = np.concatenate(
                [kept_empty_local, *balanced_local],
            )
        else:
            keep_local = kept_empty_local

        # group 内 index → 元 dataset の index にマップ
        per_side_keep.append(group_idx[keep_local])

    if per_side_keep:
        keep_idx = np.concatenate(per_side_keep)
    else:
        keep_idx = np.array([], dtype=np.int64)

    rng.shuffle(keep_idx)
    new_patches = dataset.patches[keep_idx]
    new_labels = labels[keep_idx]
    new_sides = sides[keep_idx]

    stats = ExtractionStats(
        frames_sampled=dataset.stats.frames_sampled,
        patches_total=len(new_labels),
        per_class_count={
            int(k): int(v)
            for k, v in zip(*np.unique(new_labels, return_counts=True))
        },
    )
    return PatchDataset(
        patches=new_patches, labels=new_labels, stats=stats,
        sides=new_sides,
    )
