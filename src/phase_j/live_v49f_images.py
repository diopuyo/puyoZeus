"""発火前画像の短いリング。未到着画像へのシークを持たない。"""
from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any

from src.recog_speed_py import ITEM_IMAGE_BATCH, ITEM_IMAGE_CROPS, STATS, enabled

HISTORY_SEC = 1.
OFFSETS = (-.3, -.15, 0.)
COLS, CELLS = 6, 78
MS = 1000.
RING_CELLS = range(COLS, CELLS)   # リングの分類が読むのは可視 12 行 x 6 列 (隠し段は読まない)
# 盤面の外枠の表。キー = (画像の高さ, 幅, 各側の領域の幾何)。値 = 側ごとの (左, 上, 右, 下)。使えない画像は None。
_BOX_CACHE: dict[tuple, tuple[tuple[int, int, int, int], ...] | None] = {}


def _region_boxes(height: int, width: int) -> tuple[tuple[int, int, int, int], ...] | None:
    """各側のリングが読むセル矩形すべてを含む最小の外枠 (画像の端で切る)。使えなければ None。

    負の座標は numpy の負の添字 (末尾から数える) になり、切り出し方が変わってしまう。
    そういう領域は外枠に直さず None (= 従来どおり全画面を保持) にする。
    """
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)
    key = (height, width) + tuple((r.x, r.y, r.width, r.height) for r in regions)
    if key in _BOX_CACHE:
        return _BOX_CACHE[key]
    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        rects = [region.cell_sample_rect(*divmod(cell, COLS)) for cell in RING_CELLS]
        if min(min(rect[0], rect[1]) for rect in rects) < 0:
            _BOX_CACHE[key] = None
            return None
        boxes.append((min(r[0] for r in rects), min(r[1] for r in rects),
                      min(width, max(r[2] for r in rects)), min(height, max(r[3] for r in rects))))
    _BOX_CACHE[key] = tuple(boxes)
    return _BOX_CACHE[key]


class BoardCrops:
    """盤面 2 領域だけの画素 (live_recog_speed_py の image_crops)。座標は元の画像のまま。

    patch(side, x1, y1, x2, y2) は、元の全画面 image[y1:y2, x1:x2] と同じ画素・同じ形を返す
    (外枠はその側のセル矩形をすべて含み、右・下は画像の端で切ってあるので、numpy が端で
    切り詰める挙動も同じ)。
    """

    __slots__ = ('origins', 'pixels')

    def __init__(self, image: Any, boxes: tuple[tuple[int, int, int, int], ...]) -> None:
        self.origins = tuple((box[0], box[1]) for box in boxes)
        self.pixels = tuple(image[box[1]:box[3], box[0]:box[2]].copy() for box in boxes)

    def patch(self, side: int, x1: int, y1: int, x2: int, y2: int) -> Any:
        left, top = self.origins[side]
        return self.pixels[side][y1-top:y2-top, x1-left:x2-left]


class LiveImages:
    def __init__(self) -> None:
        from src.image_reader import ColorClassifier
        from src.ui_mask import UiMaskMatcher
        self.frames: deque = deque()
        self.classifiers = [ColorClassifier(), ColorClassifier(vote_mode=True)]
        from src.live_native_read_board import enable_for_live_images
        enable_for_live_images(self.classifiers)   # 既定 OFF: live_native_read_board が ON のときだけ Rust 経路
        self.ui = UiMaskMatcher.load_default()

    def append(self, stamp: float, image: Any) -> None:
        self.frames.append((stamp, self._retain(image)))
        while self.frames and self.frames[0][0] < stamp-HISTORY_SEC:
            self.frames.popleft()

    @staticmethod
    def _retain(image: Any) -> Any:
        """リングへ残す画像。image_crops が ON なら盤面 2 領域だけ (読むのはそこだけ)。"""
        if enabled(ITEM_IMAGE_CROPS) and getattr(image, 'ndim', 0) == 3:
            boxes = _region_boxes(image.shape[0], image.shape[1])
            if boxes is not None:
                STATS['image_crops_retained'] += 1
                return BoardCrops(image, boxes)
        return image.copy()

    def boards(self, trigger: float, stamp: float, side: int,
               offsets: tuple = OFFSETS) -> list[dict]:
        result = []
        for offset in offsets:
            candidates = [(t, image) for t, image in self.frames if t <= min(stamp, trigger+offset)]
            if not candidates:
                raise ValueError('発火前画像待ち')
            time, image = candidates[-1]
            result.extend(self.classify(image, time, side))
        return result

    @staticmethod
    def _patch(image: Any, side: int, rect: tuple[int, int, int, int]) -> Any:
        """セルの矩形の画素。リングが盤面だけを持つ場合も、全画面の image[y1:y2, x1:x2] と同じ。"""
        x1, y1, x2, y2 = rect
        if isinstance(image, BoardCrops):
            return image.patch(side, x1, y1, x2, y2)
        return image[y1:y2, x1:x2]

    def classify(self, image: Any, stamp: float, side: int) -> list[dict]:
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        region = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)[side]
        began = perf_counter()
        if enabled(ITEM_IMAGE_BATCH):
            boards = self._classify_batched(image, region, side)
        else:
            boards = self._classify_per_cell(image, region, side)
        cost = (perf_counter()-began)*MS/len(boards)
        return [dict(kind='image', t=stamp, board=board, image_compute_ms=cost) for board in boards]

    def _classify_per_cell(self, image: Any, region: Any, side: int) -> list[list[int]]:
        """従来どおり 1 セルずつ。セルごとに UI マスクを見てから各分類器で分類する。"""
        from src.ui_mask import UI_MASK_TARGET_CELLS
        boards = [[0]*COLS for _ in self.classifiers]
        for cell in RING_CELLS:
            row, col = divmod(cell, COLS)
            patch = self._patch(image, side, region.cell_sample_rect(row, col))
            masked = (row, col) in UI_MASK_TARGET_CELLS and self.ui.is_ui(patch)
            for classifier, board in zip(self.classifiers, boards):
                board.append(0 if masked else int(classifier.classify(patch)))
        return boards

    def _classify_batched(self, image: Any, region: Any, side: int) -> list[list[int]]:
        """分類器ごとに全セルをまとめて分類する (image_batch)。

        ColorClassifier.classify は画素だけで決まる純関数なので、セルごと・分類器ごとの呼ぶ順を
        入れ替えても結果は同じ。通常 HSV 分類器がネイティブ経路 (24d の試験で既に有効) なら
        classify_patches が 1 回の Rust 呼出にまとめ、使えなければ 1 パッチずつ従来の classify に戻る。
        """
        from src.live_board_vectorized import classify_patches
        from src.ui_mask import UI_MASK_TARGET_CELLS
        patches, masked = [], []
        for cell in RING_CELLS:
            row, col = divmod(cell, COLS)
            patch = self._patch(image, side, region.cell_sample_rect(row, col))
            patches.append(patch)
            masked.append((row, col) in UI_MASK_TARGET_CELLS and self.ui.is_ui(patch))
        live = [index for index, flag in enumerate(masked) if not flag]
        boards = []
        for classifier in self.classifiers:
            colors = classify_patches(classifier, [patches[index] for index in live])
            board = [0]*(COLS+len(patches))
            for index, color in zip(live, colors):
                board[COLS+index] = int(color)
            boards.append(board)
        STATS['image_batched_classify'] += 1
        return boards

    def read_cells(self, image: Any, side: int, cells: tuple[int, ...],
                   which: tuple[int, ...] = (0, 1)) -> list[dict[int, int]]:
        """段3 (live_x_prefire) 用。指定セルだけを classify と同じ規則で読む。which の分類器ごとに {セル: 色}。"""
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        from src.ui_mask import UI_MASK_TARGET_CELLS
        region = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)[side]
        classifiers = [self.classifiers[i] for i in which]
        reads: list[dict[int, int]] = [{} for _ in classifiers]
        for cell in cells:
            if cell < COLS:
                continue
            row, col = divmod(cell, COLS)
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            patch = image[y1:y2, x1:x2]
            masked = (row, col) in UI_MASK_TARGET_CELLS and self.ui.is_ui(patch)
            for classifier, read in zip(classifiers, reads):
                read[cell] = 0 if masked else int(classifier.classify(patch))
        return reads
