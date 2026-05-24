"""
人ラベル jsonl 群を CNN 学習用 Gold npz に集約するツール。

使い方:
    ./venv/bin/python scripts/human_labels_to_npz.py

処理:
    1. data/verify/human_labels/*.jsonl を全走査
    2. 各 correction エントリから実パッチ画像を読み込み (44×44 にリサイズ)
    3. patches, labels, sides を 1 つにまとめて npz 保存:
         data/training/human_gold.npz
    4. 重複除外: 同じ (frame, side, row, col) は最新 jsonl のみ採用

出力 npz のキー:
    patches: (N, 44, 44, 3) uint8 BGR
    labels:  (N,)           int8  (0=空, 1=赤, ..., 9=お邪魔 の board.py 色コード)
    sides:   (N,)           int8  (0=unknown, 1=1P, 2=2P)
    sources: (N,)           object  (デバッグ用: 元 jsonl パス)

既存の human_gold.npz がある場合は上書きする (毎回完全再構築)。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
)

# 定数
PATCH_SIZE: int = 44
HUMAN_LABELS_DIR: Path = Path("data/verify/human_labels")
OUTPUT_NPZ: Path = Path("data/training/human_gold.npz")

LABEL_TO_CODE: dict[str, int] = {
    "empty": COLOR_EMPTY,
    "red": COLOR_RED,
    "blue": COLOR_BLUE,
    "green": COLOR_GREEN,
    "yellow": COLOR_YELLOW,
    "purple": COLOR_PURPLE,
    "ojama": COLOR_OJAMA,
}

SIDE_TO_CODE: dict[str, int] = {
    "unknown": 0,
    "1P": 1,
    "2P": 2,
}


def _load_jsonl(path: Path) -> list[dict]:
    """jsonl ファイルの各行を辞書リストで返す。"""
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON 解析失敗 ({path.name}): {e}")
    return entries


def _gather_corrections() -> list[dict]:
    """全 jsonl から correction だけを抽出 (重複は後勝ち)。"""
    jsonls = sorted(HUMAN_LABELS_DIR.glob("*/*.jsonl")) + sorted(HUMAN_LABELS_DIR.glob("*.jsonl"))
    # 重複キー: (frame, side, row, col) → 最新の訂正のみ
    latest: dict[tuple[str, str, int, int], dict] = {}
    for jp in jsonls:
        for entry in _load_jsonl(jp):
            if entry.get("kind") != "correction":
                continue
            frame = entry.get("frame") or ""
            side = entry.get("side") or "unknown"
            row = entry.get("row")
            col = entry.get("col")
            if row is None or col is None:
                continue
            # jsonl の patch_file が絶対 or 相対の場合に対応
            patch_file = entry.get("patch_file")
            if patch_file and not Path(patch_file).is_absolute():
                patch_file = str(jp.parent / patch_file)
            entry["_resolved_patch_file"] = patch_file
            entry["_source_jsonl"] = str(jp)
            latest[(frame, side, int(row), int(col))] = entry
    return list(latest.values())


def _load_and_normalize_patch(patch_file: str) -> np.ndarray | None:
    """パッチ画像を 44×44 BGR に正規化して返す。失敗時 None。"""
    if not patch_file or not Path(patch_file).exists():
        return None
    img = cv2.imread(patch_file)
    if img is None:
        return None
    if img.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def main() -> None:
    if not HUMAN_LABELS_DIR.exists():
        print(f"人ラベルディレクトリが存在しません: {HUMAN_LABELS_DIR}")
        print("まず collect_human_labels.py export で訂正を作成してください。")
        sys.exit(0)

    corrections = _gather_corrections()
    if not corrections:
        print(f"訂正が 1 件も見つかりません。{HUMAN_LABELS_DIR}/*.jsonl を確認してください。")
        # それでも空 npz を書いておく (long_improve_v2 の Phase 4 skip 用)
        OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUTPUT_NPZ,
            patches=np.zeros((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8),
            labels=np.zeros((0,), dtype=np.int8),
            sides=np.zeros((0,), dtype=np.int8),
            sources=np.array([], dtype=object),
        )
        print(f"空 npz を書きました: {OUTPUT_NPZ}")
        return

    patches: list[np.ndarray] = []
    labels: list[int] = []
    sides: list[int] = []
    sources: list[str] = []
    skipped = 0

    for entry in corrections:
        patch_file = entry.get("_resolved_patch_file")
        patch = _load_and_normalize_patch(patch_file) if patch_file else None
        if patch is None:
            skipped += 1
            continue

        label_str = entry.get("true_label", "")
        code = LABEL_TO_CODE.get(label_str)
        if code is None:
            skipped += 1
            continue

        side_str = entry.get("side", "unknown")
        side_code = SIDE_TO_CODE.get(side_str, 0)

        patches.append(patch)
        labels.append(code)
        sides.append(side_code)
        sources.append(entry.get("_source_jsonl", ""))

    if not patches:
        print("有効な訂正パッチが 1 件もありません (画像ファイル欠損 or ラベル不正)")
        sys.exit(1)

    patches_arr = np.stack(patches, axis=0)
    labels_arr = np.array(labels, dtype=np.int8)
    sides_arr = np.array(sides, dtype=np.int8)
    sources_arr = np.array(sources, dtype=object)

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        patches=patches_arr,
        labels=labels_arr,
        sides=sides_arr,
        sources=sources_arr,
    )

    print(f"Gold npz 書き出し完了: {OUTPUT_NPZ}")
    print(f"  総訂正数: {len(corrections)}")
    print(f"  取り込み: {len(patches)}  / skip: {skipped}")
    print(f"  内訳ラベル:")
    unique, counts = np.unique(labels_arr, return_counts=True)
    code_to_name = {v: k for k, v in LABEL_TO_CODE.items()}
    for code, count in zip(unique, counts):
        name = code_to_name.get(int(code), f"code={code}")
        print(f"    {name}: {count}")
    print(f"  内訳 side:")
    unique_s, counts_s = np.unique(sides_arr, return_counts=True)
    code_to_side = {v: k for k, v in SIDE_TO_CODE.items()}
    for code, count in zip(unique_s, counts_s):
        name = code_to_side.get(int(code), f"code={code}")
        print(f"    {name}: {count}")


if __name__ == "__main__":
    main()
