"""予告お邪魔ぷよのラベリング用候補フレームから 12 セル grid 画像を生成する。

戦略:
    - score_series_cache から大連鎖イベント (delta >= 1000) を抽出
    - その時刻 +2 秒のフレーム (連鎖アニメ完了後、予告表示が安定) を取得
    - 各フレームの 1P/2P 上部の 12 セルを切り出し
    - 候補フレーム 12 枚 × 12 セル = 144 セルを大きな grid 画像に並べる
    - 各セルに「F<frame_idx> 1P/2P S<cell_idx>」のラベルを付ける

ユーザは grid を目視して、各セルが小/大/岩/星/月/王冠/空 のどれかをラベル付けする。

出力:
    data/verify/ojama_label_grid.png
    data/verify/ojama_label_index.tsv  (frame_idx, t_sec, video, side, cell_idx)
"""
from __future__ import annotations

import csv
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

from src.ojama_warning import (
    BOARD_WIDTH,
    CELL_COUNT,
    CELL_WIDTH,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_HEIGHT,
    WARNING_TOP_Y,
)

CACHE_PATH = Path("data/training/score_series_cache.json")
OUT_GRID = Path("data/verify/ojama_label_grid.png")
OUT_INDEX = Path("data/verify/ojama_label_index.tsv")
VIDEO_DIR = Path("data/frames")
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)

# 大連鎖判定: score 増分がこの値以上なら採用
LARGE_CHAIN_DELTA_MIN: int = 1000
# 採用候補フレーム数
N_CANDIDATES: int = 12
# フレームオフセット: 連鎖発火直後 +2 秒で予告表示が安定
SAMPLE_OFFSET_SEC: float = 2.0
# 描画用拡大倍率 (cell サイズが小さいので 2x)
SCALE: int = 3
# ラベル領域高さ
LABEL_BAR_H: int = 20
# 各セルの間隔
CELL_GAP: int = 4
# フレーム間の間隔
FRAME_GAP: int = 12


def find_large_chain_events(cache: dict) -> list[tuple[str, str, float, str, int]]:
    """全試合の score series から大連鎖を抽出。
    Returns list of (vid, match_idx, t_sec, side, delta).
    """
    events: list[tuple[str, str, float, str, int]] = []
    for vid, matches in cache.items():
        for midx, samples in matches.items():
            valid = [s for s in samples
                     if s["1p"] is not None and s["2p"] is not None]
            if len(valid) < 2:
                continue
            for i in range(1, len(valid)):
                d1 = valid[i]["1p"] - valid[i - 1]["1p"]
                d2 = valid[i]["2p"] - valid[i - 1]["2p"]
                if d1 >= LARGE_CHAIN_DELTA_MIN:
                    events.append(
                        (vid, midx, valid[i]["t"], "1P", d1),
                    )
                if d2 >= LARGE_CHAIN_DELTA_MIN:
                    events.append(
                        (vid, midx, valid[i]["t"], "2P", d2),
                    )
    # delta 降順 (大連鎖優先) で安定ソート
    events.sort(key=lambda x: x[4], reverse=True)
    return events


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


def extract_cells(
    frame: np.ndarray, side: str
) -> list[np.ndarray]:
    """1 サイドの 6 セルを切り出す。"""
    base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
    out: list[np.ndarray] = []
    for i in range(CELL_COUNT):
        x1 = base_x + i * CELL_WIDTH
        x2 = x1 + CELL_WIDTH
        y1, y2 = WARNING_TOP_Y, WARNING_BOTTOM_Y
        cell = frame[y1:y2, x1:x2].copy()
        out.append(cell)
    return out


def annotate_cell(
    cell: np.ndarray,
    label: str,
    scale: int = SCALE,
) -> np.ndarray:
    """セルを拡大し、ラベルバーを上に付加。"""
    h, w = cell.shape[:2]
    big = cv2.resize(cell, (w * scale, h * scale),
                     interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((LABEL_BAR_H, big.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)
    return np.vstack([bar, big])


def build_frame_panel(
    frame_idx: int,
    title: str,
    cells_1p: list[np.ndarray],
    cells_2p: list[np.ndarray],
) -> np.ndarray:
    """1 フレーム分のパネル: タイトル + 1P 6 セル + 2P 6 セル を横に並べる。"""
    annotated_1p = [
        annotate_cell(c, f"F{frame_idx} 1P S{i}")
        for i, c in enumerate(cells_1p)
    ]
    annotated_2p = [
        annotate_cell(c, f"F{frame_idx} 2P S{i}")
        for i, c in enumerate(cells_2p)
    ]
    sep = np.full((annotated_1p[0].shape[0], CELL_GAP, 3),
                  60, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for c in annotated_1p:
        parts.append(c)
        parts.append(sep)
    # 1P と 2P の境界を明示するため広めの区切り
    big_sep = np.full((annotated_1p[0].shape[0], 12, 3),
                      120, dtype=np.uint8)
    parts.append(big_sep)
    for c in annotated_2p:
        parts.append(c)
        parts.append(sep)
    body = np.hstack(parts[:-1])

    # タイトルバー
    title_bar = np.zeros((24, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 200, 100), 1)
    return np.vstack([title_bar, body])


def main() -> int:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.is_file():
        print(f"cache なし: {CACHE_PATH}")
        return 1
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    events = find_large_chain_events(cache)
    print(f"大連鎖イベント検出: {len(events)} 件 (top delta={events[0][4] if events else 0})")
    if not events:
        return 1

    # 多様性のため、video / match のバランスを取って先頭 N_CANDIDATES を選ぶ
    selected: list[tuple[str, str, float, str, int]] = []
    seen_match: set[tuple[str, str]] = set()
    for ev in events:
        key = (ev[0], ev[1])
        if key in seen_match:
            continue
        seen_match.add(key)
        selected.append(ev)
        if len(selected) >= N_CANDIDATES:
            break
    if len(selected) < N_CANDIDATES:
        # 残りは重複試合からも採用
        for ev in events:
            if ev in selected:
                continue
            selected.append(ev)
            if len(selected) >= N_CANDIDATES:
                break
    print(f"採用フレーム: {len(selected)}")

    panels: list[np.ndarray] = []
    index_rows: list[dict] = []
    for fi, (vid, midx, t_sec, side, delta) in enumerate(selected):
        sample_t = t_sec + SAMPLE_OFFSET_SEC
        video_path = VIDEO_DIR / f"{vid}.mp4"
        frame = get_frame(video_path, sample_t)
        if frame is None:
            print(f"  [skip] フレーム取得失敗: {vid} t={sample_t}")
            continue
        cells_1p = extract_cells(frame, "1P")
        cells_2p = extract_cells(frame, "2P")
        title = (f"F{fi}: {vid} match{midx} t={sample_t:.1f}s "
                 f"(chain by {side}, delta={delta})")
        panel = build_frame_panel(fi, title, cells_1p, cells_2p)
        panels.append(panel)
        for ci in range(CELL_COUNT):
            index_rows.append({
                "frame_idx": fi, "t_sec": round(sample_t, 2),
                "video": vid, "match": midx,
                "side": "1P", "cell_idx": ci,
            })
            index_rows.append({
                "frame_idx": fi, "t_sec": round(sample_t, 2),
                "video": vid, "match": midx,
                "side": "2P", "cell_idx": ci,
            })
        print(f"  [ok] F{fi} {vid} m{midx} t={sample_t:.1f}s side={side} delta={delta}")

    # 全パネル縦結合
    sep = np.full((FRAME_GAP, panels[0].shape[1], 3),
                  30, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for p in panels:
        parts.append(p)
        parts.append(sep)
    grid = np.vstack(parts[:-1])

    cv2.imwrite(str(OUT_GRID), grid)

    # インデックス出力
    with open(OUT_INDEX, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, delimiter="\t",
            fieldnames=["frame_idx", "t_sec", "video", "match", "side", "cell_idx"],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\n出力:")
    print(f"  grid: {OUT_GRID} (shape={grid.shape})")
    print(f"  index: {OUT_INDEX} ({len(index_rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
