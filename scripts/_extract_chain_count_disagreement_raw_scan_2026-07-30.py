"""連鎖数OCR食い違い事例の追加調査: OCR信頼度に頼らない生フレーム grid scan。

## 背景

_extract_chain_count_disagreement_frames_2026-07-30.py の一次結果で、以下の
イベントは代表フレームが「N れんさ!」ポップアップではなく無関係な画面
(SEGA/実況者ロゴ、全消し演出「やった!」、光エフェクトで隠れた表示等) を
誤って捉えていたことが判明した (本番と同じ confidence>=0.60 の
ChainCountOcr.read_side ベースの検出だったため、テンプレ判別力が弱い場面で
誤検出/検出漏れが起きた)。

本ファイルは OCR に一切頼らず、指定イベントの window を一定間隔で機械的に
サンプリングしてサムネイルgridを作る (人間の目で「N れんさ!」表示の位置を
探すための素材)。grid から候補時刻を絞り込んだ後、
scripts/_extract_chain_count_disagreement_zoom_2026-07-30.py で個別に
高解像度 crop を作る想定 (2段階アプローチ)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain_count_ocr import _crop_search_roi, _ensure_1080p  # noqa: E402

VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "chain_count_disagreement_2026-07-29"

SAMPLE_INTERVAL_SEC: float = 0.3
WINDOW_LEAD_SEC: float = 1.0
WINDOW_TAIL_SEC: float = 3.0
THUMB_WIDTH: int = 320
GRID_COLS: int = 6


@dataclass(frozen=True)
class RawScanTarget:
    label: str
    video_stem: str
    side: str
    t_chain_start: float
    t_fire: float


TARGETS: tuple[RawScanTarget, ...] = (
    RawScanTarget("E_diff6_lowtohigh_1P", "c21", "1P", 292.600006, 304.000000),
)


def _thumb(frame: np.ndarray, side: str, t: float) -> np.ndarray:
    """2026-07-30 修正: _ensure_1080p() を経由してから ROI を切り出す
    (scripts/_extract_chain_count_disagreement_frames_2026-07-30.py
    の _crop_enlarged() と同じ修正、理由もそちらのdocstring参照)。"""
    frame = _ensure_1080p(frame)
    if frame is None:
        return np.full((200, 100, 3), 128, dtype=np.uint8)
    roi = _crop_search_roi(frame, side)  # type: ignore[arg-type]
    if roi is None or roi.size == 0:
        roi = np.zeros((200, 100, 3), dtype=np.uint8)
    h, w = roi.shape[:2]
    scale = THUMB_WIDTH / w
    thumb = cv2.resize(roi, (THUMB_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    band = np.full((22, THUMB_WIDTH, 3), 30, dtype=np.uint8)
    cv2.putText(band, f"t={t:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, thumb])


def _build_grid(thumbs: list[np.ndarray]) -> np.ndarray:
    if not thumbs:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    th, tw = thumbs[0].shape[:2]
    n = len(thumbs)
    rows = (n + GRID_COLS - 1) // GRID_COLS
    canvas = np.full((rows * th, GRID_COLS * tw, 3), 15, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, GRID_COLS)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return canvas


def _process(target: RawScanTarget) -> None:
    video_path = VIDEO_DIR / f"video_{target.video_stem}.mp4"
    if not video_path.exists():
        print(f"[SKIP] {target.label}: 動画なし")
        return
    cap = cv2.VideoCapture(str(video_path))
    t_start = max(0.0, target.t_chain_start - WINDOW_LEAD_SEC)
    t_end = target.t_fire + WINDOW_TAIL_SEC
    thumbs: list[np.ndarray] = []
    t = t_start
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            thumbs.append(_thumb(frame, target.side, t))
        t += SAMPLE_INTERVAL_SEC
    cap.release()
    grid = _build_grid(thumbs)
    ev_dir = OUT_DIR / target.label
    ev_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ev_dir / "raw_scan_grid.png"), grid)
    print(f"[OK] {target.label}: {len(thumbs)}枚 → {ev_dir / 'raw_scan_grid.png'}")


def main() -> None:
    for target in TARGETS:
        _process(target)


if __name__ == "__main__":
    main()
