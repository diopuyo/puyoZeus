"""COLOR_EMPTY (背景) の seed dataset を抽出する。

cycle_14 で発覚: 5 puyo color のみで fine-tune した model は empty 推定が消失して
背景まで puyo として認識する。 → empty を含めて再 fine-tune するための seed 抽出。

実装方針:
    base model (cnn_phase_b_large_v3.pt = fine-tune 前) で動画再生し、
    STABLE 中の confirmed_board=EMPTY な cell の patch を保存する。
    fine-tune 前の model は empty を正しく出すため、 信頼サンプルが取れる。

出力:
    data/pseudo_labels_hsv_seed_with_empty/<vid>/cell.jsonl
    (既存の 5 puyo color seed も同 dir にコピーする想定)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_EMPTY, HIDDEN_ROWS
from src.board_state_machine import BoardState
from src.hybrid_classifier import HybridClassifier
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.recognition_pipeline import RecognitionPipeline
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL, PseudoLabelSample

# empty 専用品質閾値 (背景は S が低い、 V もそこそこ低い前提)
MAX_S_MEDIAN_EMPTY: float = 80.0   # S 高すぎ = puyo の可能性 → 除外
MAX_V_STD: float = 50.0            # V が均一 = 背景、 ばらつき大 = puyo の影


def _pre_inject_hsv(pipeline: RecognitionPipeline, hsv_state: Path) -> None:
    import json as _json
    with hsv_state.open("r", encoding="utf-8") as f:
        state = _json.load(f)
    ranges = state.get("per_video_ranges", {})
    if not ranges:
        return
    ranges_int = {
        int(k): tuple(int(x) for x in v) for k, v in ranges.items()
    }
    hc = pipeline._reader._classifier
    if (
        isinstance(hc, HybridClassifier)
        and hasattr(hc._hsv, "set_color_ranges_from_simple")
    ):
        hc._hsv.set_color_ranges_from_simple(ranges_int)
        print(f"[seed-empty] HSV pre-inject: {len(ranges_int)} colors")


def _extract_patch(
    frame: np.ndarray, region, row: int, col: int,
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(y1 + 1, min(int(y2), h))
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


def _is_quality_empty(patch: np.ndarray) -> bool:
    """背景らしい patch か (S 低 + V std 低)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s_med = float(np.median(hsv[:, :, 1]))
    v_std = float(np.std(hsv[:, :, 2]))
    return s_med <= MAX_S_MEDIAN_EMPTY and v_std <= MAX_V_STD


def _collect_empty_samples(
    frame: np.ndarray, side_result, region, side: str,
    video_id: str, fi: int, t_sec: float,
    count: list[int], max_count: int,
) -> list[PseudoLabelSample]:
    if side_result.state != BoardState.STABLE:
        return []
    if side_result.confirmed_board is None:
        return []
    out: list[PseudoLabelSample] = []
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(side_result.confirmed_board.get(row, col))
            if color != COLOR_EMPTY:
                continue
            if count[0] >= max_count:
                return out
            patch = _extract_patch(frame, region, row, col)
            if patch is None or not _is_quality_empty(patch):
                continue
            out.append(PseudoLabelSample(
                component=COMPONENT_CELL,
                timestamp=t_sec,
                input_data={"patch": patch},
                label=COLOR_EMPTY,
                confidence=1.0,
                metadata={
                    "video_id": video_id, "frame_idx": fi,
                    "row": row, "col": col, "side": side,
                    "empty_seed": True,
                },
            ))
            count[0] += 1
    return out


def extract(
    video: Path, video_id: str, out_root: Path,
    max_empty: int, cnn_model: Path | None,
    hsv_state: Path | None,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    pipeline = RecognitionPipeline.load_default(cnn_model_path=cnn_model)
    if hsv_state is not None and hsv_state.exists():
        try:
            _pre_inject_hsv(pipeline, hsv_state)
        except Exception as e:
            print(f"[seed-empty] pre-inject failed: {e}", file=sys.stderr)
    store = LabelStore(video_id=video_id, root=out_root)
    count = [0]
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        samples = _collect_empty_samples(
            frame, result.p1, DEFAULT_P1_REGION, "1P",
            video_id, fi, t_sec, count, max_empty,
        )
        samples.extend(_collect_empty_samples(
            frame, result.p2, DEFAULT_P2_REGION, "2P",
            video_id, fi, t_sec, count, max_empty,
        ))
        if samples:
            store.append(samples)
        if count[0] >= max_empty:
            print(
                f"[{video_id}] empty reached {max_empty} @frame={fi}",
            )
            break
        if fi % 300 == 0:
            print(f"[{video_id}] frame={fi} empty_count={count[0]}")
        fi += 1
    cap.release()
    print(f"[{video_id}] DONE empty_count={count[0]}")
    return count[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--video-id", type=str, required=True)
    p.add_argument(
        "--out-root", type=Path,
        default=Path("data/pseudo_labels_hsv_seed_with_empty"),
    )
    p.add_argument("--max-empty", type=int, default=500)
    p.add_argument("--cnn-model", type=Path, default=None)
    p.add_argument(
        "--hsv-state", type=Path,
        default=Path("data/per_video_hsv_ranges/_merged_default.json"),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    extract(
        args.video, args.video_id, args.out_root,
        args.max_empty, args.cnn_model, args.hsv_state,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
