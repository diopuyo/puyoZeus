"""NextDetector._classify_side の 4.5ms の内訳を切り分ける (2026-07-31)。

実測: next_detector.detect_both = 4.5ms/frame で、その**全額**が _classify_side
(2回/frame)。内部は 4 ROI x (CNN + HSV + centroid) = 1フレームで
**8回の単発CNN呼び出し**になっている (盤面側は156セルを2回に束ねている)。

まとめる前に「4.5ms のうち CNN が何 ms か」を測る。
CNN が支配的でなければ、バッチ化しても目標の -2.7ms には届かない。
(2026-07-30〜31 にバッチ化が3回連続で不正解だった教訓を踏まえ、先に測る)
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080

_ACC: dict[str, float] = defaultdict(float)
_CNT: dict[str, int] = defaultdict(int)
_PATCHED: list[tuple[Any, str, Callable]] = []


def _patch_obj(obj: Any, method: str, label: str) -> bool:
    """インスタンスのメソッドを計時ラッパーに差し替える (成功で True)。"""
    original = getattr(obj, method, None)
    if original is None:
        return False

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _ACC[label] += time.perf_counter() - t0
            _CNT[label] += 1

    setattr(obj, method, wrapper)
    _PATCHED.append((obj, method, original))
    return True


def _patch_module_fn(module: Any, name: str, label: str) -> bool:
    """モジュール関数を計時ラッパーに差し替える。"""
    original = getattr(module, name, None)
    if original is None:
        return False

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _ACC[label] += time.perf_counter() - t0
            _CNT[label] += 1

    setattr(module, name, wrapper)
    _PATCHED.append((module, name, original))
    return True


def _unpatch() -> None:
    """差し替えを全て戻す。"""
    for obj, name, original in reversed(_PATCHED):
        setattr(obj, name, original)
    _PATCHED.clear()


def _read_frames(video: Path, frames: int, start_sec: float) -> list[np.ndarray]:
    """動画から連続フレームを読み出す。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    out: list[np.ndarray] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        out.append(frame)
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/frames/video_c60.mp4"))
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    import src.next_detector as nd
    from src.recognition_pipeline import RecognitionPipeline

    frames = _read_frames(args.video, args.frames, args.start_sec)
    n = len(frames)
    print(f"動画: {args.video.name} t={args.start_sec}s フレーム数: {n}")

    pipe = RecognitionPipeline.load_default()
    detector = getattr(pipe, "_next_detector", None)
    if detector is None:
        for _k, v in vars(pipe).items():
            if type(v).__name__ == "NextDetector":
                detector = v
                break
    if detector is None:
        print("NextDetector が見つからない")
        return

    ok = []
    try:
        if _patch_obj(detector._classifier, "classify", "GatedCnn.classify(合計)"):
            ok.append("GatedCnn.classify(合計)")
        # ゲート込みの合計では「CNN本体かゲートか」が分からないので内側も測る
        inner_cnn = getattr(detector._classifier, "_color", None)
        if inner_cnn is not None and _patch_obj(
            inner_cnn, "classify", "  └ CNN本体(CnnPatch)",
        ):
            ok.append("  └ CNN本体(CnnPatch)")
        gate = getattr(detector._classifier, "_gate", None)
        if gate is not None and _patch_obj(gate, "is_puyo", "  └ 存在ゲート.is_puyo"):
            ok.append("  └ 存在ゲート.is_puyo")
        um = getattr(detector._classifier, "_ui_matcher", None)
        if um is not None and _patch_obj(um, "is_ui", "  └ ui_matcher.is_ui"):
            ok.append("  └ ui_matcher.is_ui")
        if detector._centroid is not None and _patch_obj(
            detector._centroid, "classify", "centroid.classify",
        ):
            ok.append("centroid.classify")
        if _patch_module_fn(nd, "hsv_dominant_color", "hsv_dominant_color"):
            ok.append("hsv_dominant_color")
        if _patch_obj(detector, "_classify_side", "_classify_side(全体)"):
            ok.append("_classify_side(全体)")
        t0 = time.perf_counter()
        for idx, frame in enumerate(frames):
            pipe.update(idx, idx / 30.0, frame)
        total = time.perf_counter() - t0
    finally:
        _unpatch()

    print(f"\n認識全体: {total / n * 1000:.1f} ms/frame\n")
    print(f"{'項目':<26}{'ms/frame':>10}{'回/frame':>10}{'1回あたり':>12}")
    print("-" * 58)
    for label in ok:
        if _CNT[label] == 0:
            print(f"{label:<26}{'(呼出なし)':>10}")
            continue
        ms = _ACC[label] / n * 1000
        per = _ACC[label] / _CNT[label] * 1e6
        print(f"{label:<26}{ms:>10.2f}{_CNT[label] / n:>10.1f}{per:>10.1f}us")
    inner = sum(
        _ACC[k] for k in ok if k in ("GatedCnn.classify(合計)", "hsv_dominant_color", "centroid.classify")
    )
    if _ACC["_classify_side(全体)"] > 0:
        print("-" * 58)
        print(
            f"内訳合計が _classify_side に占める割合: "
            f"{100 * inner / _ACC['_classify_side(全体)']:.1f}%  "
            f"(残り = 多数決ロジック・切り出し)"
        )
        print(
            "\n→ CNN が支配的ならバッチ化 (8回→1回) が有効。"
            "そうでなければ別の的を探す。"
        )


if __name__ == "__main__":
    main()
