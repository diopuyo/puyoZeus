"""認識時間の「内訳不明な拡散部分」をサブシステム単位で実時間分解する (2026-07-30)。

背景 (実測):
    matchTemplate は認識時間の 62.1% を占め、呼び出し元別の内訳も確定済み
    (ui_mask 40.9% / score_ocr 19.9% / 大ROI 10.2%)。
    UI マスクのセル限定を配線すると 226.3ms → 122.1ms になるが、
    **その 122ms のうち非 matchTemplate が約 70ms あり、内訳が分かっていない。**
    matchTemplate を全部消しても 14.3fps 止まりなので、
    **30fps (予算 33.3ms) の可否はこの 70ms の正体で決まる。**

なぜ cProfile を使わないか:
    cProfile は Python フレームごとにオーバーヘッドを乗せるため、
    Python 側が重い処理を過大に、C 実装 (matchTemplate 等) を過小に見せる。
    拡散部分はまさに Python 側なので、cProfile では割合を誤読する。
    → time.perf_counter で入口メソッドを直接ラップして実時間で測る。

方針 (厳守):
    - src/ は一切変更しない。クラスのメソッドをスクリプト内で差し替え、
      finally で必ず復元する。
    - **入れ子の二重計上を避ける**ため、階層を明示して報告する
      (例: read_board の中に classify_batch が含まれる)。
    - 計装オーバーヘッド自体を baseline との差で報告する。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_diffuse_breakdown_2026-07-30 \
        --video data/frames/video_c60.mp4 --start-sec 1451 --frames 60
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
WARMUP_FRAMES: int = 10

_ACC: dict[str, float] = defaultdict(float)
_CNT: dict[str, int] = defaultdict(int)
# 復元用: (クラス, メソッド名, 元の関数)
_PATCHED: list[tuple[type, str, Callable]] = []

# 計測対象。(モジュール, クラス名, メソッド名, 表示名, 階層)
# 階層は入れ子関係を示す (親の時間に子が含まれる)。
TARGETS: tuple[tuple[str, str, str, str, str], ...] = (
    ("src.image_reader", "ImageReader", "read_both_boards", "read_both_boards", "L1"),
    ("src.image_reader", "ImageReader", "read_board", "  read_board", "L2"),
    ("src.hybrid_classifier", "HybridClassifier", "classify_batch",
     "    classify_batch(CNN+HSV)", "L3"),
    ("src.image_reader", "ColorClassifier", "classify", "      classify(HSV1セル)", "L4"),
    ("src.next_detector", "NextDetector", "detect_both", "next_detector.detect_both", "L1"),
    ("src.next_detector", "NextDetector", "detect_stable", "next_detector.detect_stable", "L1"),
    # --- 残余 37.5ms を掘るために追加 (2026-07-30 第2段) ---
    # recognition_pipeline.update 直下で frame を受け取る detector 群
    ("src.match_state", "MatchStateDetector", "detect", "match_detector.detect", "L1"),
    ("src.score_zero", "ScoreZeroDetector", "detect", "score_zero.detect", "L1"),
    ("src.match_end_detector", "MatchEndDetector", "update", "match_end.update", "L1"),
    ("src.telop_detector", "TelopDetector", "detect", "telop.detect", "L1"),
    # 1P/2P ごとの状態遷移処理本体 (state_machine.update より外側の糊付け)
    ("src.recognition_pipeline", "RecognitionPipeline", "_step_side",
     "pipeline._step_side", "L1"),
    ("src.board_state_machine", "BoardStateMachine", "update",
     "  state_machine.update", "L2"),
    ("src.recognition_pipeline", "RecognitionPipeline", "_update_score_tracker",
     "pipeline._update_score_tracker", "L1"),
    ("src.score_ocr", "ScoreOcr", "read_side", "  score_ocr.read_side", "L2"),
)


def _patch(module_name: str, cls_name: str, method: str, label: str) -> bool:
    """クラスメソッドを計時ラッパーに差し替える。成功したら True。"""
    try:
        mod = __import__(module_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        original = getattr(cls, method)
    except (ImportError, AttributeError):
        return False
    # staticmethod は self が渡らないので、ラップ後も staticmethod で戻さないと
    # 呼び出し側で引数が 1 個ずれる (実際に _update_score_tracker で踏んだ)
    raw = cls.__dict__.get(method)
    is_static = isinstance(raw, staticmethod)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _ACC[label] += time.perf_counter() - t0
            _CNT[label] += 1

    setattr(cls, method, staticmethod(wrapper) if is_static else wrapper)
    _PATCHED.append((cls, method, raw if raw is not None else original))
    return True


def _unpatch_all() -> None:
    """差し替えたメソッドを全て復元する。"""
    for cls, method, original in reversed(_PATCHED):
        setattr(cls, method, original)
    _PATCHED.clear()


def _read_frames(video: Path, frames: int, start_sec: float) -> list[np.ndarray]:
    """動画から連続フレームを読み出す (1920x1080 に正規化)。"""
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


def _run(frames: list[np.ndarray]) -> float:
    """パイプラインを 1 本走らせ、定常フレームの中央値 ms を返す。"""
    from src.recognition_pipeline import RecognitionPipeline

    pipe = RecognitionPipeline.load_default()
    times: list[float] = []
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        pipe.update(idx, idx / 30.0, frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    steady = arr[WARMUP_FRAMES:] if arr.size > WARMUP_FRAMES else arr
    return float(np.median(steady))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/frames/video_c60.mp4"))
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    frames = _read_frames(args.video, args.frames, args.start_sec)
    n = len(frames)
    print(f"動画: {args.video.name} t={args.start_sec}s フレーム数: {n} cv_threads=1")

    # --- baseline (無計装) ---
    base_ms = _run(frames)
    print(f"\nbaseline (無計装): 定常中央 {base_ms:.1f}ms → {1000 / base_ms:.2f} fps")

    # --- 計装あり ---
    ok_labels: list[tuple[str, str]] = []
    try:
        for module_name, cls_name, method, label, tier in TARGETS:
            if _patch(module_name, cls_name, method, label):
                ok_labels.append((label, tier))
            else:
                print(f"  [skip] 差し替え失敗: {cls_name}.{method}")
        inst_ms = _run(frames)
    finally:
        _unpatch_all()

    overhead = 100.0 * (inst_ms - base_ms) / base_ms if base_ms else 0.0
    print(
        f"計装あり: 定常中央 {inst_ms:.1f}ms "
        f"(オーバーヘッド {overhead:+.1f}%)"
    )

    print(f"\n{'サブシステム':<30}{'ms/frame':>10}{'baseline比':>11}{'回/frame':>10}")
    print("-" * 61)
    for label, tier in ok_labels:
        if _CNT[label] == 0:
            print(f"{label:<30}{'(呼出なし)':>10}")
            continue
        ms = _ACC[label] / n * 1000
        pct = 100.0 * ms / base_ms if base_ms else 0.0
        print(f"{label:<30}{ms:>10.1f}{pct:>10.1f}%{_CNT[label] / n:>10.1f}")

    # L1 (入れ子でない最上位) の合計と、そこから漏れている残余を出す
    l1_ms = sum(
        _ACC[label] / n * 1000 for label, tier in ok_labels if tier == "L1"
    )
    print("-" * 61)
    print(f"{'L1 合計 (入れ子なし)':<30}{l1_ms:>10.1f}{100 * l1_ms / base_ms:>10.1f}%")
    print(
        f"{'★どこにも属さない残余':<30}{base_ms - l1_ms:>10.1f}"
        f"{100 * (base_ms - l1_ms) / base_ms:>10.1f}%"
    )
    print(
        "\n※残余が大きい場合は recognition_pipeline.update 直下の処理 "
        "(状態管理・おじゃま会計・各種 detector) が計測対象から漏れている。"
    )


if __name__ == "__main__":
    main()
