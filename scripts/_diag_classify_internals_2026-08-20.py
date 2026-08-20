"""HSV 分類 (classify) の内部内訳を測り、Rust 化の対象を確定する (2026-08-20)。

`scripts/_diag_diffuse_breakdown_2026-07-30.py` で classify が 4.5ms/frame
(全体の29.3%、103.9回/frame) と最大の単一項目であることが判明した。その中を
さらに分解し、どこを Rust 化すれば効くかを数値で決める。

対象 (src/image_reader.py ColorClassifier.classify の構成要素):
  - cv2.cvtColor (BGR->HSV)          … OpenCV の C 実装。Rust 移植は
                                        丸めの互換性が難しく bit-identical
                                        のリスクがある
  - _compute_stable_h_median         … 赤の色相折り返し補正つき median
  - _compute_specular_robust_s       … 光沢ハイライト除外の彩度 median
  - _median_fast (V)                 … np.partition ベース (既に最適化済み)
  - 色範囲の閾値照合ループ            … Python の二重ループ。整数比較のみ
                                        なので Rust 化で bit-identical を
                                        保てる本命

計測方針は既存計測器と同じ (memory: cProfile 禁止、src/ は変更せず
メソッドを差し替えて finally で復元、time.perf_counter で実時間)。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

from src import image_reader as ir  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

_ACC: dict[str, float] = {}
_CNT: dict[str, int] = {}


def _wrap(obj: object, name: str, key: str) -> object:
    """メソッドを perf_counter 計装で包む。戻り値は元の関数 (復元用)。"""
    orig = getattr(obj, name)

    def timed(*a, **k):  # noqa: ANN002, ANN003, ANN202
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _ACC[key] = _ACC.get(key, 0.0) + (time.perf_counter() - t0)
            _CNT[key] = _CNT.get(key, 0) + 1

    setattr(obj, name, timed)
    return orig


def main() -> None:
    """指定動画の指定区間で classify 内部の内訳を出す。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/frames/video_c34.mp4"))
    ap.add_argument("--start-sec", type=float, default=900.0)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    pipe = RecognitionPipeline.load_default()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[error] 動画を開けない: {args.video}")
        return
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    cls = ir.ColorClassifier
    saved: list[tuple[object, str, object]] = []
    # classify 内部の構成要素を包む (クラス単位なので全インスタンスに効く)
    for name, key in (
        ("classify", "classify(全体)"),
        ("_compute_stable_h_median", "  H median (折返し補正)"),
        ("_compute_specular_robust_s", "  S median (光沢除外)"),
    ):
        if hasattr(cls, name):
            saved.append((cls, name, _wrap(cls, name, key)))

    # モジュール関数 _median_fast (V の median) と cvtColor
    if hasattr(ir, "_median_fast"):
        orig_mf = ir._median_fast

        def timed_mf(*a, **k):  # noqa: ANN002, ANN003, ANN202
            t0 = time.perf_counter()
            try:
                return orig_mf(*a, **k)
            finally:
                _ACC["  V median (_median_fast)"] = _ACC.get("  V median (_median_fast)", 0.0) + (time.perf_counter() - t0)
                _CNT["  V median (_median_fast)"] = _CNT.get("  V median (_median_fast)", 0) + 1

        ir._median_fast = timed_mf
    orig_cvt = cv2.cvtColor

    def timed_cvt(*a, **k):  # noqa: ANN002, ANN003, ANN202
        t0 = time.perf_counter()
        try:
            return orig_cvt(*a, **k)
        finally:
            _ACC["  cv2.cvtColor"] = _ACC.get("  cv2.cvtColor", 0.0) + (time.perf_counter() - t0)
            _CNT["  cv2.cvtColor"] = _CNT.get("  cv2.cvtColor", 0) + 1

    cv2.cvtColor = timed_cvt

    per_frame: list[float] = []
    try:
        n = 0
        while n < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            # 引数順は (frame_idx, time_sec, frame) — 既存計測器と同じ呼び方
            pipe.update(n, args.start_sec + n / 30.0, frame)
            per_frame.append((time.perf_counter() - t0) * 1000.0)
            n += 1
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)
        cv2.cvtColor = orig_cvt
        if hasattr(ir, "_median_fast"):
            ir._median_fast = orig_mf
        cap.release()

    if not per_frame:
        print("[error] フレームを読めなかった")
        return
    # 定常部分の中央値 (最初の数フレームは初期化コストを含むため除く)
    steady = per_frame[10:] or per_frame
    med = statistics.median(steady)
    print(f"=== {args.video.name} {args.start_sec:.0f}s から {len(per_frame)} フレーム ===")
    print(f"1 フレーム中央値 (計装あり): {med:.1f}ms\n")
    print(f"{'区分':<26} {'ms/frame':>9} {'全体比':>7} {'回/frame':>9}")
    print("-" * 56)
    for key in ("classify(全体)", "  cv2.cvtColor", "  H median (折返し補正)",
                "  S median (光沢除外)", "  V median (_median_fast)"):
        if key not in _ACC:
            continue
        ms = _ACC[key] / len(per_frame) * 1000.0
        print(f"{key:<26} {ms:8.2f} {ms/med*100:6.1f}% {_CNT[key]/len(per_frame):8.1f}")
    print("-" * 56)
    inner = sum(
        _ACC.get(k, 0.0) for k in
        ("  cv2.cvtColor", "  H median (折返し補正)", "  S median (光沢除外)",
         "  V median (_median_fast)")
    ) / len(per_frame) * 1000.0
    total = _ACC.get("classify(全体)", 0.0) / len(per_frame) * 1000.0
    print(f"{'内訳の合計':<26} {inner:8.2f}")
    print(f"{'classify 内の未帰属':<26} {total - inner:8.2f}"
          f"  ← 閾値照合ループ + UIマスク + CNN + gate")
    print()
    print("Rust 化の判断材料:")
    print("  ・cv2.cvtColor が大きい → Rust 移植は丸め互換のリスクあり、慎重に")
    print("  ・median 群が大きい → 整数演算なので Rust で bit-identical 可能")
    print("  ・未帰属が大きい → 閾値照合ループが本命 (整数比較のみ、最も安全)")


if __name__ == "__main__":
    main()
