"""Rust の HSV セル分類が Python 実装と bit-identical か検証する (2026-08-20)。

fable 設計の T1 (全一致) と T3 (陽性対照) を実装する。

## なぜ合成パッチで検証するか

初版は実動画から適当に切り出した領域を使ったが、大半が背景で全セルが
COLOR_EMPTY に倒れ、**陽性対照 (閾値を1ずらす) でも不一致が出なかった** =
比較器が何も検証していない状態だった。実画像を使うなら正しい盤面座標が
必要で、それは ImageReader の内部状態に依存する。

そこで**合成パッチを主検証**にする。色レンジの内側・境界・外側を意図的に
突くので、実画像より網羅性が高く、境界ぎわの 1LSB 差も検出できる。
実画像での検証は本番経路 (npz 全列一致 = T2) が担うので二重には要らない。

## 検証の考え方

`classify` は認識の最深部にあり、1 セルでも違えば盤面が変わって学習データが
汚染される。よって「ほぼ一致」では採用できない。**全一致**が条件。
T3 (陽性対照) を先に見ること — 比較器が壊れていれば「全一致」は無意味
(測定器事故12件の対策)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

from src.image_reader import (  # noqa: E402
    EMPTY_V_THRESHOLD,
    OJAMA_S_THRESHOLD,
    OJAMA_V_MIN,
    RED_GREEN_DIFF_FOR_RED,
    RED_HUE_WRAP_CORRECTED_MAX,
    RED_HUE_WRAP_THRESHOLD,
    SPECULAR_FALLBACK_MIN_RATIO,
    SPECULAR_S_MAX,
    SPECULAR_V_MIN,
    ColorClassifier,
)

# Python 実装内のリテラル (image_reader.py の該当行から転記)
_RED_HUE_LOW_MAX = 30
_RED_BIMODAL_MIN_RATIO = 0.15
_RED_EXTEND_H_MIN = 11
_RED_EXTEND_H_MAX = 18
_SUBREGION_MIN = 4


def _params(clf: ColorClassifier, *, break_it: bool = False) -> dict:
    """Rust に渡すパラメータ束。

    break_it=True は陽性対照用に閾値を**大きく**ずらす。初版は +1 だけ
    ずらしたが、V median がちょうど境界値に乗るパッチが検体に無く不一致が
    1 件も出なかった (= 比較器の検出力を確認できなかった)。陽性対照の目的は
    「比較器が差を検出できることの確認」なので、確実に差が出る値を使う。
    """
    return {
        "s_min_scale": float(clf._s_min_scale),
        "empty_v_threshold": 200 if break_it else int(EMPTY_V_THRESHOLD),
        "ojama_s_threshold": int(OJAMA_S_THRESHOLD),
        "ojama_v_min": int(OJAMA_V_MIN),
        "red_green_diff_for_red": int(RED_GREEN_DIFF_FOR_RED),
        "red_hue_wrap_threshold": int(RED_HUE_WRAP_THRESHOLD),
        "red_hue_wrap_corrected_max": int(RED_HUE_WRAP_CORRECTED_MAX),
        "red_hue_low_max": int(_RED_HUE_LOW_MAX),
        "red_bimodal_min_ratio": float(_RED_BIMODAL_MIN_RATIO),
        "red_extend_h_min": int(_RED_EXTEND_H_MIN),
        "red_extend_h_max": int(_RED_EXTEND_H_MAX),
        "specular_v_min": int(SPECULAR_V_MIN),
        "specular_s_max": int(SPECULAR_S_MAX),
        "specular_fallback_min_ratio": float(SPECULAR_FALLBACK_MIN_RATIO),
        "enable_red_hue_wrap_fix": bool(clf._enable_red_hue_wrap_fix),
        "enable_specular_robust_saturation": bool(
            clf._enable_specular_robust_saturation
        ),
        "subregion_min_h": int(_SUBREGION_MIN),
        "subregion_min_w": int(_SUBREGION_MIN),
    }


def _ranges_flat(clf: ColorClassifier) -> np.ndarray:
    """色レンジを (R,7) に平坦化。**dict の挿入順を保つ** (先勝ち判定のため)。"""
    rows: list[list[int]] = []
    for color_code, ranges in clf._ranges.items():
        for r in ranges:
            rows.append([
                int(color_code), int(r.h_min), int(r.h_max),
                int(r.s_min), int(r.s_max), int(r.v_min), int(r.v_max),
            ])
    return np.asarray(rows, dtype=np.int32)


def _synth_patches(clf: ColorClassifier, rng: np.random.Generator) -> list[np.ndarray]:
    """色レンジの内側/境界/外側を突く合成パッチ群を作る。

    HSV で作って BGR に戻す (本番と同じ BGR 入力にするため)。パッチ内に
    ばらつきとハイライトを混ぜ、median・2峰補正・光沢除外の分岐を踏ませる。
    """
    out: list[np.ndarray] = []
    sizes = [(16, 16), (32, 32), (15, 15), (5, 5), (4, 4), (3, 3), (8, 20)]
    for row in _ranges_flat(clf):
        _, h_min, h_max, s_min, s_max, v_min, v_max = row.tolist()
        for h in {h_min, h_max, (h_min + h_max) // 2, max(0, h_min - 1), min(179, h_max + 1)}:
            for s in {s_min, s_max, (s_min + s_max) // 2, max(0, s_min - 1)}:
                for v in {v_min, v_max, (v_min + v_max) // 2, max(0, v_min - 1),
                          EMPTY_V_THRESHOLD, EMPTY_V_THRESHOLD - 1}:
                    hh, ww = sizes[rng.integers(len(sizes))]
                    hsv = np.empty((hh, ww, 3), dtype=np.uint8)
                    # ±2 のばらつきを与えて median の偶数/奇数分岐を踏ませる
                    hsv[:, :, 0] = np.clip(h + rng.integers(-2, 3, (hh, ww)), 0, 179)
                    hsv[:, :, 1] = np.clip(s + rng.integers(-2, 3, (hh, ww)), 0, 255)
                    hsv[:, :, 2] = np.clip(v + rng.integers(-2, 3, (hh, ww)), 0, 255)
                    out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    # 赤の折り返し 2 峰 (H が 0付近と170付近に二峰) を明示的に作る
    for ratio in (0.1, 0.15, 0.2, 0.5, 0.85, 0.9):
        for hh, ww in ((16, 16), (15, 15), (32, 32)):
            hsv = np.empty((hh, ww, 3), dtype=np.uint8)
            n_high = int(hh * ww * ratio)
            vals = np.array([175] * n_high + [3] * (hh * ww - n_high), dtype=np.uint8)
            rng.shuffle(vals)
            hsv[:, :, 0] = vals.reshape(hh, ww)
            hsv[:, :, 1] = 200
            hsv[:, :, 2] = 180
            out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    # 光沢ハイライト混在 (V高・S低の画素を比率を変えて混ぜる)
    for ratio in (0.0, 0.1, 0.5, 0.79, 0.81, 1.0):
        for hh, ww in ((16, 16), (5, 5), (4, 4)):
            hsv = np.empty((hh, ww, 3), dtype=np.uint8)
            n = hh * ww
            n_spec = int(n * ratio)
            s_vals = np.array([30] * n_spec + [200] * (n - n_spec), dtype=np.uint8)
            v_vals = np.array([250] * n_spec + [150] * (n - n_spec), dtype=np.uint8)
            hsv[:, :, 0] = 100
            hsv[:, :, 1] = s_vals.reshape(hh, ww)
            hsv[:, :, 2] = v_vals.reshape(hh, ww)
            out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    # 一様ランダム (網羅の保険)
    for _ in range(4000):
        hh, ww = sizes[rng.integers(len(sizes))]
        out.append(rng.integers(0, 256, (hh, ww, 3), dtype=np.uint8))
    return out


def _compare(patches: list[np.ndarray], clf: ColorClassifier, *, break_it: bool) -> tuple[int, int, list]:
    """Python 実装と Rust 実装を 1 パッチずつ突き合わせる。"""
    import puyo_core

    ranges = _ranges_flat(clf)
    params = _params(clf, break_it=break_it)
    total = 0
    bad = 0
    examples: list = []
    for patch in patches:
        bgr = np.ascontiguousarray(patch)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        rects = np.asarray([[0, 0, bgr.shape[1], bgr.shape[0]]], dtype=np.int32)
        got = int(np.asarray(puyo_core.classify_cells_hsv(bgr, hsv, rects, ranges, params))[0])
        exp = int(clf.classify(bgr))
        total += 1
        if got != exp:
            bad += 1
            if len(examples) < 10:
                h = int(np.median(hsv[:, :, 0]))
                s = int(np.median(hsv[:, :, 1]))
                v = int(np.median(hsv[:, :, 2]))
                examples.append(
                    f"期待={exp} 実際={got} 形状={bgr.shape} HSV中央値=({h},{s},{v})"
                )
    return total, bad, examples


def _run(clf: ColorClassifier, patches: list[np.ndarray], label: str) -> bool:
    """1 つのフラグ構成について T3 → T1 を実行する。"""
    print(f"\n### 構成: {label}")
    print(f"    red_hue_wrap={clf._enable_red_hue_wrap_fix} "
          f"specular={clf._enable_specular_robust_saturation} "
          f"s_min_scale={clf._s_min_scale}")

    t3_total, t3_bad, _ = _compare(patches[:600], clf, break_it=True)
    print(f"  T3 陽性対照: 比較 {t3_total} / 不一致 {t3_bad}", end="  ")
    if t3_bad == 0:
        print("✗ 比較器が不一致を検出できない → この結果は信用できない")
        return False
    print("✓ 比較器は機能")

    total, bad, ex = _compare(patches, clf, break_it=False)
    print(f"  T1 本番比較: 比較 {total} / 不一致 {bad} "
          f"({bad/max(1,total)*100:.4f}%)", end="  ")
    if bad:
        print("✗ 不一致あり")
        for e in ex:
            print(f"      {e}")
        return False
    print("✓ 全一致 (bit-identical)")
    return True


def main() -> int:
    """本番で使われる 4 通りのフラグ組み合わせすべてで検証する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    try:
        import puyo_core
        if not hasattr(puyo_core, "classify_cells_hsv"):
            print("[error] classify_cells_hsv が未ビルド")
            return 1
    except ImportError as e:
        print(f"[error] puyo_core を import できない: {e}")
        return 1

    rng = np.random.default_rng(args.seed)
    base = ColorClassifier()
    patches = _synth_patches(base, rng)
    print(f"=== 合成パッチ {len(patches)} 枚で検証 (seed={args.seed}) ===")
    print(f"色レンジ数: {len(_ranges_flat(base))}")

    ok = True
    # 本番 (RecognitionPipeline) は両補正 ON。ColorClassifier 既定は specular OFF。
    # 両方の組み合わせを検証しておく (どちらの経路でも使われうる)。
    for wrap in (True, False):
        for spec in (True, False):
            clf = ColorClassifier(
                enable_red_hue_wrap_fix=wrap,
                enable_specular_robust_saturation=spec,
            )
            if not _run(clf, patches, f"wrap={wrap}, specular={spec}"):
                ok = False
    print()
    if ok:
        print("=== 全構成で bit-identical を確認 ===")
        return 0
    print("=== 不一致あり: 採用不可 ===")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
