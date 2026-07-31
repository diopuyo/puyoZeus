"""np.median のオーバーヘッド切り分け (ステップ2 の方式選定用)。

実測 (2026-07-30): h_median + s_median = 30.8 ms/frame、呼び出し 584 回/frame ずつ。
1 回あたり約 26us。**セルパッチは数百画素しかないので、これは計算量ではなく
numpy 呼び出しのオーバーヘッドが支配的**という仮説を検証する。

仮説が正しければ:
  - 方式A (セル横断ベクトル化) = 呼び出し回数を 584 → 数回に削減。効果大だが要 refactor。
  - 方式B (np.median を高速な等価実装に差し替え) = 呼び出し回数はそのまま。
    オーバーヘッドが支配的なら効果は小さいので**やるべきでない**と分かる。

本スクリプトは実パッチサイズでの 1 回あたり時間と、
同じ総画素数を「1 回のまとめ計算」にした場合の時間を比較する。
"""

from __future__ import annotations

import time

import numpy as np

# 実測のセルパッチ想定サイズ (ぷよ 1 セルは約 60x60、サブ region は更に小さい)
PATCH_SIZES: tuple[int, ...] = (16, 24, 32, 48, 60)
# 1 フレームあたりの実測呼び出し回数 (h_median / s_median それぞれ)
CALLS_PER_FRAME: int = 584
REPEAT: int = 2000


def _bench(fn, *args) -> float:
    """1 回あたりの所要秒を返す。"""
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        fn(*args)
    return (time.perf_counter() - t0) / REPEAT


def main() -> None:
    rng = np.random.default_rng(42)
    print(f"{'パッチ':>8}{'画素数':>8}{'np.median':>12}{'partition':>12}{'bincount':>12}")
    for size in PATCH_SIZES:
        patch = rng.integers(0, 180, size=(size, size), dtype=np.uint8)
        flat16 = patch.ravel().astype(np.int16)
        n = flat16.size

        t_med = _bench(np.median, flat16)

        def _part(a: np.ndarray) -> float:
            k = a.size // 2
            p = np.partition(a, [k - 1, k])
            return (float(p[k - 1]) + float(p[k])) / 2.0 if a.size % 2 == 0 else float(p[k])

        t_part = _bench(_part, flat16)

        def _binc(a: np.ndarray) -> float:
            counts = np.bincount(a, minlength=181)
            cum = np.cumsum(counts)
            half = a.size // 2
            if a.size % 2:
                return float(np.searchsorted(cum, half + 1))
            lo = float(np.searchsorted(cum, half))
            hi = float(np.searchsorted(cum, half + 1))
            return (lo + hi) / 2.0

        t_binc = _bench(_binc, patch.ravel().astype(np.int32))

        print(
            f"{size}x{size:<4}{n:>8}"
            f"{t_med * 1e6:>10.2f}us{t_part * 1e6:>10.2f}us{t_binc * 1e6:>10.2f}us"
        )

    # まとめ計算 (方式A) の効果: 全セル分を 1 回の axis 指定 median で処理
    print("\n=== 方式A (セル横断ベクトル化) の見込み ===")
    for size in (24, 32):
        stacked = np.random.default_rng(0).integers(
            0, 180, size=(CALLS_PER_FRAME, size * size), dtype=np.uint8
        ).astype(np.int16)
        t_batch = _bench(lambda a: np.median(a, axis=1), stacked)
        per_patch = np.ascontiguousarray(stacked[0])
        t_single = _bench(np.median, per_patch)
        loop_total = t_single * CALLS_PER_FRAME
        print(
            f"  パッチ{size}x{size}: 個別{CALLS_PER_FRAME}回={loop_total * 1e3:.2f}ms  "
            f"→ まとめ1回={t_batch * 1e3:.2f}ms  "
            f"削減 {100 * (loop_total - t_batch) / loop_total:.1f}%"
        )


if __name__ == "__main__":
    main()
