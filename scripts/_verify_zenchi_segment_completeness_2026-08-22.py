"""全編再走査の各区間dumpが要求終点まで完走したかを検証する (2026-08-22)。

背景: `scripts/visualize_advantage_overlay.py:4893-4895` の `cap.read()`
失敗時サイレント break により、区間5が要求終点4379.5秒に対し実測4340.8秒
(約39秒=約660フレーム不足) で打ち切られていたことが発覚 (2026-08-22)。
dump 行数はサンプリング仕様上 duration*30fps と一致しないため、
「dump 内の最終 t_sec が要求終点にどれだけ近いか」で完走を判定する
(サイレントbreak自体の修正は別課題として温存、本スクリプトは検知のみ)。

使い方:
    python -m scripts._verify_zenchi_segment_completeness_2026-08-22 \
        --npz path/to/segNN.npz --expected-end-sec 4379.5

終了コード: 0=完走 (許容誤差内)、1=不足 (要再走査)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 許容誤差の根拠 (シーン逆算禁止、feedback_overfitting_awareness_2026-08-04):
# 旧基準 (data/verify/zenchi_render_2026-08-21、112エピソード基準・正常完走
# 済み) の8区間全数を実測したところ、要求終点との差は 0.03〜7.53 秒の範囲
# だった (末尾dump書き出しの仕様上の間引きによる自然なばらつき、故障ではない)。
# 一方、故障していた区間5 (2026-08-22) は 38.67 秒 (約660フレーム) 不足して
# いた。正常範囲の最大値 (7.53秒) に安全マージンを載せ、10.0 秒を閾値とする
# (このバグの39秒からの逆算ではなく、正常8区間の実測範囲からの決定)。
DEFAULT_TOLERANCE_SEC: float = 10.0


def verify_segment(npz_path: Path, expected_end_sec: float,
                    tolerance_sec: float = DEFAULT_TOLERANCE_SEC) -> bool:
    """1区間の dump が要求終点まで完走しているか判定する。

    Args:
        npz_path: dump-timeline で書き出した npz ファイル。
        expected_end_sec: この区間の要求終点 (--end-sec に渡した値)。
        tolerance_sec: 許容誤差 (秒)。

    Returns:
        True = 完走 (last_t_sec >= expected_end_sec - tolerance_sec)。
    """
    if not npz_path.exists():
        print(f"[NG] {npz_path}: ファイル不在")
        return False
    d = np.load(npz_path, allow_pickle=True)
    t_sec = d["t_sec"]
    if len(t_sec) == 0:
        print(f"[NG] {npz_path}: dump 0 行 (空)")
        return False
    last_t = float(t_sec[-1])
    n_rows = len(t_sec)
    shortfall = expected_end_sec - last_t
    if shortfall > tolerance_sec:
        print(
            f"[NG] {npz_path}: 実測終点={last_t:.2f}s 要求終点="
            f"{expected_end_sec:.2f}s 不足={shortfall:.2f}s "
            f"(dump行数={n_rows}) → 再走査が必要"
        )
        return False
    print(
        f"[OK] {npz_path}: 実測終点={last_t:.2f}s 要求終点="
        f"{expected_end_sec:.2f}s (差={shortfall:.2f}s、dump行数={n_rows})"
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--expected-end-sec", type=float, required=True)
    ap.add_argument("--tolerance-sec", type=float, default=DEFAULT_TOLERANCE_SEC)
    a = ap.parse_args()
    ok = verify_segment(a.npz, a.expected_end_sec, a.tolerance_sec)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
