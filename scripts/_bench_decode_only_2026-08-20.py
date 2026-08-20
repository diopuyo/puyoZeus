"""動画デコードだけの速度を測り、ボトルネックの所在を確定する (2026-08-20)。

収集が 14並列で1本あたり単独の4.3倍に劣化している。CNN は GPU で動いており
(GPU34%、VRAM139MiB/本) 律速ではないため、CPU 側の何が重いかを切り分ける。

fable の指摘: `scripts/collect_boards_lean.py:2146-2154` は `cap.read()` を
**全フレーム**呼び、認識は stride-2 (normalize_fps_30=True が既定) なので
60fps 動画では半分のフレームをデコード+色変換してから捨てている。

本スクリプトは3条件を同一動画・同一区間で比較する:
  A. read()  : デコード + retrieve (色変換+コピー) を全フレーム
  B. grab()  : デコードのみ (retrieve を省く)、認識対象だけ retrieve
  C. read() を1フレームおきに retrieve (現行と同じ形の対照)

これで「デコードが支配的か」「retrieve の無駄が効いているか」が分かる。
認識処理は一切呼ばないので、純粋な入力コストの上限が測れる。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402

cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRAMES = PROJECT_ROOT / "data" / "frames"

_N_FRAMES = 3000  # 測る本数 (60fps なら 50秒ぶん)


def _bench(path: Path, mode: str, n: int) -> tuple[float, int]:
    """指定モードで n フレーム分進め、所要秒と実際に進んだ数を返す。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1.0, 0
    # 先頭のイントロを避けて中盤から測る (試合中の映像で測る)
    cap.set(cv2.CAP_PROP_POS_MSEC, 600_000)
    t0 = time.monotonic()
    got = 0
    try:
        for i in range(n):
            if mode == "read":
                ok, _ = cap.read()
            elif mode == "grab":
                ok = cap.grab()
            else:  # grab+retrieve1of2 : 現行相当 (半分だけ画像を取り出す)
                ok = cap.grab()
                if ok and i % 2 == 0:
                    ok, _ = cap.retrieve()
            if not ok:
                break
            got += 1
    finally:
        cap.release()
    return time.monotonic() - t0, got


def main() -> int:
    """usage: <target_id> [n_frames]"""
    tid = sys.argv[1] if len(sys.argv) > 1 else "c34"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else _N_FRAMES
    path = _FRAMES / f"video_{tid}.mp4"
    if not path.exists():
        print(f"[error] 動画なし: {path}")
        return 1

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    print(f"=== {path.name} : {fps:.0f}fps {total:,.0f}frames "
          f"({total/max(fps,1e-9)/60:.1f}分) ===")
    print(f"    {n} フレーム分を計測 (中盤600秒地点から)\n")
    print(f"{'方式':>22} {'所要':>8} {'実効fps':>10}")
    print("-" * 44)

    labels = {
        "read": "read (現行: 全frame取出)",
        "grab": "grab のみ (取出なし)",
        "half": "grab + 半分だけ取出",
    }
    res: dict[str, float] = {}
    for mode in ("read", "grab", "half"):
        sec, got = _bench(path, mode, n)
        eff = got / sec if sec > 0 else 0.0
        res[mode] = eff
        print(f"{labels[mode]:>22} {sec:7.1f}s {eff:9.1f}fps")

    print("-" * 44)
    if res.get("read", 0) > 0:
        print(f"  grab のみ / read     = {res['grab']/res['read']:.2f}倍")
        print(f"  半分取出 / read      = {res['half']/res['read']:.2f}倍")
    print()
    print("  参考: 収集全体の実効は 1本あたり約10.5fps (14並列時の実測)。")
    print("  デコードだけでこれに近い数字なら入力コストが支配的で、")
    print("  認識処理を速くしても効かない (デコード側の改善が本命)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
