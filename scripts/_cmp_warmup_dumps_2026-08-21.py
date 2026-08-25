"""暖機秒数ごとの判定値を、通しレンダ (基準) と突き合わせる (2026-08-21)。

## なぜ必要か

30先2セット動画 (117分) を区間並列でレンダするため、試合開始で区切って
並列に流す方針になった (user 指示)。試合開始なら状態機械がリセットされるので
暖機はほぼ不要だが、**HSVの自動較正は動画の最初から学習し続けている**ため、
途中から始めると履歴がなく認識が変わり得る。

だから「試合開始の少し前から処理を始めて、書き出しは試合開始から」という形にする。
**何秒前から始めれば通しレンダと一致するか**を実測で決める必要がある。

## 何を比べるか

`--dump-timeline` が出す判定値の列。同じ時刻の行同士を突き合わせ、
有利不利スコアと1P勝率が一致するかを見る。

一致の定義: 浮動小数の完全一致を第一に見て、一致しない場合は最大差・
中央差・不一致行の割合を出す (どれだけ暖機を伸ばせば収束するかの傾向を見る)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 突き合わせる列 (存在するものだけ使う)
_CMP_KEYS = ("adv_raw", "adv", "p1", "p1_last", "disp_adv", "disp_p1")
# 時刻の一致許容 (実効30fpsなので1フレーム=0.0333秒)
_T_TOL_SEC = 0.02


def _load(path: Path) -> dict:
    """dump npz を読む。"""
    z = np.load(path, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def _align(ref: dict, tgt: dict) -> "tuple[np.ndarray, np.ndarray] | None":
    """t_sec を突き合わせて、共通時刻の行番号ペアを返す。"""
    if "t_sec" not in ref or "t_sec" not in tgt:
        return None
    rt = np.asarray(ref["t_sec"], dtype=float)
    tt = np.asarray(tgt["t_sec"], dtype=float)
    ri: list[int] = []
    ti: list[int] = []
    j = 0
    for i, t in enumerate(rt):
        while j < len(tt) - 1 and tt[j] < t - _T_TOL_SEC:
            j += 1
        if abs(tt[j] - t) <= _T_TOL_SEC:
            ri.append(i)
            ti.append(j)
    return np.array(ri, dtype=int), np.array(ti, dtype=int)


def _report_one(name: str, ref: dict, tgt: dict) -> None:
    """1つの暖機設定について一致状況を出す。"""
    al = _align(ref, tgt)
    if al is None or len(al[0]) == 0:
        print(f"  {name:>6}: 突合できる行が無い (t_sec 不在または重なりゼロ)")
        return
    ri, ti = al
    print(f"  {name:>6}: 共通行 {len(ri):,} / 基準 {len(ref['t_sec']):,}")
    for key in _CMP_KEYS:
        if key not in ref or key not in tgt:
            continue
        a = np.asarray(ref[key], dtype=float)[ri]
        b = np.asarray(tgt[key], dtype=float)[ti]
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            continue
        d = np.abs(a[ok] - b[ok])
        exact = int((d == 0.0).sum())
        n = int(ok.sum())
        print(
            f"      {key:>10}: 完全一致 {exact}/{n} ({exact / n * 100:5.1f}%)"
            f"  最大差 {d.max():.6f}  中央差 {np.median(d):.6f}"
        )


def main() -> int:
    """暖機ごとの一致状況を並べる。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=Path("data/verify/zenchi_warmup_2026-08-21"))
    ap.add_argument("--ref", type=str, default="ref")
    args = ap.parse_args()

    d = PROJECT_ROOT / args.dir
    ref_path = d / f"{args.ref}.npz"
    if not ref_path.exists():
        print(f"[error] 基準が無い: {ref_path}")
        return 1
    ref = _load(ref_path)
    print(f"=== 基準 {args.ref}.npz (行数 {len(ref.get('t_sec', []))}) ===")
    print(f"    列: {sorted(ref.keys())}")
    print()

    others = sorted(p for p in d.glob("*.npz") if p.stem != args.ref)
    for p in others:
        _report_one(p.stem, ref, _load(p))
    print()
    print("--- 判定 ---")
    print("完全一致100%になった最小の暖機秒数を採用する。")
    print("100%に届かない場合は、最大差が判定に影響しない大きさか (有利不利スコアは")
    print("-100〜+100、EVEN閾値3.0) を見て、暖機を伸ばすか許容するかを決める。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
