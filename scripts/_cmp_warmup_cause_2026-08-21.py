"""暖機の残差6%が「盤面のずれ」か「判定のずれ」かを切り分ける (2026-08-21)。

## なぜ必要か

区間並列レンダの成立性を測ったところ、暖機5秒で判定値の94%が通しレンダと
完全一致するが、**残る6%が暖機を伸ばしても消えない**ことが分かった
(15秒でも26秒でも94〜95%で頭打ち)。最大差5.11は互角の閾値3.0を超えるので、
判定の色が変わる場面があり得る。

原因が「認識側 (盤面がずれている)」なら区間並列そのものの限界で、
HSVの較正結果を先に計算して渡す等の対策が必要。
「判定側 (盤面は同じなのに数値がずれている)」なら内部状態 (EMA等) の
引き継ぎ問題で、対策が変わる。

## どう切り分けるか

dump には盤面のハッシュ (`b1_hash`/`b2_hash`) と状態機械の状態
(`state1`/`state2`)、それに認識の出力そのもの (`score1`/`score2`/
`pending_p1`/`pending_p2`) が入っている。

判定値が不一致だった行について、これらが一致しているかを見る:
- 盤面ハッシュが違う → **認識側のずれ**
- 盤面ハッシュは同じで判定値が違う → **判定側のずれ**
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_T_TOL_SEC = 0.02
# 認識側の出力 (これがずれていれば認識のずれ)
_RECOG_KEYS = ("b1_hash", "b2_hash", "state1", "state2",
               "score1", "score2", "pending_p1", "pending_p2")
_EVEN_THRESHOLD = 3.0  # 互角判定の閾値 (CLAUDE.md)


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def _align(rt: np.ndarray, tt: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """t_sec を突き合わせて共通行の番号ペアを返す。"""
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


def _cmp_col(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """列の一致マスクを返す (数値でも文字列でも扱える)。"""
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        af = a.astype(float)
        bf = b.astype(float)
        both_nan = ~np.isfinite(af) & ~np.isfinite(bf)
        return (af == bf) | both_nan
    return a.astype(str) == b.astype(str)


def _analyze(name: str, ref: dict, tgt: dict) -> None:
    """1設定について、判定不一致行の内訳を出す。"""
    ri, ti = _align(np.asarray(ref["t_sec"], dtype=float),
                    np.asarray(tgt["t_sec"], dtype=float))
    if len(ri) == 0:
        print(f"  {name}: 突合できる行が無い")
        return
    a = np.asarray(ref["adv_raw"], dtype=float)[ri]
    b = np.asarray(tgt["adv_raw"], dtype=float)[ti]
    ok = np.isfinite(a) & np.isfinite(b)
    diff = np.zeros(len(a), dtype=bool)
    diff[ok] = a[ok] != b[ok]
    n_diff = int(diff.sum())
    print(f"\n  === {name}: 共通行 {len(ri)} / 判定が不一致 {n_diff} "
          f"({n_diff / len(ri) * 100:.1f}%) ===")
    if n_diff == 0:
        return

    # 互角の閾値を跨ぐか (判定の色が変わる場面)
    sign_ref = np.where(np.abs(a) <= _EVEN_THRESHOLD, 0, np.sign(a))
    sign_tgt = np.where(np.abs(b) <= _EVEN_THRESHOLD, 0, np.sign(b))
    flip = ok & (sign_ref != sign_tgt)
    print(f"    互角±{_EVEN_THRESHOLD} を跨いで判定が変わる行: {int(flip.sum())} "
          f"({int(flip.sum()) / len(ri) * 100:.2f}%)")

    # 認識側の列が一致しているか (不一致行に限って集計)
    print("    判定が不一致だった行で、認識側の列は一致しているか:")
    for key in _RECOG_KEYS:
        if key not in ref or key not in tgt:
            continue
        m = _cmp_col(np.asarray(ref[key])[ri][diff],
                     np.asarray(tgt[key])[ti][diff])
        n_ok = int(m.sum())
        verdict = "一致" if n_ok == n_diff else "**ずれあり**"
        print(f"      {key:>12}: {n_ok}/{n_diff} 一致  {verdict}")

    # 不一致行の時刻分布 (較正の注入と関係するか見るため)
    t = np.asarray(ref["t_sec"], dtype=float)[ri][diff]
    print(f"    不一致行の時刻: 最小 {t.min():.1f}s 最大 {t.max():.1f}s "
          f"中央 {np.median(t):.1f}s")
    print(f"      先頭10件: {' '.join(f'{x:.1f}' for x in t[:10])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=Path("data/verify/zenchi_warmup_2026-08-21"))
    ap.add_argument("--ref", type=str, default="ref")
    args = ap.parse_args()

    d = PROJECT_ROOT / args.dir
    ref = _load(d / f"{args.ref}.npz")
    print("=== 暖機の残差は認識側か判定側か ===")
    print("盤面ハッシュ(b1_hash/b2_hash)がずれていれば認識側、")
    print("一致しているのに判定値がずれていれば判定側の内部状態が原因。")
    for p in sorted(x for x in d.glob("*.npz") if x.stem != args.ref):
        _analyze(p.stem, ref, _load(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
