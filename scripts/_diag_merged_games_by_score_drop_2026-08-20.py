"""2つの試合を1つに繋げた見逃しを「スコアの恒久的な減少」から検出する
(2026-08-20、user 指示)。

試合境界を「score が 0 になった」ときだけ認める方式 (--enable-score-reset-
requires-zero) にすると、0 の検知に失敗したときに**本物の区切りを見逃して
2試合が1つの game_idx に繋がる**という逆方向の失敗が起こりうる。

その痕跡は score に必ず残る: 試合中の score は単調増加しかしないので、
同じ game_idx の中でスコアが大きく下がり、**下がったまま戻らない**なら、
そこが見逃した試合の切れ目である。

一時的な誤読 (実測: 5,759->5,259 のあと次の行で 5,899 に戻る) と区別する
ため、減少幅ではなく**恒久性**で判定する。減少幅は桁の位置しだいで
いくらでも変わる (1桁誤読でも -70,000 になりうる) が、誤読は次に読めた
行で必ず戻るため、持続の方が本質的で閾値も物理的に決めやすい。

判定: game_idx ごとに running_max (それまでの最大値) を追い、
「score < running_max」が MIN_PERSIST_ROWS 行以上連続したら、その先頭を
見逃した境界の候補として報告する。

修正はしない (診断専用)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2"

# 誤読は「次に読めた行」で戻る (実測2件とも1行で復帰) ため、数行の持続を
# 求めれば単発の誤読は落ちる。一方、試合を繋げた場合は新しい試合が続く限り
# ずっと低いままなので、この程度の行数は必ず超える。
MIN_PERSIST_ROWS: int = 5

# 持続だけでは足りない: 同じ誤読が続くケースが実在する (2026-08-20 実測、
# 39番で5件。例 55,994 -> 55,002 が20行continue)。これらは全て「わずかに
# 低いまま」であり、比 (減少後 / 直前最大) は 0.86〜0.998 だった。
# 対して試合を繋げた場合、新しい試合は score 0 から始まるので比は 0 近傍に
# なる。両者は大きく離れているため、「半分以下に落ちた」を条件にすれば
# 誤読を除外できる。単調増加するはずの score が半減するのは、試合が
# 変わった以外に物理的な説明がつかない (シーン逆算ではなく、単調増加と
# いうゲーム仕様から決めた値)。
MAX_DROP_RATIO: float = 0.5


def _scan_side(t: np.ndarray, s: np.ndarray) -> list[tuple[float, int, int, int]]:
    """1つの (game_idx, side) 系列から恒久的な減少の開始点を返す。

    Returns:
        [(開始時刻, 直前までの最大値, 減少後の値, 継続行数), ...]
    """
    order = np.argsort(t)
    t, s = t[order], s[order]
    hits: list[tuple[float, int, int, int]] = []
    running_max = -1
    run_start: int | None = None
    for i, v in enumerate(s):
        if np.isnan(v):
            continue
        v = int(v)
        if running_max >= 0 and v < running_max:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                _emit(hits, t, s, run_start, i - run_start, running_max)
                run_start = None
            running_max = max(running_max, v)
    if run_start is not None:
        _emit(hits, t, s, run_start, len(s) - run_start, running_max)
    return hits


def _emit(
    hits: list[tuple[float, int, int, int]],
    t: np.ndarray, s: np.ndarray, start: int, n: int, running_max: int,
) -> None:
    """持続 + 落差の両条件を満たす区間だけを候補として積む。"""
    if n < MIN_PERSIST_ROWS or running_max <= 0:
        return
    v = int(s[start])
    if v / running_max > MAX_DROP_RATIO:
        return  # わずかな低下 = 誤読の持続 (試合の繋ぎではない)
    hits.append((float(t[start]), running_max, v, n))


def _analyze(path: Path) -> int:
    """1本の npz を走査し、繋ぎ見逃し候補を出す。戻り値は候補数。"""
    d = np.load(path, allow_pickle=True)
    game = np.asarray(d["game_idx"])
    side = np.asarray(d["side"])
    t = np.asarray(d["t_sec"], dtype=float)
    s = np.asarray(d["score"], dtype=float)

    total = 0
    rows: list[str] = []
    for g in sorted(np.unique(game).tolist()):
        for sd in ("1P", "2P"):
            sel = (game == g) & (side == sd)
            if sel.sum() < MIN_PERSIST_ROWS:
                continue
            for t0, mx, v, n in _scan_side(t[sel], s[sel]):
                total += 1
                rows.append(
                    f"    試合{g:>3} {sd}  t={t0:7.1f}s  "
                    f"{mx:>8,} -> {v:>8,} が {n:>3}行continue"
                )
    print(f"\n=== {path.stem}: 繋ぎ見逃し候補 {total} 件 "
          f"(試合{len(np.unique(game))} 行{len(t)}) ===")
    for r in rows[:20]:
        print(r)
    if len(rows) > 20:
        print(f"    ... 他 {len(rows) - 20} 件")
    return total


def main() -> int:
    """usage: <npz_dir> [<target_id> ...]  (target 省略で全件)"""
    if len(sys.argv) < 2:
        print("usage: <npz_dir> [<target_id> ...]")
        return 1
    npz_dir = _NPZ_DIR / sys.argv[1]
    targets = sys.argv[2:]
    paths = (
        [npz_dir / f"{x}.npz" for x in targets] if targets
        else sorted(npz_dir.glob("*.npz"))
    )
    grand = 0
    for p in paths:
        if p.exists():
            grand += _analyze(p)
        else:
            print(f"[skip] なし: {p}")
    print(f"\n合計 {grand} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
