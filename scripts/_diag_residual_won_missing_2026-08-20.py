"""配線是正後に残る勝敗ラベル欠損の所在を特定する診断 (2026-08-20)。

境界修正3フラグの配線是正後も、勝敗ラベル (won) が付かない行が残っている
(2本平均 9.8%、39 は 19.3%、38 は 0.2%)。その残りが
  (a) 動画の端 (冒頭イントロ / 最終試合後) に偏っているのか
  (b) 中間の試合にも散っているのか
  (c) 特定の試合が丸ごと落ちているのか
を切り分ける。原因を推測せず所在を数値で確定させるのが目的で、修正はしない。

判定の考え方: 試合単位で「その試合の行が全部欠損」なら試合まるごと勝敗不明
(パネル読み取り or 境界の問題)、「一部だけ欠損」なら試合の切れ目のズレ。
端点かどうかは試合番号が最初/最後かで見る。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2"


def _analyze(path: Path) -> None:
    """1本の npz について、欠損行が どの試合に どう分布するかを出す。"""
    d = np.load(path, allow_pickle=True)
    won = np.asarray(d["won"], dtype=float)
    game = np.asarray(d["game_idx"])
    t = np.asarray(d["t_sec"], dtype=float)
    miss = np.isnan(won)
    games = sorted(np.unique(game).tolist())

    print(f"\n=== {path.stem} : 全{len(won)}行 欠損{miss.sum()}行 ({miss.mean()*100:.1f}%) "
          f"試合数{len(games)} ===")

    full, part, clean = [], [], 0
    for g in games:
        sel = game == g
        m = miss[sel]
        if m.all():
            full.append(g)
        elif m.any():
            part.append((g, int(m.sum()), int(sel.sum())))
        else:
            clean += 1

    print(f"  丸ごと欠損の試合 : {len(full)}本 {full[:12]}{'...' if len(full) > 12 else ''}")
    print(f"  一部だけ欠損     : {len(part)}本")
    print(f"  完全に取れた試合 : {clean}本")

    # 丸ごと欠損が「端点」に寄っているかを見る (最初/最後の試合番号との距離)
    if full:
        rows_full = sum(int((game == g).sum()) for g in full)
        first, last = games[0], games[-1]
        at_edge = [g for g in full if g in (first, last)]
        print(f"  → 丸ごと欠損の行数 = {rows_full} (全欠損の{rows_full/max(1,miss.sum())*100:.0f}%)")
        print(f"  → うち端点の試合 (先頭{first}/末尾{last}) : {at_edge}")
        # 時間帯も出す (冒頭・末尾に寄っているかの裏取り)
        for g in full[:6]:
            sel = game == g
            print(f"     試合{g}: {t[sel].min():7.1f}s 〜 {t[sel].max():7.1f}s "
                  f"({int(sel.sum())}行)")
    if part:
        rows_part = sum(n for _, n, _ in part)
        print(f"  → 一部欠損の行数 = {rows_part} (全欠損の{rows_part/max(1,miss.sum())*100:.0f}%)")
        for g, n, tot in part[:6]:
            print(f"     試合{g}: {n}/{tot}行 欠損")


def main() -> int:
    """引数で渡された npz (ディレクトリ名/ファイル名) を順に診断する。"""
    targets = sys.argv[1:]
    if not targets:
        print("usage: <dir>/<name>.npz ...")
        return 1
    for spec in targets:
        p = _NPZ_DIR / spec
        if not p.exists():
            print(f"[skip] 見つからない: {p}")
            continue
        _analyze(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
