"""境界修正3件の効果を「同一動画ペア」で対等比較する診断スクリプト (2026-08-20)。

背景: 2026-08-19 の20本収集で勝敗ラベル欠損 38.3% -> 45.2% と悪化したと報告
されたが、比較元 (subset50、50本) と比較先 (model50、20本) で**動画セットが
異なる**ため、悪化が修正のせいなのか動画構成の違いなのかが分離できていない。
本スクリプトは両ディレクトリに共通して存在する動画だけを取り、1本ずつ
before/after を突き合わせる (母集団を固定した対等比較)。

測定する4指標 (user が毎回見ると決めた数値):
  - 勝敗ラベル欠損率 (won が NaN の行の割合)
  - 試合検出数 (game_idx のユニーク数)
  - ラッチON比率 (post_match_lockdown_active の平均)
  - 行数

修正はしない (診断専用)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2"
_BEFORE_DIR = _NPZ_DIR / "boards_lean_subset50_2026-08-19"
_AFTER_DIR = _NPZ_DIR / "boards_lean_model50_2026-08-19"

# 第1/第2引数でディレクトリ名を差し替え可能にする (既定は上記の 修正前/本番収集)。
# フラグ付き収集 (boards_lean_lockfix_2026-08-19) との比較に使う。
if len(sys.argv) >= 3:
    _BEFORE_DIR = _NPZ_DIR / sys.argv[1]
    _AFTER_DIR = _NPZ_DIR / sys.argv[2]


def _stats(path: Path) -> dict[str, float]:
    """1本の npz から4指標を取り出す。

    won は NaN 混在のため float 配列として読む。post_match_lockdown_active が
    存在しない旧構成 npz では latch を NaN 扱いにして「列なし」を明示する
    (0.0 で埋めると「ラッチが一度も立たなかった」と誤読されるため)。
    """
    d = np.load(path, allow_pickle=True)
    won = np.asarray(d["won"], dtype=float)
    rows = int(won.shape[0])
    missing = float(np.isnan(won).mean()) if rows else float("nan")
    games = int(len(np.unique(np.asarray(d["game_idx"]))))
    if "post_match_lockdown_active" in d:
        latch = float(np.asarray(d["post_match_lockdown_active"], dtype=float).mean())
    else:
        latch = float("nan")
    return {"rows": rows, "missing": missing, "games": games, "latch": latch}


def _fmt(v: float, pct: bool = False) -> str:
    """NaN を明示しつつ整形する (欠測を 0 と見分けるため)。"""
    if np.isnan(v):
        return "  n/a"
    return f"{v * 100:5.1f}%" if pct else f"{v:5.0f}"


def main() -> int:
    """共通動画だけを対で比較し、1本ごと + 合計を出力する。"""
    before = {p.name for p in _BEFORE_DIR.glob("*.npz")}
    after = {p.name for p in _AFTER_DIR.glob("*.npz")}
    common = sorted(before & after)
    if not common:
        print("[error] 共通動画が無い")
        return 1

    print(f"[paired] 共通動画 {len(common)} 本 (before={len(before)} after={len(after)})")
    print(f"  before = {_BEFORE_DIR.name}")
    print(f"  after  = {_AFTER_DIR.name}")
    print()
    header = f"{'video':>8} | {'行数(B→A)':>16} | {'won欠損(B→A)':>16} | {'試合数(B→A)':>12} | {'ラッチ(B→A)':>16}"
    print(header)
    print("-" * len(header))

    agg: dict[str, list[dict[str, float]]] = {"b": [], "a": []}
    for name in common:
        b = _stats(_BEFORE_DIR / name)
        a = _stats(_AFTER_DIR / name)
        agg["b"].append(b)
        agg["a"].append(a)
        print(
            f"{name.replace('.npz',''):>8} | "
            f"{_fmt(b['rows'])}→{_fmt(a['rows'])} | "
            f"{_fmt(b['missing'], True)}→{_fmt(a['missing'], True)} | "
            f"{_fmt(b['games'])}→{_fmt(a['games'])} | "
            f"{_fmt(b['latch'], True)}→{_fmt(a['latch'], True)}"
        )

    print("-" * len(header))
    # 合計は行数重みつき (本ごとの単純平均だと短い動画が過大評価される)
    for tag, key in (("B(修正前)", "b"), ("A(修正後)", "a")):
        rows = sum(s["rows"] for s in agg[key])
        miss = sum(s["missing"] * s["rows"] for s in agg[key]) / rows if rows else float("nan")
        games = sum(s["games"] for s in agg[key])
        latches = [s for s in agg[key] if not np.isnan(s["latch"])]
        lrows = sum(s["rows"] for s in latches)
        latch = (
            sum(s["latch"] * s["rows"] for s in latches) / lrows if lrows else float("nan")
        )
        print(f"{tag:>10} | 行数 {rows:7d} | won欠損 {_fmt(miss, True)} | 試合 {games:4d} | ラッチ {_fmt(latch, True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
