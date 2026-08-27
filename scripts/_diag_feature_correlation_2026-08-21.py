"""判定に使う数値どうしの相関を実測する (2026-08-21)。

## なぜ必要か

これまで数値の有効性を `permutation importance` (その列だけをシャッフルして
精度がどれだけ落ちるか) だけで判断していた。しかし **組み合わせで効く数値は
この方法では検出できない**。他の列が代わりを務めるため、シャッフルしても
精度が落ちないからである (user 指摘 2026-08-21「単独で見るものではない」)。

具体例:
  - `match_progress` (局面) は両者で同じ値なので単独では勝敗を識別できない。
    掛け算の相手があって初めて意味を持つ
  - `ojama_damage_forecast` は `ojama_forecast` と強く相関するはずで、片方を
    シャッフルしてももう片方が情報を持っているため落ちない

## 何を出すか

1. **強相関ペア** (|r| が大きい組) — 情報が重複している = 共食い。
   捨てるのではなく統合の対象。どちらを残すかは貢献度と解釈しやすさで決める
2. **過去に有効だった数値の相関** — 148本で上位だったのに今の構成で消えた
   数値が、何に吸われたのかを特定する
3. **新しく足した2列の相関** — 既存の予告6列とどれだけ重複しているか

相関だけでは因果は分からないので、あくまで「統合候補の洗い出し」に使う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 148本 (2026-08-14) で上位だったのに、62本の構成で消えた数値。
# 「動画数が減ったせい」説は 47本→61本で回復しなかったため否定済み
# (2026-08-21 実測)。ならば何かに吸われている可能性を相関で確かめる。
_LOST_IN_NEW_BUILD = (
    "color_ojama_ratio_own",          # 148本 rank4  → 62本 rank21
    "color_diversity_evenness",       # 148本 rank8  → 62本 rank22
    "diff_conn_max_group_size",       # 148本 rank10 → 62本 圏外
    "color_diff_x_ojama_diff",        # 148本 rank22 → 62本 rank35
)

# 今回追加した2列 (単独の貢献度は誤差以下だった)
_NEW_COLS = ("match_progress", "ojama_damage_forecast")

# 予告おじゃま系 (6通りの表現があり共食いの疑いが濃い)
_FORECAST_FAMILY = (
    "ojama_forecast", "ojama_forecast_uncapped", "ojama_forecast_log",
    "ojama_forecast_progress_interaction", "ojama_damage_forecast",
    "ojama_net_balance", "ojama_net_balance_uncapped",
    "ojama_net_balance_synced", "ojama_margin",
)

# 連結系 (相互に強相関のはずで重要度が割れている疑い)
_CONN_FAMILY = (
    "main_linked_pair_count", "main_linked_ratio", "conn_triple_count",
    "diff_conn_pair_count", "isolated_pair_count", "conn_pair_count",
    "conn_max_group_size", "diff_conn_max_group_size",
)

_META = frozenset({
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won",
})


def _load(csv_path: Path, sample: int) -> pd.DataFrame:
    """学習CSVを読み、判定に使う数値の列だけを返す (行はサンプリング)。"""
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c not in _META]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    if sample and len(num) > sample:
        num = num.sample(n=sample, random_state=20260821)
    return num


def _report_pairs(corr: pd.DataFrame, thresh: float, limit: int) -> None:
    """|r| が閾値以上のペアを強い順に出す。"""
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[float, str, str]] = []
    for a in corr.columns:
        for b in corr.columns:
            if a >= b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= thresh:
                rows.append((abs(r), a, b))
    rows.sort(reverse=True)
    print(f"  |r| >= {thresh} のペア: {len(rows)} 組")
    for ar, a, b in rows[:limit]:
        r = corr.loc[a, b]
        print(f"    r={r:+.3f}  {a}  <->  {b}")
    if len(rows) > limit:
        print(f"    ... 他 {len(rows) - limit} 組")


def _report_target(corr: pd.DataFrame, targets: tuple[str, ...], top: int) -> None:
    """指定した数値それぞれについて、最も相関が強い相手を出す。"""
    for t in targets:
        if t not in corr.columns:
            print(f"  {t}: (この構成に列が無い)")
            continue
        s = corr[t].drop(labels=[t]).dropna()
        s = s.reindex(s.abs().sort_values(ascending=False).index)
        head = s.head(top)
        print(f"  {t}:")
        for name, r in head.items():
            print(f"    r={r:+.3f}  {name}")


def main() -> int:
    """相関を測って統合候補を洗い出す。"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path,
        default=Path("data/verify/labeled_win_model62_2col_2026-08-21/labeled_win_model62_2col.csv"),
    )
    ap.add_argument("--sample", type=int, default=120000,
                    help="行のサンプル数 (0で全行)")
    ap.add_argument("--thresh", type=float, default=0.90)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[error] CSV が無い: {args.csv}")
        return 1
    num = _load(args.csv, args.sample)
    print(f"=== {args.csv.name} ===")
    print(f"数値の列 {len(num.columns)} / 使った行 {len(num):,}\n")

    corr = num.corr(method="pearson")

    print("--- 1. 強相関ペア (情報が重複している = 統合候補) ---")
    _report_pairs(corr, args.thresh, limit=25)

    print("\n--- 2. 今回追加した2列は何と重複しているか ---")
    _report_target(corr, _NEW_COLS, top=6)

    print("\n--- 3. 148本で上位だったのに消えた数値は何に吸われたか ---")
    _report_target(corr, _LOST_IN_NEW_BUILD, top=6)

    print("\n--- 4. 予告おじゃま系の相関行列 (6通りの表現の重複度) ---")
    fam = [c for c in _FORECAST_FAMILY if c in corr.columns]
    if fam:
        sub = corr.loc[fam, fam]
        print("     " + "".join(f"{i:>7}" for i in range(len(fam))))
        for i, a in enumerate(fam):
            cells = "".join(f"{sub.loc[a, b]:+7.2f}" for b in fam)
            print(f"  {i:>2} {cells}  {a}")

    print("\n--- 5. 連結系の相関行列 (重要度が割れている疑い) ---")
    fam2 = [c for c in _CONN_FAMILY if c in corr.columns]
    if fam2:
        sub2 = corr.loc[fam2, fam2]
        print("     " + "".join(f"{i:>7}" for i in range(len(fam2))))
        for i, a in enumerate(fam2):
            cells = "".join(f"{sub2.loc[a, b]:+7.2f}" for b in fam2)
            print(f"  {i:>2} {cells}  {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
