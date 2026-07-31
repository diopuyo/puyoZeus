"""game_idx desync による CNN 学習ペアの脱落量を既存 npz で実測する (2026-07-31)。

## 追跡で確定した経路の違い

- **指標/AUC 側 (labeled_win.csv)**: `label_win_from_winners.py` は game_idx を
  使わず **時刻区間 `s <= t_sec < e` でマッチ**する (docstring に明記)。
  → **game_idx desync の影響を受けない。**
- **CNN 側 (board_pairs_lean)**: `build_board_pairs_lean.py` は
  「labeled_win.csv 不要: npz 内蔵の won を直接使用」と明記し、
  さらに `groupby(["video_id", "game_idx"])` でグループ化して
  その中で 1P/2P を近傍 t_sec (許容 MAX_PAIR_T_DIFF_SEC=2.0s) でペア化する。
  → **二重に影響**: (a) npz 内蔵 won が別試合同士の突き合わせで決まりうる
     (b) desync すると同じ game_idx の 1P/2P が別時間帯を指し、
        2 秒の許容差を超えて **ペアが黙って消える**

## 本スクリプトが測るもの

既存 npz について (video_id, game_idx) ごとに:
  - 1P/2P の時刻範囲の重なり
  - 実際のペア化アルゴリズム (貪欲 + 2秒許容) を再現してペア成立数を数える
  - **ペアが 1 つも作れなかったグループ** = desync による全損グループ

「ペア数 / 理論上の上限 (min(1P行数, 2P行数))」で歩留まりを出す。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

# build_board_pairs_lean.MAX_PAIR_T_DIFF_SEC と同値
MAX_PAIR_T_DIFF_SEC: float = 2.0
# 「大きくずれている」と判定する閾値 [秒]
DESYNC_THRESHOLD_SEC: float = 5.0


def _pair_count(t1: np.ndarray, t2: np.ndarray) -> int:
    """build_board_pairs_lean と同じ貪欲マッチでペア成立数を数える。"""
    a = np.sort(t1)
    b = np.sort(t2)
    j = 0
    n = 0
    for t in a:
        while j + 1 < len(b) and abs(b[j + 1] - t) < abs(b[j] - t):
            j += 1
        if j >= len(b):
            break
        if abs(b[j] - t) <= MAX_PAIR_T_DIFF_SEC:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz-dir", type=Path,
        default=Path("data/indicators_v2/boards_lean_fixed"),
        help="既存の boards_lean npz ディレクトリ (CNN 学習の入力)",
    )
    ap.add_argument("--limit", type=int, default=0, help="先頭 N ファイルのみ (0=全件)")
    args = ap.parse_args()

    files = sorted(args.npz_dir.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"npz が無い: {args.npz_dir}")
        return
    print(f"対象: {args.npz_dir} ({len(files)} ファイル)")
    print(f"ペア許容差 {MAX_PAIR_T_DIFF_SEC}s / ずれ判定 {DESYNC_THRESHOLD_SEC}s\n")

    tot_groups = 0
    tot_dead_groups = 0
    tot_pairs = 0
    tot_ceiling = 0
    gaps: list[float] = []
    per_video_dead: dict[str, tuple[int, int]] = {}
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:  # 壊れた npz は skip して続行
            print(f"[skip] {f.name}: {e}")
            continue
        if "game_idx" not in d or "side" not in d or "t_sec" not in d:
            continue
        gidx = np.asarray(d["game_idx"])
        side = np.asarray(d["side"]).astype(str)
        tsec = np.asarray(d["t_sec"]).astype(float)
        vid = str(np.asarray(d["video_id"]).astype(str)[0]) if len(gidx) else f.stem

        buckets: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"1P": [], "2P": []},
        )
        for g, s, t in zip(gidx, side, tsec):
            if s in ("1P", "2P"):
                buckets[int(g)][s].append(float(t))

        v_groups = 0
        v_dead = 0
        for g, sides in sorted(buckets.items()):
            t1 = np.asarray(sides["1P"])
            t2 = np.asarray(sides["2P"])
            if t1.size == 0 or t2.size == 0:
                # 片側しか無いグループ = そもそもペアが作れない (desync の極端形)
                tot_groups += 1
                v_groups += 1
                tot_dead_groups += 1
                v_dead += 1
                tot_ceiling += 0
                continue
            tot_groups += 1
            v_groups += 1
            ceiling = int(min(t1.size, t2.size))
            n_pairs = _pair_count(t1, t2)
            tot_ceiling += ceiling
            tot_pairs += n_pairs
            gaps.append(abs(float(t1.min()) - float(t2.min())))
            if n_pairs == 0:
                tot_dead_groups += 1
                v_dead += 1
        per_video_dead[vid] = (v_dead, v_groups)

    print(f"{'項目':<44}{'値':>18}")
    print("-" * 62)
    print(f"{'(video, game_idx) グループ数':<44}{tot_groups:>18}")
    print(
        f"{'うちペアが1つも作れなかったグループ':<44}"
        f"{tot_dead_groups:>10} ({100.0 * tot_dead_groups / max(1, tot_groups):.1f}%)"
    )
    print(f"{'成立ペア数':<44}{tot_pairs:>18}")
    print(f"{'理論上限 sum(min(1P,2P))':<44}{tot_ceiling:>18}")
    if tot_ceiling:
        print(
            f"{'歩留まり (成立/上限)':<44}"
            f"{100.0 * tot_pairs / tot_ceiling:>17.1f}%"
        )
        print(
            f"{'★脱落したペア':<44}"
            f"{tot_ceiling - tot_pairs:>10} "
            f"({100.0 * (tot_ceiling - tot_pairs) / tot_ceiling:.1f}%)"
        )
    if gaps:
        arr = np.asarray(gaps)
        over = int((arr > DESYNC_THRESHOLD_SEC).sum())
        print(
            f"\n1P/2P 開始時刻差: 中央 {float(np.median(arr)):.2f}s / "
            f"最大 {float(arr.max()):.2f}s / "
            f"{DESYNC_THRESHOLD_SEC:.0f}s超 {over}/{arr.size} "
            f"({100.0 * over / arr.size:.1f}%)"
        )
    worst = sorted(
        ((v, d, g) for v, (d, g) in per_video_dead.items() if g),
        key=lambda x: -(x[1] / x[2]),
    )[:8]
    if worst:
        print("\n全損グループ比率の悪い動画 上位:")
        for v, d, g in worst:
            print(f"  {v:<14}{d:>4}/{g:<4} ({100.0 * d / g:.0f}%)")


if __name__ == "__main__":
    main()
