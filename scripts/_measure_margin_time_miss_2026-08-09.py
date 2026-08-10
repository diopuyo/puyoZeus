"""マージンタイム逓減を無視したことによる「おじゃま着弾の見逃し」量を測る.

## 背景 (2026-08-09 アーキ指摘 + user 伝授)
`src/state_detectors.py` の `OjamaPhaseDetector` はおじゃま判定の閾値に
**70 点固定**を使っていた。 ルール上、 おじゃまレートは一定時間後に 16 秒ごと
×0.75 で下がる (144 秒地点で 22 点)。 つまり長い試合の後半では、 相手が実際に
おじゃまを送っているのに score_delta が 70 未満で OJAMA_FALL に遷移せず、
**着弾を丸ごと見逃す** fail-silent な欠損が構造的に存在していた。

起点は user 伝授により **最初の1手から 95.5 秒**
([[reference-margin-time-from-first-move-2026-08-09]])。

## 測り方 (npz ベース = 動画デコード不要で速い)
boards_lean npz には score と t_sec が入っている。 各 side の score 差分を取り、
    「固定 70 では閾値未満 だが 実効レートなら閾値以上」
に該当する差分がどれだけあるかを数える。 これがそのまま **見逃していた
おじゃま着弾の件数**の下限になる。

読み取り専用。 認識・評価には影響しない。
出力: data/verify/margin_time_miss_2026-08-09.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.scoring import compute_effective_rate  # noqa: E402

NPZ_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
OUT_TSV = _ROOT / "data" / "verify" / "margin_time_miss_2026-08-09.tsv"
FIXED_THRESHOLD: int = 70


def _measure(npz_path: Path) -> dict:
    """1 動画分の見逃し件数を返す。"""
    d = np.load(npz_path, allow_pickle=True)
    if "score" not in d.files:
        return {}
    score = np.asarray(d["score"], dtype=float)
    t_sec = np.asarray(d["t_sec"], dtype=float)
    side = np.asarray(d["side"])
    game = np.asarray(d["game_idx"], dtype=int)

    n_hit_fixed = 0      # 固定70で拾えた着弾
    n_missed = 0         # 実効レートなら拾えるのに固定70で落としていた着弾
    n_long_games = 0
    n_games = 0
    max_elapsed = 0.0
    for g in np.unique(game):
        for s in ("1P", "2P"):
            m = (game == g) & (side == s)
            if m.sum() < 2:
                continue
            n_games += 1
            ts = t_sec[m]
            sc = score[m]
            order = np.argsort(ts)
            ts, sc = ts[order], sc[order]
            # 最初の1手 = この試合・この side の最初のスナップショット時刻を代理値に使う
            # (npz は STABLE 確定盤面のみなので、 最初の確定 = 最初の設置直後)
            first_move = float(ts[0])
            elapsed = ts - first_move
            max_elapsed = max(max_elapsed, float(elapsed[-1]))
            if elapsed[-1] > 95.5:
                n_long_games += 1
            delta = np.diff(sc)
            el = elapsed[1:]
            for dv, e in zip(delta, el):
                if dv <= 0:
                    continue
                rate = compute_effective_rate(
                    float(e), FIXED_THRESHOLD, from_first_move=True,
                )
                if dv >= FIXED_THRESHOLD:
                    n_hit_fixed += 1
                elif dv >= rate:
                    n_missed += 1
    return {
        "games": n_games, "long_games": n_long_games,
        "hit_fixed": n_hit_fixed, "missed": n_missed,
        "max_elapsed": max_elapsed,
    }


def main() -> int:
    npzs = sorted(NPZ_DIR.glob("*.npz"))
    if not npzs:
        print(f"npz が無い: {NPZ_DIR}")
        return 1
    rows = ["video\tgames\tlong_games\thit_fixed\tmissed\tmiss_rate\tmax_elapsed_sec"]
    tot = {"games": 0, "long_games": 0, "hit_fixed": 0, "missed": 0}
    for p in npzs:
        r = _measure(p)
        if not r:
            continue
        total_ev = r["hit_fixed"] + r["missed"]
        rate = r["missed"] / total_ev if total_ev else 0.0
        rows.append(
            f"{p.stem}\t{r['games']}\t{r['long_games']}\t{r['hit_fixed']}\t"
            f"{r['missed']}\t{rate:.4f}\t{r['max_elapsed']:.1f}"
        )
        for k in tot:
            tot[k] += r[k]
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
    total_ev = tot["hit_fixed"] + tot["missed"]
    print(f"動画 {len(npzs)} 本 / 試合×side {tot['games']}")
    print(f"  うち 95.5 秒を超えた試合: {tot['long_games']} "
          f"({tot['long_games'] / max(1, tot['games']):.1%})")
    print(f"  固定70 で拾えた着弾   : {tot['hit_fixed']}")
    print(f"  **見逃していた着弾**  : {tot['missed']} "
          f"({tot['missed'] / max(1, total_ev):.2%} of 全着弾)")
    print(f"出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
