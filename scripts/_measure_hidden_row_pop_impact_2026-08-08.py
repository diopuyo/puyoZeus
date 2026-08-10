"""「13段目は消えない」ルールを適用すると連鎖数がどれだけ変わるか測る (2026-08-08).

user 伝授 + 公開資料で確認したルール: **13段目 (隠し段) に置かれたぷよは
4 つ繋がっても消えない**。 この性質を使う積み方が「幽霊連鎖」。

現行の `src/chain_bitboard.py` は 13 行すべてを連結判定に参加させており
(docstring に「12bit制限は採用しない」と明記)、 このルールに反していた。
`exclude_hidden_row_from_pop=True` で正しい挙動になる。

本スクリプトは **実データ (Phase L の boards_lean npz) で両者を比較**し、
影響範囲を数値で確定する。 現行運用では隠し段が「空」か「UNKNOWN」のどちらか
なので実害は限定的なはずだが、 それを推測でなく実測で確かめる
(過学習防止則: シーンでなく全域で確認する)。

出力: data/verify/hidden_row_pop_impact_2026-08-08.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_EMPTY, Board  # noqa: E402
from src.chain_bitboard import simulate_single  # noqa: E402

NPZ_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
OUT_TSV = _ROOT / "data" / "verify" / "hidden_row_pop_impact_2026-08-08.tsv"
# 1 動画あたりのサンプル数 (全数だと 89 万件で時間がかかるため等間隔抽出)。
SAMPLES_PER_VIDEO: int = 400
# 対象動画数 (先頭から。 全 123 本を回す必要はなく分布が見えれば十分)。
MAX_VIDEOS: int = 30


def _sample_grids(npz_path: Path) -> np.ndarray:
    """等間隔にサンプリングした盤面配列を返す。"""
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"]
    n = grids.shape[0]
    if n == 0:
        return grids
    idx = np.linspace(0, n - 1, min(SAMPLES_PER_VIDEO, n)).astype(int)
    return grids[idx]


def main() -> int:
    npzs = sorted(NPZ_DIR.glob("*.npz"))[:MAX_VIDEOS]
    if not npzs:
        print(f"npz が無い: {NPZ_DIR}")
        return 1
    rows = ["video_id\tsamples\thidden_nonempty\tdiff_count\tdiff_rate\tmax_abs_diff"]
    tot_samples = tot_diff = tot_hidden = 0
    max_diff_overall = 0
    for p in npzs:
        grids = _sample_grids(p)
        n_diff = 0
        n_hidden = 0
        max_abs = 0
        for g in grids:
            board = Board.from_list([list(map(int, r)) for r in g])
            # 隠し段に何か入っているか (空でない = 影響が出うる)
            if any(board.get(0, c) != COLOR_EMPTY for c in range(BOARD_COLS)):
                n_hidden += 1
            old = int(simulate_single(board).chain_count)
            new = int(simulate_single(
                board, exclude_hidden_row_from_pop=True,
            ).chain_count)
            if old != new:
                n_diff += 1
                max_abs = max(max_abs, abs(old - new))
        n = int(grids.shape[0])
        rate = n_diff / n if n else 0.0
        rows.append(
            f"{p.stem}\t{n}\t{n_hidden}\t{n_diff}\t{rate:.6f}\t{max_abs}"
        )
        tot_samples += n
        tot_diff += n_diff
        tot_hidden += n_hidden
        max_diff_overall = max(max_diff_overall, max_abs)
        print(f"  {p.stem:10s} n={n:4d} 隠し段非空={n_hidden:4d} 差={n_diff:4d}")

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n=== 合計 ===")
    print(f"サンプル数 {tot_samples}")
    print(f"隠し段が非空 {tot_hidden} ({tot_hidden / max(1, tot_samples):.2%})")
    print(f"連鎖数が変わった {tot_diff} ({tot_diff / max(1, tot_samples):.4%})")
    print(f"最大の差 {max_diff_overall} 連鎖")
    print(f"出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
