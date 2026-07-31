"""フェーズ0診断: build_ceiling_chain (XII-1b) の目的関数が逆転していないか実証する。

アーキ仮説: `_build_ceiling_expand` は各手を chain_count 降順で枝刈りしており、
これは「今すぐ発火する盤面」を優先している = 「発火させず積む」定義と正反対。

検証方法:
    data/indicators_v2/boards/v29.npz からサンプル盤面を抽出し、
    _build_ceiling_expand(depth=1) の frontier (上位 beam_width 件) それぞれについて
    ChainSimulator.find_erasable_groups(dropped) で「消去可能な4連結以上グループが
    盤面上に残ったまま (未消去) 存在するか」を確認する。

    これが真であれば、frontier 上位候補は「1手追加した瞬間に発火した盤面」を
    そのまま (消去・重力未適用のまま) 保持しており、次の手はその発火済み盤面の
    上にさらに積む形になる = 「発火させず積む」目的とは逆。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_diag_ceiling_objective
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import Board
from src.chain import ChainSimulator
import src.indicators_v2 as iv

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")
N_SAMPLE = 40
BEAM_WIDTH = 8


def main() -> None:
    print("=== フェーズ0診断: build_ceiling_chain 目的関数の検証 ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(0)
    n = min(N_SAMPLE, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)

    sim = ChainSimulator()

    n_boards_checked = 0
    n_top1_already_erasable = 0
    n_beam_any_erasable = 0
    total_beam_erasable_count = 0
    total_beam_count = 0

    for i in idxs:
        grid = grids[i]
        board = Board.from_list(grid.tolist())
        if board.is_dead():
            continue
        frontier = iv._build_ceiling_expand([(0, board)], sim, BEAM_WIDTH)
        if not frontier:
            continue
        n_boards_checked += 1

        # top1 (chain_count が最大の候補) が「未消去の4連結以上グループ」を
        # 盤面上に残したままか (= 発火済み状態をそのまま次に持ち越しているか)。
        top1_chain, top1_board = frontier[0]
        top1_erasable = sim.find_erasable_groups(top1_board)
        if top1_chain > 0 and len(top1_erasable) > 0:
            n_top1_already_erasable += 1

        # frontier 全体 (beam_width 件) のうち何件が未消去グループを持つか。
        beam_erasable = 0
        for chain_count, cand_board in frontier:
            erasable = sim.find_erasable_groups(cand_board)
            total_beam_count += 1
            if chain_count > 0 and len(erasable) > 0:
                beam_erasable += 1
                total_beam_erasable_count += 1
        if beam_erasable > 0:
            n_beam_any_erasable += 1

        print(
            f"  board#{i}: top1_chain={top1_chain} "
            f"top1_未消去4連結={len(top1_erasable)}件 "
            f"beam内未消去あり={beam_erasable}/{len(frontier)}"
        )

    print("")
    print(f"検証盤面数: {n_boards_checked}")
    print(
        f"top1候補が「未消去の発火済みグループ」を含む割合: "
        f"{n_top1_already_erasable}/{n_boards_checked} "
        f"({100.0 * n_top1_already_erasable / max(1, n_boards_checked):.1f}%)"
    )
    print(
        f"beam内に1件以上「未消去の発火済みグループ」を含む盤面の割合: "
        f"{n_beam_any_erasable}/{n_boards_checked} "
        f"({100.0 * n_beam_any_erasable / max(1, n_boards_checked):.1f}%)"
    )
    print(
        f"beam候補全体のうち未消去発火済みの割合: "
        f"{total_beam_erasable_count}/{total_beam_count} "
        f"({100.0 * total_beam_erasable_count / max(1, total_beam_count):.1f}%)"
    )
    print("")
    print("=== 診断結論 ===")
    if n_boards_checked > 0 and n_top1_already_erasable / n_boards_checked > 0.5:
        print(
            "確証: frontier top1 (chain_count最大採用) の過半数が「未消去のまま"
            "発火済みグループを保持した盤面」。目的関数は「発火させず積む」の"
            "正反対 (=既に発火した盤面を最優先している) ことが実証された。"
        )
    else:
        print(
            "反証: 未消去発火済みグループを持つ候補は少数派。目的関数逆転仮説は"
            "この抽出条件では支持されない (要再検証)。"
        )


if __name__ == "__main__":
    main()
