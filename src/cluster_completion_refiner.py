"""連結 3 cell + 隣接 1 cell が異色 → 4 cluster 完成補正 (試行 I)。

ぷよぷよでは 4+ 同色クラスタが消去対象。CNN/HSV が 1 cell だけ
誤認した場合、3 cell 同色が連結し隣接 1 cell が「同色だったら 4 cluster」
という状況になる。

戦略:
    - 各 cell について 4 近傍に「自分と異色だが、隣接の 3 cell が同色」
      な配置を探索
    - 3 cell 同色のクラスタに隣接する 1 cell が異色 → 同色に補正候補
    - ただし誤補正リスク高なので、補正は次の条件全て満たした時のみ:
        a. 隣接の同色クラスタサイズ ≥ 3
        b. 自身の color が puyo (EM/UNKNOWN/OJM 以外)
        c. 連鎖中ではない (is_chain=False)

連鎖直後の 4 cluster は消滅するので、影響は限定的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
    HIDDEN_ROWS, Board,
)
from src.chain import ChainSimulator


# 隣接同色クラスタのサイズ閾値 (3 以上で補正候補)
CLUSTER_SIZE_THRESHOLD: int = 3


@dataclass
class ClusterCompletionRefiner:
    """3 cell 同色クラスタ + 隣接 1 cell 異色 → 同色に補正。"""
    cluster_size_threshold: int = CLUSTER_SIZE_THRESHOLD

    def refine(
        self,
        board: Board,
        is_chain: bool = False,
    ) -> tuple[Board, np.ndarray]:
        """3 連結 + 隣接 1 cell 異色 → 同色に補正。

        Returns:
            (refined_board, completion_mask shape=(BOARD_ROWS, BOARD_COLS) bool)
        """
        out = board.copy()
        completion_mask = np.zeros(
            (BOARD_ROWS, BOARD_COLS), dtype=bool,
        )
        if is_chain:
            return out, completion_mask
        # 全ての puyo グループを取得
        sim = ChainSimulator()
        try:
            groups = sim.find_groups(board)
        except Exception:
            return out, completion_mask
        # サイズ ≥ 3 (但し ≤ 3 で完成候補) のグループを抽出
        # 4+ で既に消去対象なので、3 cell ぴったり に近いものが対象
        for group in groups:
            if group.size != 3:
                continue
            target_color = group.color
            # この group の 4 近傍 cell で異色 puyo を探す
            for cr, cc in group.cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                        continue
                    if (nr, nc) in group.cells:
                        continue  # 同 group 内は skip
                    nc_color = int(out.get(nr, nc))
                    if nc_color in (
                        COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
                    ):
                        continue  # 空・OJM は対象外
                    if nc_color == target_color:
                        continue  # 既に同色
                    # 異色 puyo cell → 補正
                    if not completion_mask[nr, nc]:
                        out.set(nr, nc, target_color)
                        completion_mask[nr, nc] = True
        return out, completion_mask


__all__ = [
    "ClusterCompletionRefiner",
    "CLUSTER_SIZE_THRESHOLD",
]
