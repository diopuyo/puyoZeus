"""候補G3 連鎖関節点(articulation points)の予備計算(小サンプル)。

定義: 連鎖の各消去ステップの「消去グループ」のうち、それを消さなかったら
以降の連鎖が短くなる(=途切れる)グループ数 = 「そこを潰されると連鎖が
途切れる急所」の数。

本スクリプトは正式実装ではなく、関節点数の分布に意味があるか(=全0でも
全グループでもない分散を持つか)だけを見る予備計算。
手法: 各ステップの各グループについて「そのグループだけ消さず他は消す」
反実仮想盤面を作り、続きを再simulateして残り連鎖数が短くなるか判定する
(追加simは 1イベントあたり総ステップ内グループ数回、小サンプルなら軽量)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_g3_articulation_pilot
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import Board, COLOR_OJAMA  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from scripts.label_exchange_outcome import _load_npz, _detect_fire_events, NPZ_DIR  # noqa: E402

SAMPLE_VIDEOS = ["c1.npz", "c2.npz", "c3.npz"]
MAX_EVENTS: int = 40


def _board_from_grid(grid: np.ndarray) -> Board:
    return Board.from_list(grid.tolist())


def count_articulation_groups(board: Board, sim: ChainSimulator) -> tuple[int, int, int]:
    """1盤面の発火結果に対し関節点候補グループ数を数える。

    Returns:
        (total_groups, critical_groups, chain_count)
    """
    result = sim.simulate(board)
    steps = result.steps
    chain_count = len(steps)
    if chain_count < 2:
        return (sum(len(s.erased_groups) for s in steps), 0, chain_count)

    total_groups = 0
    critical_groups = 0

    for i, step in enumerate(steps):
        groups = step.erased_groups
        remaining_orig = chain_count - i  # このステップ含め残り何ステップか
        total_groups += len(groups)

        for g_idx in range(len(groups)):
            held = groups[g_idx]
            other_groups = [g for j, g in enumerate(groups) if j != g_idx]
            wb = step.board_before.copy()
            # held グループを不活性なおじゃま色に凍結(=消えなかった事にする)。
            # これをしないと simulate() が同色4連結を再検出し実質ノーオペになる。
            for (r, c) in held.cells:
                wb.set(r, c, COLOR_OJAMA)
            if other_groups:
                sim._erase_groups(wb, other_groups)  # type: ignore[attr-defined]
                sim.apply_gravity(wb)
                alt_result = sim.simulate(wb)
                remaining_alt = 1 + alt_result.chain_count
            else:
                # このステップの唯一のグループ -> 凍結後は何も消えず重力もかからない
                sim.apply_gravity(wb)
                alt_result = sim.simulate(wb)
                remaining_alt = alt_result.chain_count

            if remaining_alt < remaining_orig:
                critical_groups += 1

    return (total_groups, critical_groups, chain_count)


def main() -> None:
    print("=== _tmp_g3_articulation_pilot 開始 ===")
    sim = ChainSimulator()
    rows: list[tuple[int, int, int]] = []
    n_events = 0

    for vname in SAMPLE_VIDEOS:
        path = NPZ_DIR / vname
        if not path.exists():
            print(f"  [WARN] {path} 見つからずスキップ")
            continue
        records = _load_npz(path)
        for rec in records:
            fire_idx = _detect_fire_events(rec.t_sec, rec.score)
            for fi in fire_idx:
                if n_events >= MAX_EVENTS:
                    break
                board_idx = max(0, fi - 1)
                board = _board_from_grid(rec.grids[board_idx])
                total, critical, chain_count = count_articulation_groups(board, sim)
                if chain_count < 2:
                    continue  # 連鎖1以下は関節点の概念が無意味
                rows.append((total, critical, chain_count))
                n_events += 1
            if n_events >= MAX_EVENTS:
                break
        print(f"  {vname}: 累計イベント {n_events}")
        if n_events >= MAX_EVENTS:
            break

    if not rows:
        print("[WARN] 連鎖2以上のイベントが取得できませんでした")
        return

    totals = np.array([r[0] for r in rows])
    criticals = np.array([r[1] for r in rows])
    chains = np.array([r[2] for r in rows])
    ratio = criticals / np.clip(totals, 1, None)

    print(f"\nサンプル数(連鎖>=2): {len(rows)}")
    print(f"連鎖数: mean={chains.mean():.2f} min={chains.min()} max={chains.max()}")
    print(f"総グループ数: mean={totals.mean():.2f}")
    print(f"関節点候補グループ数: mean={criticals.mean():.2f}"
          f" min={criticals.min()} max={criticals.max()}")
    print(f"関節点比率(critical/total): mean={ratio.mean():.3f}"
          f" std={ratio.std():.3f}")
    print("\n分布 (critical_groups 件数):")
    for v in sorted(set(criticals.tolist())):
        cnt = int((criticals == v).sum())
        print(f"  critical={int(v)}: {cnt} 件")

    corr_chain = float(np.corrcoef(chains, criticals)[0, 1])
    corr_total = float(np.corrcoef(totals, criticals)[0, 1])
    print(f"\nPearson相関 (chain_count vs critical_groups): r={corr_chain:.3f}")
    print(f"Pearson相関 (total_groups vs critical_groups): r={corr_total:.3f}")

    print("\n全0/全同一なら情報量なし、分散があれば継続検討価値あり。")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
