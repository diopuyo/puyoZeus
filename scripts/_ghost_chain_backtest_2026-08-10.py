"""幽霊連鎖ルール ON配線 全域バックテスト (2026-08-10)。

`data/indicators_v2/boards_lean_phase_l_2026-08-07/*.npz` (148動画・実測
約110万盤面フレーム) の全盤面で、13段目 (隠し段, row=0) を消去判定から
除外する幽霊連鎖ルール ON/OFF の chain_bitboard.simulate_batch を実行し、
差分を集計する (過学習ゲート、user恒久指示 feedback_overfitting_awareness)。

高速化のため、フレーム全数は chain_bitboard (numpy バッチ) で判定する
(src.chain.ChainSimulator の BFS 版と等価であることは
tests/test_chain.py::TestGhostChainRule::test_matches_chain_bitboard_reference_implementation
で担保済み)。変化が見つかった盤面のうち代表数件のみ、詳細な連鎖ステップ・
得点比較のため src.chain.ChainSimulator (BFS 版、calculate_chain_score 併用)
で再計算する。

出力:
    - 標準出力: 集計サマリ
    - data/verify/ghost_chain_backtest_2026-08-10/summary.txt: 集計サマリ
    - data/verify/ghost_chain_backtest_2026-08-10/changed_boards.tsv: 変化した
      盤面の一覧 (video_id, side, t_sec, game_idx, chain_count_off/on 等)
    - data/verify/ghost_chain_backtest_2026-08-10/sample_*.txt: 代表盤面の
      ASCII図 + ON/OFF 連鎖数・得点差 (userレビュー用)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, Board
from src.chain import ChainSimulator
from src.chain_bitboard import _pack_grids_to_planes, simulate_batch
from src.scoring import calculate_chain_score

# ============================
# 定数
# ============================

BOARDS_DIR: Path = Path("data/indicators_v2/boards_lean_phase_l_2026-08-07")
OUT_DIR: Path = Path("data/verify/ghost_chain_backtest_2026-08-10")

# 1 npz 内でもさらにチャンク分割する上限 (メモリ安全弁)。
CHUNK_SIZE: int = 20_000

# 色ぷよ (お邪魔・空・UNKNOWN を除く)
COLOR_PUYO_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5)

# 代表盤面として保存する件数上限
N_SAMPLES: int = 3


def _row0_stats(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """row0(隠し段) の非空/色ぷよ有無を frame ごとに返す。"""
    row0 = grids[:, 0, :]
    any_nonempty = np.any(row0 != COLOR_EMPTY, axis=1)
    color_puyo = np.any(np.isin(row0, COLOR_PUYO_VALUES), axis=1)
    return any_nonempty, color_puyo


def _simulate_both(grids: np.ndarray) -> tuple[list, list]:
    """chain_bitboard.simulate_batch を ON/OFF 両方で実行する。"""
    planes = _pack_grids_to_planes(grids)
    off = simulate_batch(planes, exclude_hidden_row_from_pop=False)
    on = simulate_batch(planes, exclude_hidden_row_from_pop=True)
    return off, on


def _ascii_board(grid: np.ndarray) -> str:
    """盤面グリッドを ASCII 図にする (row0 に [] を付けて隠し段だとわかるようにする)。"""
    symbols = {
        COLOR_EMPTY: ".", 1: "R", 2: "B", 3: "G", 4: "Y", 5: "P", COLOR_OJAMA: "@", 10: "?",
    }
    lines: list[str] = []
    for row in range(BOARD_ROWS):
        cells = "".join(symbols.get(int(v), "?") for v in grid[row])
        tag = " <- row0(隠し段/幽霊連鎖対象)" if row == 0 else ""
        lines.append(f"row{row:2d}: {cells}{tag}")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(BOARDS_DIR / "*.npz")))
    print(f"[ghost_chain_backtest] {len(files)} files found in {BOARDS_DIR}")

    total_frames = 0
    total_row0_nonempty = 0
    total_row0_color = 0
    total_changed = 0
    # 変化の内訳: chain_count が減った/増えた/同数だが得点(erased/ojama)が違う
    n_chain_decreased = 0
    n_chain_increased = 0
    n_chain_same_but_erased_diff = 0

    changed_rows: list[str] = []
    sample_candidates: list[dict] = []

    for fi, fpath in enumerate(files):
        video_id = Path(fpath).stem
        d = np.load(fpath, allow_pickle=True)
        grids = d["grids"]
        if grids.shape[0] == 0:
            continue
        sides = d["side"]
        t_secs = d["t_sec"]
        game_idxs = d["game_idx"]
        frame_idxs = d["frame_idx"]

        n = grids.shape[0]
        total_frames += n
        any_nonempty, color_puyo = _row0_stats(grids)
        total_row0_nonempty += int(any_nonempty.sum())
        total_row0_color += int(color_puyo.sum())

        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            sub_grids = grids[start:end]
            off_results, on_results = _simulate_both(sub_grids)
            for i, (r_off, r_on) in enumerate(zip(off_results, on_results)):
                idx = start + i
                if not color_puyo[idx]:
                    # row0 に色ぷよが無ければ理論上絶対に差が出ない (安全チェック用に continue)
                    continue
                changed = (
                    r_off.chain_count != r_on.chain_count
                    or r_off.total_erased != r_on.total_erased
                    or r_off.total_ojama != r_on.total_ojama
                )
                if not changed:
                    continue
                total_changed += 1
                if r_on.chain_count < r_off.chain_count:
                    n_chain_decreased += 1
                elif r_on.chain_count > r_off.chain_count:
                    n_chain_increased += 1
                else:
                    n_chain_same_but_erased_diff += 1

                changed_rows.append(
                    f"{video_id}\t{sides[idx]}\t{t_secs[idx]:.2f}\t{game_idxs[idx]}\t"
                    f"{frame_idxs[idx]}\t{r_off.chain_count}\t{r_on.chain_count}\t"
                    f"{r_off.total_erased}\t{r_on.total_erased}\t"
                    f"{r_off.total_ojama}\t{r_on.total_ojama}"
                )
                sample_candidates.append({
                    "video_id": video_id, "side": str(sides[idx]),
                    "t_sec": float(t_secs[idx]), "game_idx": int(game_idxs[idx]),
                    "grid": sub_grids[i].copy(),
                    "chain_off": r_off.chain_count, "chain_on": r_on.chain_count,
                })
        if (fi + 1) % 20 == 0:
            print(f"  ... {fi + 1}/{len(files)} files processed "
                  f"(累計 changed={total_changed})")

    # ============================
    # サマリ出力
    # ============================
    pct_row0_nonempty = 100.0 * total_row0_nonempty / max(total_frames, 1)
    pct_row0_color = 100.0 * total_row0_color / max(total_frames, 1)
    pct_changed_of_color_row0 = (
        100.0 * total_changed / max(total_row0_color, 1)
    )
    pct_changed_of_all = 100.0 * total_changed / max(total_frames, 1)

    summary_lines = [
        "幽霊連鎖ルール ON配線 全域バックテスト結果 (2026-08-10)",
        f"対象: {BOARDS_DIR} ({len(files)} 動画)",
        "",
        f"総盤面フレーム数: {total_frames}",
        f"row0(隠し段)に何かある盤面: {total_row0_nonempty} "
        f"({pct_row0_nonempty:.3f}%)",
        f"row0(隠し段)に色ぷよがある盤面 (影響の理論上限): {total_row0_color} "
        f"({pct_row0_color:.3f}%)",
        "",
        f"ON/OFF で simulate 結果が変化した盤面: {total_changed} "
        f"(全フレーム比 {pct_changed_of_all:.4f}%, "
        f"row0色ぷよあり盤面比 {pct_changed_of_color_row0:.2f}%)",
        f"  - 連鎖数が減った (ON<OFF、想定される主典型): {n_chain_decreased}",
        f"  - 連鎖数が増えた (ON>OFF、要個別確認): {n_chain_increased}",
        f"  - 連鎖数は同じだが消去数/おじゃま数が違う: {n_chain_same_but_erased_diff}",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    (OUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")

    header = (
        "video_id\tside\tt_sec\tgame_idx\tframe_idx\t"
        "chain_count_off\tchain_count_on\ttotal_erased_off\ttotal_erased_on\t"
        "total_ojama_off\ttotal_ojama_on"
    )
    (OUT_DIR / "changed_boards.tsv").write_text(
        header + "\n" + "\n".join(changed_rows), encoding="utf-8",
    )

    # ============================
    # 代表サンプル (viz レポート)
    # ============================
    sim_off_full = ChainSimulator(exclude_hidden_row_from_pop=False)
    sim_on_full = ChainSimulator(exclude_hidden_row_from_pop=True)

    # 差が大きい (chain_count 差が大きい) 順に上位を選ぶ
    sample_candidates.sort(
        key=lambda c: abs(c["chain_off"] - c["chain_on"]), reverse=True,
    )
    n_samples = min(N_SAMPLES, len(sample_candidates))
    for rank in range(n_samples):
        cand = sample_candidates[rank]
        board = Board.from_list(cand["grid"].tolist())
        result_off = sim_off_full.simulate(board)
        result_on = sim_on_full.simulate(board)
        score_off = calculate_chain_score(result_off).total_score
        score_on = calculate_chain_score(result_on).total_score

        report_lines = [
            f"サンプル {rank + 1}: video_id={cand['video_id']} side={cand['side']} "
            f"t_sec={cand['t_sec']:.2f} game_idx={cand['game_idx']}",
            "",
            "== 入力盤面 (発火前) ==",
            _ascii_board(cand["grid"]),
            "",
            f"== OFF (従来挙動、幽霊連鎖ルール無効) ==",
            f"連鎖数: {result_off.chain_count}  消去数: {result_off.total_erased}  "
            f"おじゃま消去: {result_off.total_ojama}  得点(概算): {score_off}",
            "",
            f"== ON (幽霊連鎖ルール有効、2026-08-10 本番採用) ==",
            f"連鎖数: {result_on.chain_count}  消去数: {result_on.total_erased}  "
            f"おじゃま消去: {result_on.total_ojama}  得点(概算): {score_on}",
            "",
            f"差分: 連鎖数 {result_off.chain_count} -> {result_on.chain_count} "
            f"({result_on.chain_count - result_off.chain_count:+d})、"
            f"得点 {score_off} -> {score_on} ({score_on - score_off:+d})",
        ]
        out_path = OUT_DIR / f"sample_{rank + 1}.txt"
        out_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"[saved] {out_path}")

    print(f"\n完了。出力先: {OUT_DIR}")


if __name__ == "__main__":
    main()
