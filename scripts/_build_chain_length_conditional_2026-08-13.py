"""#3 修正用: E[最終連鎖数 | 観測N連鎖到達] の条件付き分布テーブルを構築する。

## 背景 (docs/DEMO_REVIEW_2026-08-13.md #3)

打ち合い応手の時間予算を「観測済み連鎖数 × 0.4秒」で計算していたが、連鎖の
発火検知 (chain_trigger_sec/chain_mechanism) は連鎖進行中の途中経過を捉える
ことがあり、そのフレーム時点の observed chain_count は最終連鎖数より小さい
ことがある (連鎖アニメが実時間で進行するため)。この過小評価を補正するため、
「N連鎖まで到達したことが観測された連鎖は、最終的に何連鎖で終わることが
多いか」の条件付き期待値テーブルを実測データから作る。

## 手法 (npz に chain_count が直接保存されていないための再構成)

`data/indicators_v2/boards_lean_phase_l_2026-08-11/*.npz` は
(video_id, side, game_idx, t_sec, grids, chain_trigger_sec, chain_mechanism) の
1 snapshot = 1 行の記録であり、連鎖数そのものは保存されていない。しかし
`ChainSimulator.simulate(before_board)` で当時の連鎖を再現すれば、その
chain_count を「最終連鎖数」の実測値として使える (`src/scoring.py` の
score_consistency_ratio が採用しているのと同じ「simulateで実イベントを
検証する」手法の応用)。

⚠️ **実測で判明した重要な補正 (2026-08-13)**: chain_trigger_sec/
chain_mechanism タグは「発火が確定した直後の snapshot」に付くとは限らず、
実データでは多くの場合そのタグが付いた行の**直前の行は既に発火後の盤面**
(スコアも連鎖後の値) であり、素朴に「直前行=発火前盤面」として simulate
すると誤検出 (chain_count=0) になる (video_id=29 で 252 件中 249 件が
誤検出、実測で確認済み)。正しい発火前盤面は「非空セル数が
`ERASURE_MIN_DROP` (=4、`src.chain_detector.VideoChainTracker` と同一の
検出則) 以上減少した直前行」であり、タグ行から時系列を遡ってこの条件を
満たす最初の行を探し、その1つ前を発火前盤面とする
(`_find_before_board_index` 参照。15 npz でのサンプル調査でオフセットは
0〜3行が94%、まれに数十行に及ぶ実測分布のため固定ルックバック幅は
設けず、系列の先頭まで遡って探索する)。

## 条件付き期待値の定義

ある連鎖の最終連鎖数が K だった場合、その連鎖は現実の時間進行の中で
N=1,2,...,K の各段階を必ず通過する (連鎖中はツモ設置不可のため、観測が
「N連鎖まで進んだ」という事象は K>=N の連鎖全てで起こりうる)。したがって

    E[最終 | N到達] = Σ_{K>=N} K * count(K) / Σ_{K>=N} count(K)

を全実測連鎖のヒストグラムから計算する (集合全体を1回集計するだけで済む、
連鎖ごとの経路シミュレーションは不要)。

## 使い方
    python -m scripts._build_chain_length_conditional_2026-08-13 \\
        --npz-dir data/indicators_v2/boards_lean_phase_l_2026-08-11 \\
        --out data/verify/chain_length_conditional_2026-08-13.json
"""
from __future__ import annotations

import argparse
import datetime
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.board import Board
from src.chain import ChainSimulator
from src.chain_detector import ERASURE_MIN_DROP
from src.production_config import GHOST_CHAIN_RULE_ENABLED

DEFAULT_NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
DEFAULT_OUT_PATH = Path("data/verify/chain_length_conditional_2026-08-13.json")

# chain_mechanism が「連鎖タグなし」を表す値
# (scripts/build_labeled_win_from_npz.py の _NO_CHAIN_TAG_VALUES と同一定義、
# 意図的に複製: 本スクリプトは npz→JSON の使い捨て集計ツールであり、
# 重い依存 [pandas 等] を持つ build_labeled_win_from_npz を import すると
# 起動コストが不要に増えるため)。
_NO_CHAIN_TAG_VALUES: frozenset[str] = frozenset({"", "nan", "none"})


def _find_before_board_index(nz_counts: "list[int]", idxs: "list[int]", pos: int) -> int | None:
    """タグ行 (idxs[pos]) から遡り、発火前盤面 (before_board) の行 index を探す。

    モジュール docstring の「実測で判明した重要な補正」参照。非空セル数が
    ERASURE_MIN_DROP 以上減少している直近の遷移 (idxs[j-1]->idxs[j]) を
    系列の先頭方向へ遡って探し、見つかったら idxs[j-1] (減少が起きる
    *前*の行 = 発火前盤面) を返す。見つからなければ None。
    """
    for j in range(pos, 0, -1):
        if nz_counts[idxs[j - 1]] - nz_counts[idxs[j]] >= ERASURE_MIN_DROP:
            return idxs[j - 1]
    return None


def _chain_counts_in_file(npz_path: Path) -> list[int]:
    """1 npz ファイル内の全連鎖イベントについて、実測最終連鎖数のリストを返す。

    (side, game_idx) ごとに t_sec 昇順に並べ、chain_trigger_sec が有効な行
    それぞれについて `_find_before_board_index` で発火前盤面を特定し、
    ChainSimulator で再現する。chain_count==0 (誤検出、
    VideoChainTracker.update と同じ判定則) または発火前盤面が特定できない
    場合はスキップする。
    """
    d = np.load(npz_path, allow_pickle=True)
    if "chain_trigger_sec" not in d.files or "chain_mechanism" not in d.files:
        return []
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]
    nz_counts = (grids != 0).sum(axis=(1, 2)).tolist()

    groups: dict[tuple, list[int]] = {}
    for i in range(len(grids)):
        key = (str(side[i]), int(game_idx[i]))
        groups.setdefault(key, []).append(i)

    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    chain_counts: list[int] = []
    for idxs in groups.values():
        idxs.sort(key=lambda i: float(t_sec[i]))
        prev_trigger_sec: float | None = None  # 直近タグの trigger_sec (重複排除用)
        for pos in range(len(idxs)):
            i = idxs[pos]
            if not np.isfinite(trigger[i]):
                prev_trigger_sec = None
                continue
            tag = str(mechanism[i]).strip().lower()
            if tag in _NO_CHAIN_TAG_VALUES:
                prev_trigger_sec = None
                continue
            # 同一 trigger_sec が連続する行は「同じ物理連鎖」への重複タグ
            # (hold窓の間、後続の dedup snapshot にも同じ ChainEvent が
            # 引き継がれる実測挙動、2026-08-13 発見)。1回だけ数える。
            if prev_trigger_sec is not None and float(trigger[i]) == prev_trigger_sec:
                continue
            prev_trigger_sec = float(trigger[i])
            before_i = _find_before_board_index(nz_counts, idxs, pos)
            if before_i is None:
                continue
            before_board = Board.from_list(grids[before_i].tolist())
            if before_board.is_dead():
                continue
            result = sim.simulate(before_board)
            if result.chain_count < 1:
                continue  # 誤検出 (VideoChainTracker.update と同じ判定則)
            chain_counts.append(int(result.chain_count))
    return chain_counts


def build_conditional_table(chain_counts: list[int]) -> dict:
    """実測最終連鎖数のリストから E[最終|N到達] テーブルを構築する。

    Returns:
        dict: JSON 保存用 (histogram/expected_final_given_reached_n/
        total_events/max_chain_count_observed/generated_at/methodology)。
    """
    hist = Counter(chain_counts)
    max_k = max(hist) if hist else 0
    table: dict[str, float] = {}
    for n in range(1, max_k + 1):
        reachable = [k for k in hist if k >= n]
        total = sum(hist[k] for k in reachable)
        if total == 0:
            continue
        expected = sum(k * hist[k] for k in reachable) / total
        table[str(n)] = expected
    return {
        "histogram": {str(k): int(c) for k, c in sorted(hist.items())},
        "expected_final_given_reached_n": table,
        "total_events": int(sum(hist.values())),
        "max_chain_count_observed": int(max_k),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "methodology": (
            "E[final|N] = sum(K*count(K) for K>=N) / sum(count(K) for K>=N)。"
            "chain_count は npz 未保存のため、chain_trigger_sec 行の直前"
            "snapshot を発火前盤面として ChainSimulator で再構成した実測値。"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    files = sorted(args.npz_dir.glob("*.npz"))
    print(f"[build_chain_length_conditional] npz {len(files)} 件を走査")
    all_counts: list[int] = []
    for i, f in enumerate(files):
        counts = _chain_counts_in_file(f)
        all_counts.extend(counts)
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  ... {i + 1}/{len(files)} files, 累積イベント数={len(all_counts)}")

    table = build_conditional_table(all_counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_chain_length_conditional] {args.out} に保存 "
          f"(total_events={table['total_events']}, max_chain={table['max_chain_count_observed']})")


if __name__ == "__main__":
    main()
