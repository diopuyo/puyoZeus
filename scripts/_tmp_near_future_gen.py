"""近未来最大火力 (K=1〜5、独立指標) の生成。

user方針 (2026-07-22、コーディネータ伝達): 「観測軸を提供→学習で重要度発見」の
思想に基づき、K=1,2,3,4,5 の5水準を独立列として持つ (火力の立ち上がりカーブ
=近未来準備度プロファイルを観測軸化し、どのhorizonがどのフェーズで効くかを
学習に拾わせる)。

## 指標定義

現在盤面から、
    1) ネクスト・ダブルネクスト (既知の色、位置のみ探索、22通り配置) を
       最適に置く
    2) その後 K手 (K=1..5) は理想ツモ (4色自由・色も位置も探索、
       単色1個ずつの自由配置。既存 indicators_v2 の build_ceiling_chain /
       potential_fire_power と同じ「1手=1色ぷよ自由配置」近似を踏襲する。
       真の2個組ツモの分岐(22通り)は既知ネクスト2手にのみ使い、それ以降の
       自由K手は単色1個ずつのビームサーチで近似する — 分岐数を24 (6列×4色)
       に抑え、K=5でも数百ms/盤面に収める設計判断)
で到達する最大得点(火力)を返す。連鎖数は参考値。

飽和天井 (scripts/_tmp_ama_builder.py の ama_ceiling、無制限深さ) とは別物:
本指標は "2+K" 手 (最大 2+5=7手) で打ち切る**有限ホライズン**であり、
空き空間量に支配されない (コーディネータ既知の飽和天井撤退理由の裏返し)。

次ツモ・ダブルネクストが取得できない盤面は、その2手も理想ツモで代用し
`used_real_next` フラグで区別する。

## ⚠️ 重要な発見 (実装中に判明、正直な報告)

次ツモ情報を持つ npz (data/indicators_v2/boards_lean_next/) は
`video_c1`〜`video_c84` という全く別の動画IDセットであり、本検証で使う
labeled_win.csv の対象動画 (video_29〜38) とは **1本も重複しない**
(「7動画のみの部分カバレッジ」ではなく「対象動画に対しては完全にカバレッジ
ゼロ」という、想定より厳しい状況)。そのため本実行では全行が
「次ツモ不明→フォールバック」となり、`used_real_next` は常に False になる。
実質的に本結果は「2+K手の理想ツモ (単色自由配置) ビームサーチ」を測っている。
既知ネクスト混合を実測するには、video_29〜38 に対して別途 NextDetector を
回して next/dnext を収集する追加タスクが必要 (本タスクの範囲外、正直に
切り分けて報告する)。コード自体は既知ネクストが利用可能になった場合に
正しく動作するよう汎用実装してある (将来の再利用に備える)。

## 再利用元

scripts/_tmp_ama_builder.py (_drop_one_color, _simulate_with_score,
_compute_active_colors_by_game) をそのまま再利用。src/chain_bitboard.py の
得点計算式 (src/scoring.py 経由)・4色判定ロジックも同様に流用。
src/indicators_v2.py・src/chain.py は import しない (ゼロカップリング維持)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_near_future_gen
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._tmp_validate_build_ceiling_subset import (  # noqa: E402
    _load_npz_for_video, _match_grid, _grid_to_board, TARGET_VIDEO_IDS, BOARDS_DIR,
)
from scripts._tmp_ama_builder import (  # noqa: E402
    Board, BOARD_COLS, BOARD_ROWS,
    _drop_one_color, _simulate_with_score, _compute_active_colors_by_game,
)

LABELED_WIN_CSV = Path("data/indicators_v2/study/labeled_win.csv")
OUT_CSV = Path("data/indicators_v2/study/near_future_firepower_video_result.csv")
NEXT_NPZ_DIR = Path("data/indicators_v2/boards_lean_next")

# K の水準 (user拡張指示: 1〜5独立列)。
K_LEVELS: "tuple[int, ...]" = (1, 2, 3, 4, 5)

# 既知ネクストスロット数 (next, dnext の2手。次ツモデータ不明時は理想ツモで代用)。
KNOWN_HAND_SLOTS: int = 2

# ビーム幅 (各手で保持する上位候補数。indicators_v2 の
# BUILD_CEILING_CHAIN_BEAM_WIDTH=8 と同じ値を踏襲)。
BEAM_WIDTH: int = 8

# 有効なぷよ色 (1-5)。-1(未検出)・9(ojama誤検出)はネクスト不明として扱う
# (scripts/proto_net_threat_v2.py の VALID_PUYO_COLORS と同じ定義)。
VALID_PUYO_COLORS: frozenset = frozenset({1, 2, 3, 4, 5})
NEXT_COLOR_UNKNOWN_SENTINEL: int = -1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================
# 既知ペア配置 (22通り、次ツモ・ダブルネクスト用)
# ============================
# scripts/proto_net_threat_v2.py の _enumerate_pair_placements と同一ロジック
# (import はせず、ゼロカップリングのためここで独立再実装する)。


def _vertical_pair_placements(board: Board, color_a: int, color_b: int) -> "list[Board]":
    """縦置き (同一列2段、色順2通り) の配置を列挙する。最大 6×2=12 件。"""
    out: "list[Board]" = []
    for col in range(BOARD_COLS):
        height = board.height_of(col)
        if height + 2 > BOARD_ROWS:
            continue
        row_bottom = BOARD_ROWS - 1 - height
        row_top = row_bottom - 1
        for bottom_color, top_color in ((color_a, color_b), (color_b, color_a)):
            work = board.copy()
            work.set(row_bottom, col, bottom_color)
            work.set(row_top, col, top_color)
            out.append(work)
    return out


def _horizontal_pair_placements(board: Board, color_a: int, color_b: int) -> "list[Board]":
    """横置き (隣接2列、色順2通り) の配置を列挙する。最大 5×2=10 件。"""
    out: "list[Board]" = []
    for col in range(BOARD_COLS - 1):
        col2 = col + 1
        h1, h2 = board.height_of(col), board.height_of(col2)
        if h1 >= BOARD_ROWS or h2 >= BOARD_ROWS:
            continue
        row1 = BOARD_ROWS - 1 - h1
        row2 = BOARD_ROWS - 1 - h2
        for c1, c2 in ((color_a, color_b), (color_b, color_a)):
            work = board.copy()
            work.set(row1, col, c1)
            work.set(row2, col2, c2)
            out.append(work)
    return out


def _enumerate_pair_placements(board: Board, color_a: int, color_b: int) -> "list[Board]":
    """既知の2色ぷよペアを実物理配置 (22通り) で置いた盤面群を返す。"""
    return _vertical_pair_placements(board, color_a, color_b) + _horizontal_pair_placements(
        board, color_a, color_b,
    )


def _is_valid_next_pair(pair: "tuple[int, int] | None") -> bool:
    """next_pair/dnext_pair が実ネクストとして使える値かを判定する。"""
    if pair is None:
        return False
    return all(int(c) in VALID_PUYO_COLORS for c in pair)


# ============================
# 手展開 (既知ペア / 自由1個ずつ)
# ============================


def _known_hand_expand(
    frontier: "list[tuple[float, Board]]", color_a: int, color_b: int, beam_width: int,
) -> "tuple[list[tuple[float, Board, int]], int]":
    """既知ペア (22通り) で1手展開し、得点降順で上位 beam_width 件を返す。"""
    candidates: "list[tuple[float, Board, int]]" = []
    for _, base_board in frontier:
        for placed in _enumerate_pair_placements(base_board, color_a, color_b):
            if placed.is_dead():
                continue
            sim = _simulate_with_score(placed)
            candidates.append((float(sim.total_score), sim.final_board, sim.chain_count))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:beam_width], len(candidates)


def _free_hand_expand(
    frontier: "list[tuple[float, Board]]", active_colors: "tuple[int, ...]", beam_width: int,
) -> "tuple[list[tuple[float, Board, int]], int]":
    """自由1個ずつ (6列×4色=24通り) で1手展開し、得点降順で上位 beam_width 件を返す。"""
    candidates: "list[tuple[float, Board, int]]" = []
    for _, base_board in frontier:
        for col in range(BOARD_COLS):
            for color in active_colors:
                dropped = _drop_one_color(base_board, col, color)
                if dropped is None or dropped.is_dead():
                    continue
                sim = _simulate_with_score(dropped)
                candidates.append((float(sim.total_score), sim.final_board, sim.chain_count))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:beam_width], len(candidates)


# ============================
# 近未来火力プロファイル (K=1..5 チェックポイント方式、1回のビームで同時取得)
# ============================


def near_future_profile(
    board: Board,
    active_colors: "tuple[int, ...]",
    next_pair: "tuple[int, int] | None",
    dnext_pair: "tuple[int, int] | None",
    k_levels: "tuple[int, ...]" = K_LEVELS,
    beam_width: int = BEAM_WIDTH,
) -> dict:
    """K=1..max(k_levels) の近未来最大火力を1回のビームサーチで同時に求める。

    各手の後に「その時点までの経路上で観測された最大到達得点」の running max
    を記録する (current_max_chain の K手拡張、K=0相当=現在盤面0手がベース)。
    既知ネクスト(next_pair)・ダブルネクスト(dnext_pair)が有効なら先頭2手を
    それで固定し、無効なら理想ツモで代用する (used_real_next で区別)。

    Returns:
        dict: {"k{K}_score", "k{K}_chain", "k{K}_cost_sec" (K in k_levels),
               "used_real_next", "n_sim_calls"}
    """
    max_k = max(k_levels)
    result: dict = {}
    if board.is_dead():
        for k in k_levels:
            result[f"k{k}_score"] = 0.0
            result[f"k{k}_chain"] = 0
            result[f"k{k}_cost_sec"] = 0.0
        result["used_real_next"] = False
        result["n_sim_calls"] = 0
        return result

    t_start = time.perf_counter()
    frontier: "list[tuple[float, Board]]" = [(0.0, board)]
    best_score = 0.0
    best_chain = 0
    n_sim = 0
    used_real_next = False
    total_hands = KNOWN_HAND_SLOTS + max_k

    for hand_idx in range(total_hands):
        if hand_idx == 0 and _is_valid_next_pair(next_pair):
            expanded, n_this = _known_hand_expand(frontier, next_pair[0], next_pair[1], beam_width)
            used_real_next = True
        elif hand_idx == 1 and _is_valid_next_pair(dnext_pair):
            expanded, n_this = _known_hand_expand(frontier, dnext_pair[0], dnext_pair[1], beam_width)
            used_real_next = True
        else:
            expanded, n_this = _free_hand_expand(frontier, active_colors, beam_width)
        n_sim += n_this
        if not expanded:
            break
        frontier = [(s, b) for s, b, _c in expanded]
        top_score, _top_board, top_chain = expanded[0]
        if top_score > best_score:
            best_score = top_score
            best_chain = top_chain

        k_here = hand_idx + 1 - KNOWN_HAND_SLOTS
        if k_here in k_levels:
            result[f"k{k_here}_score"] = best_score
            result[f"k{k_here}_chain"] = best_chain
            result[f"k{k_here}_cost_sec"] = time.perf_counter() - t_start

    # ビームが途中で尽きた (候補ゼロ) 場合、以降のKは直前の値を引き継ぐ
    # (得点は単調非減少のはずであり、直前値がその時点での最良の下界になる)。
    for k in sorted(k_levels):
        if f"k{k}_score" not in result:
            prev_ks = [kk for kk in k_levels if kk < k and f"k{kk}_score" in result]
            if prev_ks:
                prev_k = max(prev_ks)
                result[f"k{k}_score"] = result[f"k{prev_k}_score"]
                result[f"k{k}_chain"] = result[f"k{prev_k}_chain"]
                result[f"k{k}_cost_sec"] = result[f"k{prev_k}_cost_sec"]
            else:
                result[f"k{k}_score"] = 0.0
                result[f"k{k}_chain"] = 0
                result[f"k{k}_cost_sec"] = time.perf_counter() - t_start

    result["used_real_next"] = used_real_next
    result["n_sim_calls"] = n_sim
    return result


# ============================
# 次ツモ npz カバレッジ確認 (boards_lean_next、video_id別)
# ============================


def _next_available_video_ids() -> "set[str]":
    """boards_lean_next 配下に存在する video_id 集合を返す (対象動画との重複確認用)。"""
    vids: "set[str]" = set()
    if not NEXT_NPZ_DIR.exists():
        return vids
    for f in sorted(NEXT_NPZ_DIR.glob("*.npz")):
        data = np.load(str(f), allow_pickle=True)
        for v in np.unique(data["video_id"]):
            vids.add(str(v))
    return vids


# ============================
# メイン
# ============================


def main() -> int:
    logger.info("=== 近未来最大火力 (K=1..5) 生成開始 ===")

    next_video_ids = _next_available_video_ids()
    overlap = next_video_ids & set(TARGET_VIDEO_IDS)
    logger.info(
        "boards_lean_next 動画数=%d、対象動画(%d本)との重複=%d本 %s",
        len(next_video_ids), len(TARGET_VIDEO_IDS), len(overlap),
        "(重複ゼロ→全行フォールバック確定)" if not overlap else f"重複={sorted(overlap)}",
    )

    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("対象行数 (won付き): %d", len(df))

    npz_cache: "dict[str, dict[str, np.ndarray]]" = {}
    active_colors_cache: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for vid in TARGET_VIDEO_IDS:
        npz_cache[vid] = _load_npz_for_video(vid)
        stem = vid.replace("video_", "v")
        active_colors_cache.update(_compute_active_colors_by_game(BOARDS_DIR / f"{stem}.npz"))

    rows_out: "list[dict]" = []
    n_matched = 0
    n_missed = 0
    n_missed_colors = 0
    n_used_real_next = 0
    total = len(df)
    t_start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        vid = str(row["video_id"])
        side = str(row["side"])
        game_idx = int(row["game_idx"])
        t_sec = float(row["t_sec"])
        grid = _match_grid(npz_cache[vid], side, game_idx, t_sec)
        if grid is None:
            n_missed += 1
            continue
        colors = active_colors_cache.get((vid, game_idx))
        if colors is None:
            n_missed_colors += 1
            continue
        board = _grid_to_board(grid)

        # 次ツモは対象動画と重複ゼロと確認済みのため常に None
        # (将来 next/dnext データが video_29-38 分で用意された場合に備え、
        # ルックアップの余地は _is_valid_next_pair 経由で残しておく)。
        next_pair, dnext_pair = None, None

        profile = near_future_profile(board, colors, next_pair, dnext_pair)
        if profile["used_real_next"]:
            n_used_real_next += 1

        row_dict = row.to_dict()
        for k in K_LEVELS:
            row_dict[f"near_future_firepower_k{k}_raw"] = profile[f"k{k}_score"]
            row_dict[f"near_future_firepower_k{k}_chain_ref"] = profile[f"k{k}_chain"]
            row_dict[f"near_future_firepower_k{k}_cost_sec"] = profile[f"k{k}_cost_sec"]
        row_dict["near_future_used_real_next"] = profile["used_real_next"]
        row_dict["near_future_n_sim_calls"] = profile["n_sim_calls"]
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            avg_cost_k5 = sum(r["near_future_firepower_k5_cost_sec"] for r in rows_out) / max(1, n_matched)
            logger.info(
                "進捗: %d/%d matched=%d missed=%d missed_colors=%d real_next使用=%d "
                "K5平均コスト=%.1fms elapsed=%.0fs eta=%.0fs",
                i + 1, total, n_matched, n_missed, n_missed_colors, n_used_real_next,
                1000.0 * avg_cost_k5, elapsed, eta,
            )

    logger.info(
        "完了: matched=%d missed=%d missed_colors=%d (total=%d) real_next使用=%d/%d",
        n_matched, n_missed, n_missed_colors, total, n_used_real_next, n_matched,
    )

    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)

    for k in K_LEVELS:
        raw = out_df[f"near_future_firepower_k{k}_raw"]
        cost_ms = out_df[f"near_future_firepower_k{k}_cost_sec"] * 1000.0
        logger.info(
            "K=%d: score mean=%.0f median=%.0f max=%.0f | cost mean=%.1fms p95=%.1fms max=%.1fms",
            k, raw.mean(), raw.median(), raw.max(),
            cost_ms.mean(), cost_ms.quantile(0.95), cost_ms.max(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
