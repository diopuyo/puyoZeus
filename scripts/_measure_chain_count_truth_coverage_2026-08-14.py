"""タスク#7: 連鎖数正解v2 のカバレッジ測定 (二重照合の解決率、旧0%→新X%)。

## 背景

`src/chain_count_truth.py` (C-1a) は「れんさテロップ×得点逆算」の二重照合で
連鎖数の真値を決めるが、テロップテンプレが単一動画 (video_c54) 採取のため
他動画で全滅していた (docs/KNOWN_WEAKNESSES.md W3、解決率0%)。

本スクリプトは、テロップに依存しない**得点逆算の高信頼帯**
(`src.chain_count_truth.select_chain_count_high_confidence_band`、タスク#7
新設) 単独でどこまで「高信頼」な連鎖数を決定できるかを、実データ (16動画の
npz、`data/indicators_v2/boards_lean_phase_l_2026-08-11/`) で測定する。

## 手法 (npz からの delta_score 再構成)

npz には連鎖数も delta_score も直接保存されていないため、
`scripts/_build_chain_length_conditional_2026-08-13.py` と同一の
「発火前盤面インデックス探索」ロジック (`_find_before_board_index`) を使い、
発火タグ行 (chain_trigger_sec が有効な行) の直前の「非空セル数が
ERASURE_MIN_DROP 以上減少する遷移」の1つ前を発火前盤面とみなす。
delta_score = score[タグ行] - score[発火前盤面行] として、npz実測の score
系列 (score OCR) から直接計算する (video I/O 不要、高速)。

ChainSimulator による chain_count 再構成 (W1 の問題を抱える既存手法) も
参考値として併記するが、本スクリプトの主目的である「得点逆算の高信頼帯の
解決率」の判定には使わない (依存を持たせないための独立性維持、
`chain_count_truth.py` docstring の設計方針と同じ)。

## 使い方
    python -m scripts._measure_chain_count_truth_coverage_2026-08-14 \\
        --npz-dir data/indicators_v2/boards_lean_phase_l_2026-08-11 \\
        --videos c10,c109,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23 \\
        --out data/verify/chain_count_v2_2026-08-14/score_reversal_coverage.json
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
from src.chain_count_truth import select_chain_count_high_confidence_band
from src.chain_detector import ERASURE_MIN_DROP
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import is_pure_chain_score_delta

DEFAULT_NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
DEFAULT_OUT_PATH = Path("data/verify/chain_count_v2_2026-08-14/score_reversal_coverage.json")

# 16動画資産のデフォルト一覧 (WSL ~/frames/ 削除禁止資産、CLAUDE.md/MEMORY.md 記載)。
# c96 は npz 未生成のため除外 (実測、下記 main() の存在チェックで自動スキップ)。
DEFAULT_VIDEO_IDS: tuple[str, ...] = (
    "c10", "c109", "c11", "c12", "c13", "c14", "c15", "c16",
    "c17", "c18", "c19", "c20", "c21", "c22", "c23", "c96",
)

# chain_mechanism が「連鎖タグなし」を表す値
# (_build_chain_length_conditional_2026-08-13.py と同一定義、意図的に複製。
# 重い依存を持つ共通モジュールを新設するコストを避けるための使い捨てツール
# 間の慣行、元スクリプトの同名コメント参照)。
_NO_CHAIN_TAG_VALUES: frozenset[str] = frozenset({"", "nan", "none"})


def _find_before_board_index(nz_counts: list[int], idxs: list[int], pos: int) -> int | None:
    """タグ行 (idxs[pos]) から遡り、発火前盤面の行 index を探す。

    `_build_chain_length_conditional_2026-08-13.py` と同一ロジック
    (意図的な複製、モジュール docstring 参照)。
    """
    for j in range(pos, 0, -1):
        if nz_counts[idxs[j - 1]] - nz_counts[idxs[j]] >= ERASURE_MIN_DROP:
            return idxs[j - 1]
    return None


def _events_in_file(npz_path: Path) -> list[dict]:
    """1 npz ファイル内の全連鎖イベントについて、delta_score 等の実測値を返す。

    各イベント dict: {delta_score, sim_chain_count, t_sec, side, game_idx}。
    """
    d = np.load(npz_path, allow_pickle=True)
    if "chain_trigger_sec" not in d.files or "chain_mechanism" not in d.files:
        return []
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    score = d["score"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]
    nz_counts = (grids != 0).sum(axis=(1, 2)).tolist()

    groups: dict[tuple, list[int]] = {}
    for i in range(len(grids)):
        key = (str(side[i]), int(game_idx[i]))
        groups.setdefault(key, []).append(i)

    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    events: list[dict] = []
    for (side_key, game_idx_key), idxs in groups.items():
        idxs.sort(key=lambda i: float(t_sec[i]))
        prev_trigger_sec: float | None = None
        for pos in range(len(idxs)):
            i = idxs[pos]
            if not np.isfinite(trigger[i]):
                prev_trigger_sec = None
                continue
            tag = str(mechanism[i]).strip().lower()
            if tag in _NO_CHAIN_TAG_VALUES:
                prev_trigger_sec = None
                continue
            if prev_trigger_sec is not None and float(trigger[i]) == prev_trigger_sec:
                continue
            prev_trigger_sec = float(trigger[i])
            before_i = _find_before_board_index(nz_counts, idxs, pos)
            if before_i is None:
                continue
            before_board = Board.from_list(grids[before_i].tolist())
            if before_board.is_dead():
                continue
            sim_result = sim.simulate(before_board)
            if sim_result.chain_count < 1:
                continue  # 誤検出 (VideoChainTracker.update と同じ判定則)
            delta_score = int(score[i]) - int(score[before_i])
            events.append({
                "side": side_key,
                "game_idx": int(game_idx_key),
                "t_sec": float(t_sec[i]),
                "delta_score": delta_score,
                "sim_chain_count": int(sim_result.chain_count),
            })
    return events


def _summarize(all_events: list[dict]) -> dict:
    """全イベントを集計し、高信頼帯の解決率 (=旧W3の0%と対比する指標) を出す。"""
    n_total = len(all_events)
    n_pure = 0
    n_high_confidence = 0
    n_ratio_out_of_band = 0
    n_contaminated = 0
    n_agree_with_sim = 0
    n_disagree_with_sim = 0
    reason_counts: Counter[str] = Counter()
    for ev in all_events:
        delta = ev["delta_score"]
        is_pure = is_pure_chain_score_delta(delta)
        n_pure += int(is_pure)
        hc = select_chain_count_high_confidence_band(delta)
        reason_counts[hc.reason] += 1
        if hc.reason == "high_confidence":
            n_high_confidence += 1
            if hc.chain_count == ev["sim_chain_count"]:
                n_agree_with_sim += 1
            else:
                n_disagree_with_sim += 1
        elif hc.reason == "ratio_out_of_band":
            n_ratio_out_of_band += 1
        elif hc.reason == "contaminated":
            n_contaminated += 1
    resolution_rate = (n_high_confidence / n_total) if n_total else 0.0
    return {
        "total_events": n_total,
        "n_pure_chain_score_delta": n_pure,
        "n_high_confidence": n_high_confidence,
        "n_ratio_out_of_band": n_ratio_out_of_band,
        "n_contaminated_non_multiple_of_10": n_contaminated,
        "resolution_rate": resolution_rate,
        "n_high_confidence_agree_with_simulate": n_agree_with_sim,
        "n_high_confidence_disagree_with_simulate": n_disagree_with_sim,
        "reason_breakdown": dict(reason_counts),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "methodology": (
            "得点逆算のみ (テロップ非依存) で select_chain_count_high_confidence_band "
            "(比率帯[0.9,1.1] + delta_score%10==0 の層別フィルタ) を適用。"
            "resolution_rate = 高信頼帯を満たしたイベント数 / 全イベント数。"
            "旧W3 (テロップ複数動画0%解決) との対比指標であり、テロップ側の"
            "解決率とは別の計測軸 (本スクリプトはテロップ処理を含まない)。"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    ap.add_argument("--videos", type=str, default=",".join(DEFAULT_VIDEO_IDS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    per_video: dict[str, dict] = {}
    all_events: list[dict] = []
    skipped: list[str] = []
    for vid in video_ids:
        p = args.npz_dir / f"{vid}.npz"
        if not p.is_file():
            skipped.append(vid)
            continue
        events = _events_in_file(p)
        all_events.extend(events)
        per_video[vid] = _summarize(events)
        print(f"[coverage] {vid}: events={len(events)} "
              f"resolution_rate={per_video[vid]['resolution_rate']:.3f}")

    overall = _summarize(all_events)
    out = {
        "overall": overall,
        "per_video": per_video,
        "skipped_videos_missing_npz": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coverage] {args.out} に保存 "
          f"(total_events={overall['total_events']}, "
          f"resolution_rate={overall['resolution_rate']:.3f}, "
          f"skipped={skipped})")


if __name__ == "__main__":
    main()
