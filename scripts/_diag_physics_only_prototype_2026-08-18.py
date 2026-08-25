"""物理推論単独プロトタイプ検証 (2026-08-18、デバッガエージェント計装)。

user依頼: 新方式「物理推論 + エフェクト越し認識」の投資判断材料として、
STABLE snapshot 専用 npz (`data/indicators_v2/boards_lean_phase_l_2026-08-11`)
から、ある STABLE 盤面 A (エフェクト前) → 次の STABLE 盤面 B (エフェクト後)
の遷移を、A + 物理則だけからどこまで言い当てられるかを実測する。

対象は3動画 (36 / 52 / c100、data/frames/ に mp4 が現存し証拠フレーム抽出可能)。

## 測定1: 連鎖区間 (chain_trigger_sec タグ由来)
`_find_before_board_index` (scripts/_build_chain_length_conditional_2026-08-13.py
と同一手法、非空セル数が ERASURE_MIN_DROP 以上減少する直前行を「発火前盤面」
とみなす) で before_board を特定し、`ChainSimulator.simulate(before_board)` の
final_board を「物理推論だけの予測」として、観測後盤面 (afterタグ行群の最終行)
とセル単位で突合する。before_board 自体に消去可能グループが無い
(= 発火前盤面を捕捉できていない) 件数も同時に集計する (物理テストの前提が
崩れている頻度の指標)。

## 測定2: 連鎖以外の区間 (STABLE→STABLE 間で新規セルが出現)
グリッド差分から「おじゃま着弾/通常設置/相手バースト中設置/混在/変化なし」に
分類し (自他 side の chain_trigger_sec 重なりで相手バースト窓を判定)、各区間で
`enumerate_landing_patterns` の候補数 (=物理だけで残る自由度) を計測する。

実装しない。計装のみ (scripts/_diag_*)。
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, Board,
)
from src.chain import ChainSimulator  # noqa: E402
from src.chain_detector import ERASURE_MIN_DROP  # noqa: E402
from src.placement_inferrer import enumerate_landing_patterns  # noqa: E402
from src.production_config import GHOST_CHAIN_RULE_ENABLED  # noqa: E402

# ファイル名にハイフンを含むため動的 import (コピペ禁止指示への対応)。
_CLC = importlib.import_module("scripts._build_chain_length_conditional_2026-08-13")

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
TARGET_VIDEOS: tuple[str, ...] = ("36", "52", "c100")
OUT_LOG = Path("logs/_diag_physics_only_prototype_2026-08-18.json")

_NO_CHAIN_TAG_VALUES: frozenset[str] = frozenset({"", "nan", "none"})

# row0 (隠し段) は画面外推論のため、可視盤面 (row1-12) のみで一致率を測る。
VISIBLE_ROW_START: int = 1


# =============================================================================
# 測定1: 連鎖区間の物理予測精度
# =============================================================================


@dataclass
class ChainPredictionResult:
    video: str
    side: str
    game_idx: int
    before_idx: int
    after_idx: int
    valid_before: bool  # before_board に消去可能グループがあったか
    chain_count: int
    n_visible_cells: int
    n_match: int
    n_unknown_in_observed: int  # 観測後盤面側のUNKNOWNセル数 (母数から除外)
    match_rate: "float | None"


def _last_index_with_same_tag(
    trigger: "np.ndarray", idxs: list[int], pos: int,
) -> int:
    """同一 trigger_sec が連続する行 (hold窓の重複タグ) の最終行 index を返す。"""
    tag_val = float(trigger[idxs[pos]])
    j = pos
    while j + 1 < len(idxs) and np.isfinite(trigger[idxs[j + 1]]) and float(trigger[idxs[j + 1]]) == tag_val:
        j += 1
    return idxs[j]


def measure_chain_predictions(npz_path: Path) -> list[ChainPredictionResult]:
    """1 npz ファイル内の全連鎖イベントで物理予測とのセル一致率を測定する。"""
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]
    nz_counts = (grids != 0).sum(axis=(1, 2)).tolist()

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(grids)):
        groups[(str(side[i]), int(game_idx[i]))].append(i)

    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    out: list[ChainPredictionResult] = []
    vid = npz_path.stem

    for (s, g), idxs in groups.items():
        idxs.sort(key=lambda i: float(t_sec[i]))
        prev_trigger_sec: "float | None" = None
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

            before_i = _CLC._find_before_board_index(nz_counts, idxs, pos)
            if before_i is None:
                continue
            after_i = _last_index_with_same_tag(trigger, idxs, pos)

            before_board = Board.from_list(grids[before_i].tolist())
            if before_board.is_dead():
                continue
            result = sim.simulate(before_board)
            valid = result.chain_count >= 1

            observed_after = grids[after_i]
            predicted = result.final_board._grid

            vis_obs = observed_after[VISIBLE_ROW_START:, :]
            vis_pred = predicted[VISIBLE_ROW_START:, :]
            unknown_mask = vis_obs == COLOR_UNKNOWN
            n_unknown = int(unknown_mask.sum())
            comparable = ~unknown_mask
            n_cells = int(comparable.sum())
            n_match = int((vis_obs == vis_pred)[comparable].sum())
            match_rate = (n_match / n_cells) if n_cells > 0 else None

            out.append(ChainPredictionResult(
                video=vid, side=s, game_idx=g, before_idx=before_i, after_idx=after_i,
                valid_before=valid, chain_count=int(result.chain_count),
                n_visible_cells=n_cells, n_match=n_match,
                n_unknown_in_observed=n_unknown, match_rate=match_rate,
            ))
    return out


# =============================================================================
# 測定2: 連鎖以外の区間分類
# =============================================================================


@dataclass
class NonChainSegment:
    video: str
    side: str
    game_idx: int
    before_idx: int
    after_idx: int
    dt_sec: float
    category: str
    n_new_color: int
    n_new_ojama: int
    n_removed: int
    n_changed_existing: int
    n_landing_patterns_before: "int | None"  # おじゃま/通常設置系のみ計測


def _opponent_chain_overlaps(
    other_idxs: list[int], t_sec: "np.ndarray", trigger: "np.ndarray",
    mechanism: "np.ndarray", t_lo: float, t_hi: float,
) -> bool:
    """相手side (other_idxs) の chain_trigger_sec が (t_lo, t_hi] 内に存在するか。"""
    for j in other_idxs:
        tj = float(t_sec[j])
        if t_lo < tj <= t_hi and np.isfinite(trigger[j]):
            tag = str(mechanism[j]).strip().lower()
            if tag not in _NO_CHAIN_TAG_VALUES:
                return True
    return False


def measure_non_chain_segments(npz_path: Path) -> list[NonChainSegment]:
    """連鎖タグが付かない STABLE→STABLE 遷移を diff から分類する。"""
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(grids)):
        groups[(str(side[i]), int(game_idx[i]))].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: float(t_sec[i]))

    vid = npz_path.stem
    out: list[NonChainSegment] = []

    for (s, g), idxs in groups.items():
        other_s = "2P" if s == "1P" else "1P"
        other_idxs = groups.get((other_s, g), [])
        for pos in range(len(idxs) - 1):
            i, j = idxs[pos], idxs[pos + 1]
            tag = str(mechanism[j]).strip().lower()
            if np.isfinite(trigger[j]) and tag not in _NO_CHAIN_TAG_VALUES:
                continue  # 連鎖区間は測定1で扱う

            a = grids[i]
            b = grids[j]
            new_color_mask = (a == COLOR_EMPTY) & (b >= 1) & (b <= 5)
            new_ojama_mask = (a == COLOR_EMPTY) & (b == COLOR_OJAMA)
            removed_mask = (a != COLOR_EMPTY) & (a != COLOR_UNKNOWN) & (b == COLOR_EMPTY)
            changed_existing_mask = (
                (a != COLOR_EMPTY) & (a != COLOR_UNKNOWN)
                & (b != COLOR_EMPTY) & (b != COLOR_UNKNOWN) & (a != b)
            )
            n_new_color = int(new_color_mask.sum())
            n_new_ojama = int(new_ojama_mask.sum())
            n_removed = int(removed_mask.sum())
            n_changed = int(changed_existing_mask.sum())

            if n_new_color == 0 and n_new_ojama == 0 and n_removed == 0 and n_changed == 0:
                category = "no_change"
            elif n_removed > 0:
                category = "erasure_evidence_untagged"  # タグ漏れの消去 (連鎖検出漏れ)
            elif n_new_ojama > 0 and n_new_color > 0:
                category = "mixed_ojama_and_placement"
            elif n_new_ojama > 0:
                category = "ojama_landing"
            else:
                t_lo, t_hi = float(t_sec[i]), float(t_sec[j])
                if _opponent_chain_overlaps(other_idxs, t_sec, trigger, mechanism, t_lo, t_hi):
                    category = "opponent_burst_placement"
                else:
                    category = "normal_placement"

            n_patterns: "int | None" = None
            if category in ("normal_placement", "opponent_burst_placement", "ojama_landing"):
                before_board = Board.from_list(a.tolist())
                n_patterns = len(enumerate_landing_patterns(before_board))

            out.append(NonChainSegment(
                video=vid, side=s, game_idx=g, before_idx=i, after_idx=j,
                dt_sec=float(t_sec[j] - t_sec[i]), category=category,
                n_new_color=n_new_color, n_new_ojama=n_new_ojama,
                n_removed=n_removed, n_changed_existing=n_changed,
                n_landing_patterns_before=n_patterns,
            ))
    return out


# =============================================================================
# 集計・報告
# =============================================================================


def report_chain_predictions(results: list[ChainPredictionResult]) -> str:
    lines = ["=== 測定1: 連鎖区間の物理予測精度 ==="]
    n_total = len(results)
    valid = [r for r in results if r.valid_before]
    lines.append(f"連鎖イベント総数: {n_total}")
    lines.append(
        f"before_board に消去可能グループあり (物理テスト前提が成立): "
        f"{len(valid)}/{n_total} ({len(valid) / n_total * 100:.1f}%)" if n_total else "0件"
    )
    rates = [r.match_rate for r in valid if r.match_rate is not None]
    if rates:
        arr = np.array(rates)
        lines.append(
            f"セル一致率 (可視盤面row1-12、valid_beforeのみ, n={len(arr)}): "
            f"mean={arr.mean() * 100:.2f}% median={np.median(arr) * 100:.2f}% "
            f"min={arr.min() * 100:.2f}% p10={np.percentile(arr, 10) * 100:.2f}%"
        )
        exact = sum(1 for r in arr if r >= 0.999)
        lines.append(f"完全一致 (>=99.9%) 件数: {exact}/{len(arr)} ({exact / len(arr) * 100:.1f}%)")
        by_chain: dict[int, list[float]] = defaultdict(list)
        for r in valid:
            if r.match_rate is not None:
                by_chain[r.chain_count].append(r.match_rate)
        lines.append("連鎖数別内訳:")
        for k in sorted(by_chain):
            v = np.array(by_chain[k])
            lines.append(f"  {k}連鎖: n={len(v)} mean一致率={v.mean() * 100:.2f}%")
    else:
        lines.append("セル一致率: 有効データなし")
    n_unknown_total = sum(r.n_unknown_in_observed for r in valid)
    lines.append(f"観測後盤面のUNKNOWNセル (母数除外分) 合計: {n_unknown_total}")
    return "\n".join(lines)


def report_non_chain_segments(segments: list[NonChainSegment]) -> str:
    lines = ["\n=== 測定2: 連鎖以外の区間分類 ==="]
    cat_counter = Counter(s.category for s in segments)
    n_total = len(segments)
    lines.append(f"区間総数: {n_total}")
    for cat, cnt in cat_counter.most_common():
        lines.append(f"  {cat}: {cnt} ({cnt / n_total * 100:.1f}%)")

    for cat in ("normal_placement", "opponent_burst_placement", "ojama_landing"):
        subset = [s.n_landing_patterns_before for s in segments if s.category == cat and s.n_landing_patterns_before is not None]
        if subset:
            arr = np.array(subset)
            lines.append(
                f"{cat}: 物理的に妥当な着地パターン候補数 "
                f"mean={arr.mean():.1f} median={np.median(arr):.0f} "
                f"(min={arr.min()} max={arr.max()}) — 列位置は物理だけでは一意に決まらない"
            )
    n_ojama_new = sum(s.n_new_ojama for s in segments if s.category in ("ojama_landing", "mixed_ojama_and_placement"))
    lines.append(f"おじゃま新規着弾セル数 合計: {n_ojama_new}")
    return "\n".join(lines)


def main() -> None:
    all_chain: list[ChainPredictionResult] = []
    all_non_chain: list[NonChainSegment] = []
    for vid in TARGET_VIDEOS:
        npz_path = NPZ_DIR / f"{vid}.npz"
        if not npz_path.exists():
            print(f"[skip] {npz_path} が存在しません")
            continue
        print(f"[{vid}] 測定中...")
        all_chain.extend(measure_chain_predictions(npz_path))
        all_non_chain.extend(measure_non_chain_segments(npz_path))

    print(report_chain_predictions(all_chain))
    print(report_non_chain_segments(all_non_chain))

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(json.dumps({
        "target_videos": list(TARGET_VIDEOS),
        "chain_predictions": [asdict(r) for r in all_chain],
        "non_chain_segments": [asdict(s) for s in all_non_chain],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {OUT_LOG}")


if __name__ == "__main__":
    main()
