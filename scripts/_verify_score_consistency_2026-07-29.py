"""軽量検証: 得点整合性チェック (src.scoring.score_consistency_ratio) を
既存 FireEvent (23動画・de-frag後) に適用し、simulate() の chain_count が
信頼できないイベントを検出する (2026-07-29)。

userドメイン知識訂正 (2026-07-29): 掛け算表示は連鎖数を直接表示しない
(表示されるのは「消えたぷよ数」と「ボーナス合計値」のみ) ため、
掛け算表示から連鎖数を読む方式は撤回。代わりに得点式の既知性を使い、
simulate(before_grid) の期待得点と実測 delta_score を突き合わせる。

読み取り専用の診断スクリプト。src/、scripts/measure_exchange_dynamics.py は
import のみで一切変更しない (既存資産の再利用、動画I/O・認識の再実行なし)。

使い方:
    PYTHONPATH=. ./venv/bin/python scripts/_verify_score_consistency_2026-07-29.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.scoring import (  # noqa: E402
    SCORE_CONSISTENCY_RATIO_MAX, SCORE_CONSISTENCY_RATIO_MIN,
    calculate_chain_score, is_score_consistent, score_consistency_ratio,
)
from scripts.measure_exchange_dynamics import (  # noqa: E402
    NPZ_DIR, SCORE_MISSING_SENTINEL, TIER_MAP, FireEvent, _load_npz,
    _process_video, _subset,
)

NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"

# c62/c44/c59/c21/c54 (userタスク指定の外れ値5動画、既知の疑義事例)
OUTLIER_CASES: list[tuple[str, str, int, float]] = [
    ("c62", "2P", 6, 700.4),
    ("c44", "2P", 2, 257.8),
    ("c59", "1P", 1, 261.2),
    ("c21", "2P", 2, 279.8),
    ("c54", "1P", 1, 252.6),
]


def _reconstruct_before_board(npz_path: Path, ev: FireEvent) -> Board | None:
    """FireEvent から before_board (before_idx 時点の grid) を復元する。

    _build_fire_event / _diag_tfire_reliability_2026-07-29.py と同じ
    再構成手順 (game_idx でサブセット後、before_idx でローカルインデックス)。
    """
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if ev.fire_side not in by_side:
        return None
    rec = by_side[ev.fire_side]
    mask = rec.game_idx == ev.game_idx
    g = _subset(rec, mask)
    if ev.before_idx < 0 or ev.before_idx >= len(g.t_sec):
        return None
    return Board.from_list(g.grids[ev.before_idx].tolist())


def _check_event(sim: ChainSimulator, npz_path: Path, ev: FireEvent) -> dict | None:
    """1 FireEvent の得点整合性チェック結果を返す (score欠損は None)。"""
    if ev.delta_score == SCORE_MISSING_SENTINEL:
        return None
    before = _reconstruct_before_board(npz_path, ev)
    if before is None:
        return None
    result = sim.simulate(before)
    expected = calculate_chain_score(result).total_score
    ratio = score_consistency_ratio(expected, ev.delta_score)
    consistent = is_score_consistent(expected, ev.delta_score)
    return {
        "video_stem": ev.video_stem, "fire_side": ev.fire_side,
        "game_idx": ev.game_idx, "chain_count": ev.chain_count,
        "frag_count": ev.frag_count, "expected_score": expected,
        "observed_delta_score": ev.delta_score, "ratio": ratio,
        "consistent": consistent,
    }


def _run_all_events() -> pd.DataFrame:
    """23動画の de-frag 後 FireEvent 全件に整合性チェックを適用する。"""
    sim = ChainSimulator()
    rows: list[dict] = []
    for stem in sorted(TIER_MAP):
        npz_path = NPZ_DIR / f"{stem}.npz"
        if not npz_path.exists():
            continue
        _, defrag, _ = _process_video(npz_path, sim, 0)
        for ev in defrag:
            row = _check_event(sim, npz_path, ev)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _run_outlier_cases() -> pd.DataFrame:
    """userタスク指定の外れ値5動画を個別にチェックする。"""
    sim = ChainSimulator()
    rows: list[dict] = []
    for stem, side, game_idx, t_cs_approx in OUTLIER_CASES:
        npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
        if not npz_path.exists():
            npz_path = NPZ_DIR / f"{stem}.npz"
        if not npz_path.exists():
            rows.append({"video_stem": stem, "note": "npz不在"})
            continue
        _, defrag, _ = _process_video(npz_path, sim, 0)
        cand = [e for e in defrag if e.fire_side == side]
        if not cand:
            rows.append({"video_stem": stem, "note": "該当side イベントなし"})
            continue
        target = min(cand, key=lambda e: abs(e.t_chain_start - t_cs_approx))
        row = _check_event(sim, npz_path, target)
        if row is None:
            row = {"video_stem": stem, "note": "score欠損 or before復元失敗"}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[定数] 許容比率 = [{SCORE_CONSISTENCY_RATIO_MIN}, {SCORE_CONSISTENCY_RATIO_MAX}]")

    print("\n=== (1) 23動画 de-frag後 FireEvent 全件の整合性チェック ===")
    df = _run_all_events()
    n_total = len(df)
    n_inconsistent = int((~df["consistent"]).sum()) if n_total else 0
    print(f"score有効イベント数: {n_total}")
    print(f"不整合フラグ件数: {n_inconsistent} ({n_inconsistent / max(1, n_total):.3%})")
    if n_total:
        print("\n[frag_count 別の不整合率] (de-frag 統合数と不整合率の関係)")
        print(df.groupby("frag_count")["consistent"].agg(["mean", "count"]))
        print("\n[不整合イベント上位10件 (ratio が1から遠い順)]")
        df["log_dist"] = (df["ratio"].replace(0, np.nan)).apply(
            lambda r: abs(np.log(r)) if r and np.isfinite(r) and r > 0 else np.inf,
        )
        bad = df[~df["consistent"]].sort_values("log_dist", ascending=False)
        print(bad.head(10).drop(columns=["log_dist"]).to_string(index=False))

    print("\n=== (2) 外れ値5動画 (user指定) 個別チェック ===")
    out = _run_outlier_cases()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
