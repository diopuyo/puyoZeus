"""項目5後半: 「25件の着弾landed行」を得点整合性チェックで濾過し、
不整合イベントを除外した場合に項目1-4の数値がどう変わるかを見る (2026-07-29)。

読み取り専用の診断スクリプト。src/、scripts/measure_exchange_dynamics.py、
scripts/_diag_landing_bimodal_2026-07-29.py は import のみで一切変更しない。

突き合わせ方法: landed行 (video_stem, fire_side, game_idx, chain_count) を
_process_video() の defrag FireEvent 列と (game_idx一致 + t_chain_start最近傍)
でマッチングし、その FireEvent の delta_score で score_consistency_ratio を計算する。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from src.scoring import is_score_consistent  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    SCORE_MISSING_SENTINEL, _process_video,
)

# 2026-07-29修正: exchange_landing_delay_regen_2026-07-28.csv は
# boards_lean_fixed_regen_2026-07-28 (#51反映高速化3修正 適用後) の npz から
# 生成されている (scripts/_tmp_measure_landing_regen_2026-07-28.py:35-36 参照)。
# measure_exchange_dynamics.NPZ_DIR (= boards_lean_fixed、2026-07-18時点の旧npz)
# を使うと同じ発火イベントでも t_chain_start/chain_count がズレて誤マッチする
# (実測: 旧npzでは25件中7件しかマッチせず、うち一致確認できたのは4件のみ)。
NPZ_DIR = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bimodal = _load_module(
    "_diag_landing_bimodal_2026_07_29",
    PROJ_ROOT / "scripts" / "_diag_landing_bimodal_2026-07-29.py",
)
score_check = _load_module(
    "_verify_score_consistency_2026_07_29",
    PROJ_ROOT / "scripts" / "_verify_score_consistency_2026-07-29.py",
)


def _reconstruct_before_board(npz_path: Path, ev):
    return score_check._reconstruct_before_board(npz_path, ev)


def main() -> None:
    landed = bimodal.load_landed_rows(bimodal.EXCHANGE_CSV)
    sim = ChainSimulator()

    # video_stem別に defrag FireEvent 一覧をキャッシュ
    defrag_cache: dict[str, list] = {}

    matched_rows = []
    unmatched = []
    for row in landed:
        vs = row["video_stem"]
        npz_path = NPZ_DIR / f"{vs}.npz"
        if vs not in defrag_cache:
            if not npz_path.exists():
                defrag_cache[vs] = []
            else:
                _, defrag, _ = _process_video(npz_path, sim, 0)
                defrag_cache[vs] = defrag
        defrag = defrag_cache[vs]
        fire_side = row["fire_side"]
        game_idx = int(float(row["game_idx"]))
        t_chain_start = float(row["t_chain_start"])
        cand = [
            e for e in defrag
            if e.fire_side == fire_side and e.game_idx == game_idx
        ]
        if not cand:
            unmatched.append(row)
            continue
        ev = min(cand, key=lambda e: abs(e.t_chain_start - t_chain_start))
        # 時刻ズレが大きすぎる場合は誤マッチとみなす (同一game_idx内で
        # 複数発火があるケースの取り違え防止)。
        if abs(ev.t_chain_start - t_chain_start) > 3.0:
            unmatched.append(row)
            continue
        if ev.delta_score == SCORE_MISSING_SENTINEL:
            row["_score_status"] = "score欠損(判定不能)"
            matched_rows.append(row)
            continue
        before = _reconstruct_before_board(npz_path, ev)
        if before is None:
            row["_score_status"] = "before復元失敗(判定不能)"
            matched_rows.append(row)
            continue
        from src.scoring import calculate_chain_score  # noqa: E402
        result = sim.simulate(before)
        expected = calculate_chain_score(result).total_score
        consistent = is_score_consistent(expected, ev.delta_score)
        row["_score_status"] = "整合" if consistent else "不整合"
        row["_expected_score"] = expected
        row["_observed_delta_score"] = ev.delta_score
        matched_rows.append(row)

    print(f"landed行数: {len(landed)}")
    print(f"FireEventマッチ成功: {len(matched_rows)} / マッチ不能: {len(unmatched)}")
    if unmatched:
        print("マッチ不能 video_stem:", sorted({r['video_stem'] for r in unmatched}))

    print("\n[個別内訳]")
    header = f"{'video':6} {'fire':4} {'chain':5} {'delay_sec':9} {'score判定':16}"
    print(header)
    for r in sorted(matched_rows, key=lambda x: (x["video_stem"], x["fire_side"])):
        print(
            f"{r['video_stem']:6} {r['fire_side']:4} {r['chain_count']:>5} "
            f"{float(r['delay_sec']):9.2f} {r['_score_status']:16}"
        )

    from collections import Counter
    status_counts = Counter(r["_score_status"] for r in matched_rows)
    print(f"\n[判定サマリ] {dict(status_counts)}")

    consistent_rows = [r for r in matched_rows if r["_score_status"] == "整合"]
    inconsistent_rows = [r for r in matched_rows if r["_score_status"] == "不整合"]
    print(
        f"\n25件中: 整合={len(consistent_rows)} 不整合={len(inconsistent_rows)} "
        f"判定不能(score欠損等)={len(matched_rows) - len(consistent_rows) - len(inconsistent_rows)} "
        f"マッチ不能={len(unmatched)}"
    )

    # ------------------------------------------------------------------
    # 不整合行を除外した場合の項目1 (delay_sec 連鎖数別) 再集計
    # ------------------------------------------------------------------
    print("\n=== 不整合行を除外した delay_sec 連鎖数別再集計 (項目1) ===")
    by_chain_all: dict[int, list[float]] = {}
    by_chain_consistent_only: dict[int, list[float]] = {}
    excluded_video_ids: set[str] = {
        f"{r['video_stem']}_{r['fire_side']}_{r['game_idx']}" for r in inconsistent_rows
    }
    for row in landed:
        cc = int(float(row["chain_count"]))
        d = float(row["delay_sec"])
        by_chain_all.setdefault(cc, []).append(d)
        key = f"{row['video_stem']}_{row['fire_side']}_{int(float(row['game_idx']))}"
        if key not in excluded_video_ids:
            by_chain_consistent_only.setdefault(cc, []).append(d)

    for cc in sorted(set(by_chain_all) | set(by_chain_consistent_only)):
        print(f"  連鎖数={cc}:")
        bimodal.summarize(by_chain_all.get(cc, []), "    除外前")
        bimodal.summarize(by_chain_consistent_only.get(cc, []), "    不整合除外後")

    # ------------------------------------------------------------------
    # 不整合行を除外した場合の項目3 (連鎖アニメ中実測手数、study CSV要) 再集計
    # ------------------------------------------------------------------
    print("\n=== 不整合行を除外した 連鎖アニメ中実測手数 連鎖数別再集計 ===")
    by_cc_all: dict[int, list[int]] = {}
    by_cc_consistent_only: dict[int, list[int]] = {}
    for row in landed:
        vs = row["video_stem"]
        paths = bimodal.find_study_files(vs)
        if not paths:
            continue
        defender_side = bimodal.defender_side_of(row["fire_side"])
        series = bimodal.load_defender_series(paths, defender_side)
        t_chain_start = float(row["t_chain_start"])
        t_fire = float(row["t_fire"])
        cc = int(float(row["chain_count"]))
        inc_times = bimodal.all_increment_times(series)
        n_hands_actual = sum(1 for t in inc_times if t_chain_start < t <= t_fire)
        window_gap = bimodal.max_gap_in_window(series, t_chain_start, t_fire)
        if window_gap > bimodal.COVERAGE_GAP_THRESHOLD_SEC:
            continue  # 収集断絶帯 (既存ロジックと同じ除外基準)
        by_cc_all.setdefault(cc, []).append(n_hands_actual)
        key = f"{row['video_stem']}_{row['fire_side']}_{int(float(row['game_idx']))}"
        if key not in excluded_video_ids:
            by_cc_consistent_only.setdefault(cc, []).append(n_hands_actual)

    for cc in sorted(set(by_cc_all) | set(by_cc_consistent_only)):
        print(f"  連鎖数={cc}:")
        bimodal.summarize([float(v) for v in by_cc_all.get(cc, [])], "    除外前")
        bimodal.summarize(
            [float(v) for v in by_cc_consistent_only.get(cc, [])], "    不整合除外後",
        )


if __name__ == "__main__":
    main()
