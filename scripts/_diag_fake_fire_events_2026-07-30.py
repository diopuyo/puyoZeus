"""読み取り専用の診断: 試合外画面 (SEGAロゴ/実況者カットイン) 由来の
偽 FireEvent 汚染量を測定する (2026-07-30)。

## 背景

scripts/_extract_chain_count_disagreement_frames_2026-07-30.py の人手目視レビューで、
当初選定した3事例 (原C=c11 2P game_idx=2, 原D=c11 1P game_idx=16(最終ゲーム),
原F=c16 2P game_idx=6) は「連鎖数の食い違い」ですらなく、window全体が
試合外画面 (SEGAロゴ・実況者カットイン) しか写っていないことが判明した
(同ファイル docstring 内 差し替え履歴 参照)。本スクリプトは、この汚染が
FireEvent 母集団全体でどれだけの規模か・特定動画偏在か全体薄広がりかを、
修正は行わず測定のみで定量化する。

## 判定方法 (2種類、根拠は各関数の docstring 参照)

(A) tail_suspect (主指標・実証済み):
    FireEvent.t_fire が自分自身の (fire_side, game_idx) の npz 収録範囲の
    末尾に近いかどうか。3件の既知汚染事例で実測した remaining_to_game_end_sec
    は 0.0秒 / 0.8秒 / 1.6秒 といずれも極めて小さく (下記 _report_ground_truth_check
    で再現確認)、既存プロジェクト定数 MATCH_END_REMAINING_SEC=5.0
    (scripts/measure_exchange_dynamics.py、_diag_gap_cause_2026-07-29.py で
    no_future_frames の86パーセントが残り5秒以内として実測校正済み) をそのまま
    再利用して閾値とする。根拠: game_idx は score リセット検知のみで区切られ
    (collect_boards_lean.py)、force_in_match=True によりis_match_active分岐
    が無効化されているため (src/recognition_pipeline.py)、試合終了後の
    SEGA/カットイン画面はスコアが大幅減少するまで同じ game_idx の末尾に
    紛れ込む。これは npz のみで判定可能 (動画削除後でも算出可能)。
    注意: game_idx 末尾に近いことは汚染の必要条件を捉えているだけで、
    真に試合外画面かどうかを保証しない (誤検出の可能性は残る、目視確認は
    3件のみ)。

(B) physics_suspect (補助指標・弱い):
    STABLE 確定盤面として記録された before/after grid が
    src/self_supervised/physical_consistency.py の物理ルール
    (重力・色数5以下・4連結以上未消去なし) に違反していないか。
    ロゴ/カットイン画面の誤認識盤面は物理的にありえないパターンになりやすい
    という仮説に基づくが、この仮説自体は本スクリプトでは検証しない
    (既知3事例が物理違反を伴うかは _report_ground_truth_check で報告するのみ)。

## 制約 (userタスク指定)

- 修正は一切書かない。測定のみ。
- src/, scripts/measure_exchange_dynamics.py, scripts/measure_ojama_landing_delay.py
  は import のみで変更しない。
- 動画ファイルは使わない (npz集計のみ、軽量)。

使い方:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_fake_fire_events_2026-07-30.py
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
from src.scoring import calculate_chain_score, is_score_consistent  # noqa: E402
from src.self_supervised.physical_consistency import (  # noqa: E402
    check_color_count, check_gravity_rule, check_no_pre_chain_4_plus_connection,
)
from scripts.measure_exchange_dynamics import (  # noqa: E402
    MATCH_END_REMAINING_SEC, NPZ_DIR as NPZ_DIR_OLD, SCORE_MISSING_SENTINEL,
    TIER_MAP, FireEvent, NpzRecord, _load_npz, _process_video, _subset,
)
from scripts.measure_ojama_landing_delay import _measure_landing_for_video  # noqa: E402

# regen npz (#51修正後、2026-07-28収集、userタスク指定の対象)
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"

# 出力 (読み取り専用診断の副産物、再現性のため保存)
OUT_CSV: Path = PROJ_ROOT / "data" / "verify" / "fake_fire_events_diag_2026-07-30.csv"

# 連鎖数ビン (measure_exchange_dynamics.CHAIN_BIN_CAP と同じ考え方で層別表示用)
BIN_EDGES: tuple[tuple[int, int, str], ...] = (
    (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"),
    (5, 7, "5-7"), (8, 12, "8-12"), (13, 999, "13+"),
)

# 2026-07-30 実フレーム目視レビューで「window全体が試合外画面」と確定済みの
# 3事例 (_extract_chain_count_disagreement_frames_2026-07-30.py 差し替え履歴より
# 転記、t_fire は概算一致キー)。tail_suspect 判定の感度検証専用。
KNOWN_CONTAMINATED: tuple[tuple[str, str, int, float], ...] = (
    ("c11", "2P", 2, 318.8),
    ("c11", "1P", 16, 1192.4),
    ("c16", "2P", 6, 709.8),
)


def _compute_game_bounds(
    by_side: dict[str, NpzRecord],
) -> dict[tuple[str, int], tuple[float, float]]:
    """(side, game_idx) ごとの npz 収録 t_sec 範囲 (min, max) を返す。"""
    bounds: dict[tuple[str, int], tuple[float, float]] = {}
    for side, rec in by_side.items():
        for gidx in np.unique(rec.game_idx):
            mask = rec.game_idx == gidx
            bounds[(side, int(gidx))] = (
                float(rec.t_sec[mask].min()), float(rec.t_sec[mask].max()),
            )
    return bounds


def _proximity_flags(ev: FireEvent, bounds: dict[tuple[str, int], tuple[float, float]]) -> dict:
    """FireEvent が自分の (side, game_idx) 収録範囲の末尾/先頭にどれだけ近いか。"""
    key = (ev.fire_side, ev.game_idx)
    if key not in bounds:
        return {"remaining_to_game_end_sec": None, "elapsed_since_game_start_sec": None,
                "tail_suspect": None, "head_suspect": None}
    t_min, t_max = bounds[key]
    remaining = t_max - ev.t_fire
    elapsed = ev.t_chain_start - t_min
    return {
        "remaining_to_game_end_sec": remaining,
        "elapsed_since_game_start_sec": elapsed,
        "tail_suspect": remaining <= MATCH_END_REMAINING_SEC,
        "head_suspect": elapsed <= MATCH_END_REMAINING_SEC,
    }


def _boards_from_record(rec: NpzRecord, ev: FireEvent) -> tuple[Board | None, Board | None]:
    """FireEvent の before/after 盤面を、既読込済み NpzRecord から復元する。"""
    mask = rec.game_idx == ev.game_idx
    g = _subset(rec, mask)

    def _at(idx: int) -> Board | None:
        if idx < 0 or idx >= len(g.t_sec):
            return None
        return Board.from_list(g.grids[idx].tolist())

    return _at(ev.before_idx), _at(ev.fi_idx)


def _physics_flags(before: Board | None, after: Board | None) -> dict[str, bool | None]:
    """盤面の物理妥当性フラグ (重力/色数/未消去連結、試合外誤認識の補助指標)。"""
    flags: dict[str, bool | None] = {
        "gravity_violation_before": None, "gravity_violation_after": None,
        "color_count_violation_before": None, "color_count_violation_after": None,
        "unresolved_4plus_after": None,
    }
    if before is not None:
        ok, _ = check_gravity_rule(before)
        flags["gravity_violation_before"] = not ok
        ok, _ = check_color_count(before)
        flags["color_count_violation_before"] = not ok
    if after is not None:
        ok, _ = check_gravity_rule(after)
        flags["gravity_violation_after"] = not ok
        ok, _ = check_color_count(after)
        flags["color_count_violation_after"] = not ok
        ok, _ = check_no_pre_chain_4_plus_connection(after)
        flags["unresolved_4plus_after"] = not ok
    return flags


def _consistency_flag(
    sim: ChainSimulator, before: Board | None, delta_score: int,
) -> tuple[float | None, bool | None]:
    """simulate(before) の期待得点と実測 delta_score の整合性。"""
    if before is None or delta_score == SCORE_MISSING_SENTINEL:
        return None, None
    expected = calculate_chain_score(sim.simulate(before)).total_score
    return float(expected), bool(is_score_consistent(expected, delta_score))


def _build_event_row(
    ev: FireEvent,
    bounds: dict[tuple[str, int], tuple[float, float]],
    by_side: dict[str, NpzRecord],
    sim: ChainSimulator,
    landing: dict,
) -> dict:
    """1 FireEvent 分の全診断列 (層別キー+汚染フラグ+下流数値) を組み立てる。"""
    before, after = _boards_from_record(by_side[ev.fire_side], ev)
    row: dict = {
        "video_stem": ev.video_stem, "tier": ev.tier, "fire_side": ev.fire_side,
        "game_idx": ev.game_idx, "t_chain_start": ev.t_chain_start, "t_fire": ev.t_fire,
        "chain_count": ev.chain_count, "delta_score": ev.delta_score,
        "ojama_sent_count": ev.ojama_sent_count, "frag_count": ev.frag_count,
    }
    row.update(_proximity_flags(ev, bounds))
    row.update(_physics_flags(before, after))
    expected, consistent = _consistency_flag(sim, before, ev.delta_score)
    row["expected_score"] = expected
    row["score_consistent"] = consistent
    row["t_landed"] = landing.get("t_landed")
    row["delay_from_chain_start_sec"] = landing.get("delay_from_chain_start_sec")
    row["opp_available"] = landing.get("opp_available")
    row["detection_status"] = landing.get("detection_status")
    return row


def _process_all_videos(npz_dir: Path) -> pd.DataFrame:
    """指定 npz ディレクトリの23動画全 FireEvent を診断列つきで返す。"""
    sim = ChainSimulator()
    rows: list[dict] = []
    seq_id = 0
    for stem in sorted(TIER_MAP):
        npz_path = npz_dir / f"{stem}.npz"
        if not npz_path.exists():
            print(f"  [SKIP] {stem}: npz不在 ({npz_dir.name})")
            continue
        records = _load_npz(npz_path)
        by_side = {r.side: r for r in records}
        bounds = _compute_game_bounds(by_side)
        _, defrag, seq_id = _process_video(npz_path, sim, seq_id)
        landing_rows = _measure_landing_for_video(npz_path, defrag)
        for ev, landing in zip(defrag, landing_rows):
            rows.append(_build_event_row(ev, bounds, by_side, sim, landing))
    return pd.DataFrame(rows)


def _report_ground_truth_check(df: pd.DataFrame) -> None:
    """既知の試合外汚染3件で tail_suspect が正しく検出できるか確認する。"""
    print("\n" + "=" * 70)
    print("0. 既知汚染3件 (実フレーム目視確認済み) での proxy 感度検証")
    print("=" * 70)
    for stem, side, gidx, t_fire_approx in KNOWN_CONTAMINATED:
        sub = df[(df.video_stem == stem) & (df.fire_side == side) & (df.game_idx == gidx)]
        sub = sub[(sub["t_fire"] - t_fire_approx).abs() < 0.5]
        if sub.empty:
            print(f"  {stem} {side} g{gidx} (t approx {t_fire_approx}): 一致イベントなし (要確認)")
            continue
        r = sub.iloc[0]
        print(
            f"  {stem} {side} g{gidx}: remaining_to_game_end={r.remaining_to_game_end_sec:.2f}s "
            f"tail_suspect={r.tail_suspect} score_consistent={r.score_consistent} "
            f"gravity_violation_after={r.gravity_violation_after} "
            f"color_count_violation_after={r.color_count_violation_after}",
        )


def _report_overall(df: pd.DataFrame, label: str) -> None:
    """全体の tail_suspect / physics_suspect 件数・率。"""
    print(f"\n[{label}] 全 FireEvent 数 = {len(df)}")
    n_tail = int((df["tail_suspect"] == True).sum())  # noqa: E712
    print(f"  tail_suspect (主指標): {n_tail} 件 ({n_tail / max(1, len(df)):.2%})")
    n_head = int((df["head_suspect"] == True).sum())  # noqa: E712
    print(f"  head_suspect (参考・弱い): {n_head} 件 ({n_head / max(1, len(df)):.2%})")
    phys_cols = [
        "gravity_violation_before", "gravity_violation_after",
        "color_count_violation_after", "unresolved_4plus_after",
    ]
    for col in phys_cols:
        n = int((df[col] == True).sum())  # noqa: E712
        n_valid = int(df[col].notna().sum())
        print(f"  {col}: {n}/{n_valid} 件 ({n / max(1, n_valid):.2%})")


def _report_by_video(df: pd.DataFrame) -> None:
    """動画別 tail_suspect 率 (特定動画偏在か全体薄広がりかの判定材料)。"""
    print("\n[動画別 tail_suspect 率] (特定動画への偏在有無を確認)")
    g = df.groupby("video_stem").agg(
        n_fire=("tail_suspect", "size"),
        n_tail_suspect=("tail_suspect", lambda s: int((s == True).sum())),  # noqa: E712
    )
    g["tail_suspect_pct"] = (g["n_tail_suspect"] / g["n_fire"] * 100.0).round(1)
    print(g.sort_values("tail_suspect_pct", ascending=False).to_string())


def _bin_label(chain_count: int) -> str:
    for lo, hi, label in BIN_EDGES:
        if lo <= chain_count <= hi:
            return label
    return "13+"


def _report_by_side_and_chain(df: pd.DataFrame) -> None:
    """1P/2P別・連鎖数帯別の tail_suspect 率。"""
    print("\n[1P/2P別 tail_suspect 率]")
    g = df.groupby("fire_side")["tail_suspect"].apply(
        lambda s: (s == True).mean(),  # noqa: E712
    )
    print(g.to_string())

    print("\n[連鎖数帯別 tail_suspect 率]")
    tmp = df.copy()
    tmp["chain_bin"] = tmp["chain_count"].apply(_bin_label)
    g2 = tmp.groupby("chain_bin")["tail_suspect"].agg(
        n="size", rate=lambda s: (s == True).mean(),  # noqa: E712
    )
    print(g2.reindex([b[2] for b in BIN_EDGES]).to_string())


def _report_by_delta_score(df: pd.DataFrame) -> None:
    """delta_score の大きさ別 (quantile) tail_suspect 率。"""
    print("\n[delta_score 四分位別 tail_suspect 率]")
    valid = df[df["delta_score"] != SCORE_MISSING_SENTINEL].copy()
    if valid.empty:
        print("  (delta_score 有効データなし)")
        return
    valid["q"] = pd.qcut(valid["delta_score"], 4, duplicates="drop")
    g = valid.groupby("q")["tail_suspect"].agg(
        n="size", rate=lambda s: (s == True).mean(),  # noqa: E712
    )
    print(g.to_string())


def _summarize_delay(landed: pd.DataFrame) -> pd.DataFrame:
    """連鎖数ビン別 delay_from_chain_start_sec の中央値集計。"""
    landed = landed.copy()
    landed["bin"] = landed["chain_count"].apply(_bin_label)
    rows = []
    for _, _, label in BIN_EDGES:
        sub = landed[landed["bin"] == label]["delay_from_chain_start_sec"].dropna()
        rows.append({
            "chain": label, "n": len(sub),
            "median": round(float(sub.median()), 2) if len(sub) else None,
        })
    return pd.DataFrame(rows)


def _report_downstream_landing(df: pd.DataFrame) -> None:
    """tail_suspect 除外前後で着弾遅延の連鎖数別中央値がどう動くか。"""
    print("\n" + "=" * 70)
    print("下流影響 (A): 着弾遅延 連鎖数別中央値 (regen npz)")
    print("=" * 70)
    landed = df[df["detection_status"] == "landed"]
    print(f"[除外なし] landed n={len(landed)}")
    print(_summarize_delay(landed).to_string(index=False))
    excl = landed[landed["tail_suspect"] != True]  # noqa: E712
    print(f"\n[tail_suspect除外後] landed n={len(excl)}")
    print(_summarize_delay(excl).to_string(index=False))


def _report_downstream_consistency(df: pd.DataFrame, label: str) -> None:
    """tail_suspect 除外前後で得点不整合率がどう動くか。"""
    print(f"\n[{label}] 得点不整合率 (tail_suspect 除外前後)")
    valid = df[df["score_consistent"].notna()]
    n_total = len(valid)
    n_incons = int((~valid["score_consistent"].astype(bool)).sum())
    print(f"  除外なし: n={n_total} 不整合={n_incons} ({n_incons / max(1, n_total):.2%})")
    excl = valid[valid["tail_suspect"] != True]  # noqa: E712
    n_incons2 = int((~excl["score_consistent"].astype(bool)).sum())
    print(f"  tail_suspect除外後: n={len(excl)} 不整合={n_incons2} "
          f"({n_incons2 / max(1, len(excl)):.2%})")


def main() -> None:
    print("[INFO] regen npz (#51修正後, 2026-07-28) を処理中 ...")
    df_regen = _process_all_videos(NPZ_DIR_REGEN)
    print("[INFO] old npz (#51修正前, boards_lean_fixed) を処理中 ...")
    df_old = _process_all_videos(NPZ_DIR_OLD)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_regen.to_csv(OUT_CSV, index=False)
    print(f"\n[保存] {OUT_CSV} ({len(df_regen)}行、regen npz分)")

    _report_ground_truth_check(df_regen)

    print("\n" + "=" * 70)
    print("1. 全体件数・率 (regen npz)")
    print("=" * 70)
    _report_overall(df_regen, "regen npz (#51後)")
    _report_overall(df_old, "old npz (#51前, 参考)")

    print("\n" + "=" * 70)
    print("2. 層別 (regen npz)")
    print("=" * 70)
    _report_by_video(df_regen)
    _report_by_side_and_chain(df_regen)
    _report_by_delta_score(df_regen)

    _report_downstream_landing(df_regen)

    print("\n" + "=" * 70)
    print("下流影響 (B): 得点不整合率 (#51仮説との関係)")
    print("=" * 70)
    _report_downstream_consistency(df_regen, "regen npz (#51後)")
    _report_downstream_consistency(df_old, "old npz (#51前, 参考)")


if __name__ == "__main__":
    main()
