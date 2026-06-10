"""
scripts/generate_training_dataset.py のスモークテスト

I/O・CSV 書き出し・サンプリング時刻計算の単体検証。動画読み込みは行わない。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.old.generate_training_dataset import (
    DEFAULT_TIME_PHASES,
    FEATURE_NAMES,
    MIN_MATCH_DURATION_SEC,
    PHASE_OFFSET_FROM_START,
    TIME_PHASE_END_MINUS,
    TIME_PHASE_MIDPOINT,
    TIME_PHASE_MID_MINUS,
    TIME_PHASE_MID_PLUS,
    TIME_PHASE_START_PLUS,
    FeatureRow,
    MatchMeta,
    collect_match_meta,
    compute_sample_time,
    load_boundaries,
    load_winners,
    write_csv,
)


# ============================
# load_winners / load_boundaries
# ============================


def test_load_winners_parses_tsv(tmp_path: Path) -> None:
    """match_winners TSV のパースを検証する。"""
    tsv = tmp_path / "winners.tsv"
    tsv.write_text(
        "idx\tstart_sec\tend_sec\twinner\textra\n"
        "1\t10.0\t70.0\t1P\tx\n"
        "2\t80.0\t140.0\t2P\tx\n"
        "3\t150.0\t210.0\tUNKNOWN\tx\n",
        encoding="utf-8",
    )
    winners = load_winners(tsv)
    assert winners == {1: "1P", 2: "2P"}


def test_load_boundaries_parses_tsv(tmp_path: Path) -> None:
    """matches.tsv のパースを検証する。"""
    tsv = tmp_path / "matches.tsv"
    tsv.write_text(
        "idx\tstart_sec\tend_sec\tduration\n"
        "1\t10.0\t70.0\t60.0\n"
        "2\t80.0\t140.0\t60.0\n",
        encoding="utf-8",
    )
    bounds = load_boundaries(tsv)
    assert bounds == {1: (10.0, 70.0), 2: (80.0, 140.0)}


def test_collect_match_meta_filters_short_matches(tmp_path: Path) -> None:
    """MIN_MATCH_DURATION_SEC 未満の試合は除外されることを検証。"""
    boundary_dir = tmp_path / "bound"
    winners_dir = tmp_path / "win"
    (boundary_dir / "video_99").mkdir(parents=True)
    winners_dir.mkdir()
    (boundary_dir / "video_99" / "matches.tsv").write_text(
        f"idx\tstart_sec\tend_sec\n"
        f"1\t0.0\t{MIN_MATCH_DURATION_SEC + 5}\n"
        f"2\t100.0\t{100.0 + MIN_MATCH_DURATION_SEC - 1}\n",
        encoding="utf-8",
    )
    (winners_dir / "match_winners_v99.tsv").write_text(
        "idx\tstart\tend\twinner\n"
        "1\t0\t30\t1P\n"
        "2\t100\t124\t2P\n",
        encoding="utf-8",
    )
    metas = collect_match_meta(
        "99", boundary_dir=boundary_dir, winners_dir=winners_dir,
    )
    assert len(metas) == 1
    assert metas[0].match_idx == 1


# ============================
# compute_sample_time
# ============================


def _meta(start: float = 100.0, end: float = 200.0) -> MatchMeta:
    """テスト用メタ。"""
    return MatchMeta(
        video_id="01", match_idx=1, start_sec=start,
        end_sec=end, winner="1P",
    )


def test_compute_sample_time_phases_are_distinct() -> None:
    """5 フェーズの時刻が全て異なることを確認する。"""
    meta = _meta()
    times = {
        phase: compute_sample_time(meta, phase)
        for phase in DEFAULT_TIME_PHASES
    }
    assert len(set(times.values())) == 5


def test_compute_sample_time_midpoint() -> None:
    """midpoint は (start+end)/2 になる。"""
    assert compute_sample_time(_meta(), TIME_PHASE_MIDPOINT) == 150.0


def test_compute_sample_time_start_plus_20() -> None:
    """start + 20 秒。"""
    expected = 100.0 + PHASE_OFFSET_FROM_START
    assert compute_sample_time(_meta(), TIME_PHASE_START_PLUS) == expected


def test_compute_sample_time_clamps_to_match_range() -> None:
    """境界をはみ出す場合は試合範囲内に丸める。"""
    short = _meta(start=0.0, end=30.0)
    # mid_minus_20 は 15-20=-5 → start+1 にクランプ
    t = compute_sample_time(short, TIME_PHASE_MID_MINUS)
    assert 0.0 <= t <= 30.0
    # mid_plus_20 は 15+20=35 → end-1 にクランプ
    t = compute_sample_time(short, TIME_PHASE_MID_PLUS)
    assert 0.0 <= t <= 30.0


def test_compute_sample_time_unknown_phase_raises() -> None:
    """未知フェーズで ValueError。"""
    with pytest.raises(ValueError):
        compute_sample_time(_meta(), "unknown")


def test_compute_sample_time_end_minus_5() -> None:
    """end_minus_5 は end - 5 秒。"""
    meta = _meta(start=0.0, end=100.0)
    assert compute_sample_time(meta, TIME_PHASE_END_MINUS) == 95.0


# ============================
# write_csv
# ============================


def test_write_csv_round_trip(tmp_path: Path) -> None:
    """write_csv → 再読込で値が保持される。"""
    out = tmp_path / "out.csv"
    feat = {name: 0.5 for name in FEATURE_NAMES}
    feat["main_chain_maturity"] = -0.25
    rows = [
        FeatureRow(
            video_id="01", match_idx=1, time_phase=TIME_PHASE_MIDPOINT,
            features=feat, label=1,
        ),
        FeatureRow(
            video_id="02", match_idx=3, time_phase=TIME_PHASE_END_MINUS,
            features=feat, label=-1,
        ),
    ]
    write_csv(rows, out)
    with out.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert len(records) == 2
    assert records[0]["video_id"] == "01"
    assert records[0]["label"] == "1"
    assert float(records[0]["main_chain_maturity"]) == pytest.approx(-0.25)
    assert records[1]["label"] == "-1"


def test_feature_names_count() -> None:
    """FEATURE_NAMES は 47 個 (B-1.b で 45 → 47).

    内訳: 8 主 + 拡張 13 + Tier B 3 + I-J 形テンプレ 4 + Phase F 1 (rotation_skill)
        + Phase H1 16 (機能 7 + 戦況 8 + 形分類 1) + B-1.b 2 (sullen_gtr/fron)。
    Phase J 4 + Tier B 3 (planning_entropy / structure_solidity / base_flatness)
    + I-J 4 (form_gtr / form_llr / form_staircase / form_zabuton)。
    Phase F (B-4): rotation_skill (回し入れ巧拙) を追加し 29 個。
    Phase H1 (2026-05-08): 機能能力指標 7 個 + 戦況・タイミング指標 8 個 +
    GTR 折り返し位置 1 個 = 16 個追加し合計 45 個。
    B-1.b (2026-05-09): citrus610/ama 由来の Sullen GTR / Fron 派生形を追加し
    合計 47 個。Phase K 3 指標は多重共線性で学習に悪影響のため CSV から除外、
    IndicatorSet 上では計算継続 (推論ロジックで活用)。
    """
    assert len(FEATURE_NAMES) == 47
