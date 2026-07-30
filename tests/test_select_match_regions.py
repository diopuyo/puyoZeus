"""scripts/select_match_regions.py の単体テスト (2026-07-30)。

compute_video_match_span / merge_contiguous_games は stateless な純粋関数の
ため、games 配列を直接組み立てて検証する。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.select_match_regions import (
    build_region_rows,
    compute_video_match_span,
    load_video_ids_from_csv,
    merge_contiguous_games,
    write_jobs_txt,
    write_report_csv,
)


def _game(idx: int, start: float, end: float, confidence: str = "strict") -> dict:
    """テスト用の games 配列要素を組み立てる。"""
    return {
        "game_abs_idx": idx, "start_sec": start, "end_sec": end,
        "winner": "1P", "confidence": confidence,
    }


# ============================
# compute_video_match_span
# ============================

def test_span_covers_first_start_to_last_end() -> None:
    """区間は最初の試合開始〜最後の試合終了 (実測: 動画冒頭末尾のみ削減対象)。"""
    games = [_game(0, 130.0, 246.0), _game(1, 246.0, 310.0), _game(2, 310.0, 400.0)]
    start, end, n = compute_video_match_span(games)
    assert start == 130.0
    assert end == 400.0
    assert n == 3


def test_span_confidence_filter_excludes_asymmetric() -> None:
    """confidence_filter={"strict"} で asymmetric 試合を除外できる。"""
    games = [
        _game(0, 130.0, 246.0, confidence="asymmetric"),
        _game(1, 246.0, 310.0, confidence="strict"),
    ]
    start, end, n = compute_video_match_span(games, confidence_filter={"strict"})
    assert start == 246.0
    assert end == 310.0
    assert n == 1


def test_span_margin_applied_and_clamped_to_zero() -> None:
    """head_margin_sec が start_sec を超えても 0.0 未満にはならない。"""
    games = [_game(0, 5.0, 60.0)]
    start, end, n = compute_video_match_span(
        games, head_margin_sec=10.0, tail_margin_sec=3.0,
    )
    assert start == 0.0  # 5.0 - 10.0 = -5.0 は 0.0 にクランプ
    assert end == 63.0


def test_span_empty_games_returns_none() -> None:
    """対象試合が0件なら None (決定不能)。"""
    assert compute_video_match_span([]) is None


def test_span_all_filtered_out_returns_none() -> None:
    """confidence_filter で全件除外されると None。"""
    games = [_game(0, 130.0, 246.0, confidence="asymmetric")]
    assert compute_video_match_span(games, confidence_filter={"strict"}) is None


# ============================
# merge_contiguous_games
# ============================

def test_merge_zero_gap_becomes_single_span() -> None:
    """試合間ギャップ0秒 (combined66実測どおり) なら1区間に併合される。"""
    games = [_game(0, 130.0, 246.0), _game(1, 246.0, 310.0), _game(2, 310.0, 400.0)]
    spans = merge_contiguous_games(games, max_gap_sec=0.0)
    assert spans == [(130.0, 400.0, [0, 1, 2])]


def test_merge_large_gap_splits_into_multiple_spans() -> None:
    """ギャップが max_gap_sec を超えると別区間に分かれる (将来動画向け汎用性)。"""
    games = [_game(0, 100.0, 150.0), _game(1, 500.0, 560.0)]
    spans = merge_contiguous_games(games, max_gap_sec=10.0)
    assert spans == [(100.0, 150.0, [0]), (500.0, 560.0, [1])]


def test_merge_empty_games_returns_empty_list() -> None:
    """空入力は空リストを返す。"""
    assert merge_contiguous_games([]) == []


# ============================
# build_region_rows / write_report_csv / write_jobs_txt (I/O 統合)
# ============================

def test_build_region_rows_reads_panel_json(tmp_path: Path) -> None:
    """panel_dir から video_id ごとの JSON を読み、区間行を組み立てる。"""
    panel_dir = tmp_path / "panels"
    panel_dir.mkdir()
    (panel_dir / "video_x.json").write_text(
        json.dumps({"video_id": "video_x", "games": [
            _game(0, 10.0, 50.0), _game(1, 50.0, 90.0),
        ]}), encoding="utf-8",
    )
    rows, skipped = build_region_rows(["video_x"], panel_dir)
    assert skipped == []
    assert rows == [{
        "video_id": "video_x", "start_sec": 10.0, "end_sec": 90.0,
        "dur_sec": 80.0, "n_games": 2,
    }]


def test_load_video_ids_from_csv_dedupes_and_sorts(tmp_path: Path) -> None:
    """video_id 列から重複排除・昇順ソートしたリストを得る。"""
    csv_path = tmp_path / "labeled.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "won"])
        writer.writeheader()
        writer.writerow({"video_id": "video_c20", "won": "1"})
        writer.writerow({"video_id": "video_c10", "won": "0"})
        writer.writerow({"video_id": "video_c10", "won": "1"})
    assert load_video_ids_from_csv(csv_path) == ["video_c10", "video_c20"]


def test_write_report_csv_roundtrip(tmp_path: Path) -> None:
    """CSV 書き出し → 読み戻しで内容が一致する。"""
    rows = [{
        "video_id": "video_x", "start_sec": 1.0, "end_sec": 2.0,
        "dur_sec": 1.0, "n_games": 1,
    }]
    out_path = tmp_path / "report.csv"
    write_report_csv(rows, out_path)
    with out_path.open(encoding="utf-8") as f:
        read_back = list(csv.DictReader(f))
    assert read_back[0]["video_id"] == "video_x"
    assert read_back[0]["dur_sec"] == "1.0"


def test_write_jobs_txt_uses_start_sec_and_dur_sec(tmp_path: Path) -> None:
    """ジョブ定義に --start-sec / --max-sec が正しく渡り、CRLF が混入しない。"""
    rows = [{
        "video_id": "video_x", "start_sec": 10.0, "end_sec": 90.0,
        "dur_sec": 80.0, "n_games": 2,
    }]
    out_path = tmp_path / "jobs.txt"
    write_jobs_txt(rows, out_path, out_npz_dir="out_dir")
    content = out_path.read_text(encoding="utf-8")
    assert "\r" not in content  # CRLF 混入事故対策 (feedback_crlf_churn_before_push)
    assert "--start-sec 10.0" in content
    assert "--max-sec 80.0" in content
    assert "out_dir/video_x.npz" in content
