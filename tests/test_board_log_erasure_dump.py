"""board_log JSONL への erasure_alerts dump + evaluate_recognition 統合テスト。

visualize_recognition.py:L790-813 の dump ロジックが erasure_alerts を
JSONL に書き出し、evaluate_recognition (RecognitionEvaluator) が
正しく p_to_e_count として集計できることを確認する。

このテストは実際のファイル I/O を伴う統合テストで、
tmp_path (pytest fixture) を使って一時ファイルに書き込む。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.recognition_evaluator import RecognitionEvaluator


# ============================
# ヘルパー: JSONL を作成する
# ============================

def _write_board_log(
    path: Path,
    entries: list[dict],
) -> None:
    """board_log JSONL を指定パスに書き出す。"""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _make_entry(
    frame_idx: int,
    p1_state: str = "stable",
    p2_state: str = "stable",
    p1_erasure_alerts: list[list[int]] | None = None,
    p2_erasure_alerts: list[list[int]] | None = None,
) -> dict:
    """board_log JSONL 1 行分の dict を生成する。
    backwards compat 確認のため erasure_alerts は None 指定で省略可能。
    """
    entry: dict = {
        "frame_idx": frame_idx,
        "t_sec": frame_idx / 60.0,
        "p1_state": p1_state,
        "p2_state": p2_state,
        "p1_confirmed": None,
        "p2_confirmed": None,
    }
    if p1_erasure_alerts is not None:
        entry["p1_erasure_alerts"] = p1_erasure_alerts
    if p2_erasure_alerts is not None:
        entry["p2_erasure_alerts"] = p2_erasure_alerts
    return entry


# ============================
# テストクラス
# ============================

class TestBoardLogErasureDump:
    """board_log JSONL の erasure_alerts dump / load 統合テスト。"""

    def test_no_alerts_p_to_e_count_is_zero(self, tmp_path: Path) -> None:
        """erasure_alerts が空の board_log → p_to_e_count = 0。"""
        log_path = tmp_path / "board_log.jsonl"
        _write_board_log(log_path, [
            _make_entry(0, p1_erasure_alerts=[], p2_erasure_alerts=[]),
            _make_entry(1, p1_erasure_alerts=[], p2_erasure_alerts=[]),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        report = ev.generate_report()
        assert report["p_to_e_count"] == 0

    def test_p1_alerts_counted_correctly(self, tmp_path: Path) -> None:
        """1P 側に erasure_alerts が 2 件 → p_to_e_count = 2。"""
        log_path = tmp_path / "board_log.jsonl"
        _write_board_log(log_path, [
            _make_entry(
                0,
                p1_erasure_alerts=[[5, 2], [6, 3]],
                p2_erasure_alerts=[],
            ),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        report = ev.generate_report()
        assert report["p_to_e_count"] == 2
        assert report["p_to_e_detail"]["p1"] == 2
        assert report["p_to_e_detail"]["p2"] == 0

    def test_p2_alerts_counted_correctly(self, tmp_path: Path) -> None:
        """2P 側に erasure_alerts が 1 件 → p_to_e_count = 1。"""
        log_path = tmp_path / "board_log.jsonl"
        _write_board_log(log_path, [
            _make_entry(
                0,
                p1_erasure_alerts=[],
                p2_erasure_alerts=[[4, 1]],
            ),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        report = ev.generate_report()
        assert report["p_to_e_count"] == 1
        assert report["p_to_e_detail"]["p1"] == 0
        assert report["p_to_e_detail"]["p2"] == 1

    def test_multiple_frames_alerts_summed(self, tmp_path: Path) -> None:
        """複数 frame の alerts が累計される。
        frame 0: 1P=2件, 2P=0件
        frame 1: 1P=0件, 2P=1件
        → total = 3
        """
        log_path = tmp_path / "board_log.jsonl"
        _write_board_log(log_path, [
            _make_entry(0, p1_erasure_alerts=[[5, 2], [6, 3]], p2_erasure_alerts=[]),
            _make_entry(1, p1_erasure_alerts=[], p2_erasure_alerts=[[4, 1]]),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        report = ev.generate_report()
        assert report["p_to_e_count"] == 3
        assert report["p_to_e_detail"]["total"] == 3

    def test_backward_compat_no_erasure_alerts_key(self, tmp_path: Path) -> None:
        """古い board_log (erasure_alerts キーなし) で p_to_e_count = 0 になる。
        backwards compat 確認: キーなし → デフォルト空リスト で処理継続。
        """
        log_path = tmp_path / "board_log.jsonl"
        # erasure_alerts キーを意図的に省略した古い形式
        _write_board_log(log_path, [
            _make_entry(0),  # p1/p2_erasure_alerts は None → キーなし
            _make_entry(1),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        report = ev.generate_report()
        assert report["p_to_e_count"] == 0

    def test_p_to_e_detail_keys_present(self, tmp_path: Path) -> None:
        """generate_report() の p_to_e_detail に p1/p2/total キーが存在する。"""
        log_path = tmp_path / "board_log.jsonl"
        _write_board_log(log_path, [
            _make_entry(0, p1_erasure_alerts=[[0, 0]], p2_erasure_alerts=[]),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        report = ev.generate_report()
        detail = report.get("p_to_e_detail", {})
        assert "p1" in detail
        assert "p2" in detail
        assert "total" in detail
        assert detail["total"] == detail["p1"] + detail["p2"]

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        """JSONL に不正行が混入しても残りの行は正常に処理される。"""
        log_path = tmp_path / "board_log.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_make_entry(
                0, p1_erasure_alerts=[[5, 2]], p2_erasure_alerts=[]
            )) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(_make_entry(
                1, p1_erasure_alerts=[], p2_erasure_alerts=[[4, 1]]
            )) + "\n")
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        # 正常 2 行分の alerts のみカウント
        report = ev.generate_report()
        assert report["p_to_e_count"] == 2

    def test_non_stable_frames_alerts_still_counted(self, tmp_path: Path) -> None:
        """CHAIN/TSUMO_FALL state の frame の alerts も累計対象になる。
        (= alerts 集計は state に依存しない; dump 側が state を問わず書くため)
        """
        log_path = tmp_path / "board_log.jsonl"
        _write_board_log(log_path, [
            _make_entry(
                0, p1_state="chain", p2_state="chain",
                p1_erasure_alerts=[[5, 2]], p2_erasure_alerts=[],
            ),
        ])
        ev = RecognitionEvaluator()
        ev.load_jsonl(log_path)
        counts = ev.count_erasure_alerts()
        assert counts["p1"] == 1
        assert counts["total"] == 1
