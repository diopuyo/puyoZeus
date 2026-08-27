"""Gate 4 WIN★勝者根拠抽出器の回帰テスト。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts" / "_extract_gate4_panel_truth_2026-08-26.py")
SPEC = importlib.util.spec_from_file_location("gate4_panel_truth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def test_load_segment_starts_uses_each_game_transition(tmp_path: Path) -> None:
    path = tmp_path / "seg03_display.npz"
    np.savez_compressed(
        path, t_sec=np.asarray([10.0, 10.1, 10.2, 10.3]),
        game_idx=np.asarray([0, 0, 1, 2]))
    starts = extractor._load_segment_starts(path)
    assert [(x.segment, x.game_idx, x.start_sec) for x in starts] == [
        ("seg03", 0, 10.0), ("seg03", 1, 10.2), ("seg03", 2, 10.3)]


def test_manual_override_replaces_only_named_game(tmp_path: Path) -> None:
    rows = [
        {"segment": "seg04", "game_idx": 17, "winner": "UNKNOWN",
         "source": "win_panel_digit_delta"},
        {"segment": "seg04", "game_idx": 16, "winner": "1P",
         "source": "win_panel_digit_delta"},
    ]
    path = tmp_path / "overrides.tsv"
    path.write_text(
        "segment\tgame_idx\twinner\nseg04\t17\t2P\n", encoding="utf-8")
    extractor._apply_overrides(rows, path)
    assert rows[0]["winner"] == "2P"
    assert rows[0]["source"] == "win_panel_manual_review"
    assert rows[1]["winner"] == "1P"
