"""生成CLIの記録ルート指定と封印出力を合成記録で確認する。"""
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from scripts import b3_generate_rows as cli
from tests.b3_synth import build_record
from src.b3_timing import TimingTable


def test_cli_record_root_and_sealed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """既定と明示ルートは同じ特徴を作り、封印時は教師詳細を分離する。"""
    root = tmp_path / "records"
    build_record(root / "attempts" / "synth" / "a1")
    (root / "receipts").mkdir()
    (root / "receipts" / "synth.json").write_text(
        json.dumps({"status": "complete", "attempt": "a1"}), encoding="utf-8")
    monkeypatch.setattr(cli, "B2_ROOT_DEFAULT", root)
    outputs = []
    for name, extra in (("default", []), ("explicit", ["--b2-root", str(root)]),
                        ("sealed", ["--b2-root", str(root), "--seal-labels"])):
        out = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["b3_generate_rows", "--video", "synth", "--out", str(out), *extra])
        cli.main()
        assert (out / "synth.done").exists()
        outputs.append(pd.read_parquet(out / "synth.parquet"))
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    pd.testing.assert_frame_equal(outputs[0].drop(columns="won_p1"), outputs[2])
    manifest = json.loads((tmp_path / "sealed" / "synth.manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_death_audit_passed"]
    assert "winners" not in manifest and "label_death_audit" not in manifest
    assert "stable_death" not in manifest["frame_info"]
    assert "won_p1" not in manifest["missing"]["nan_rate"]
    assert (tmp_path / "sealed" / "synth.sealed_label.parquet").exists()
    assert (tmp_path / "sealed" / "synth.sealed_label.json").exists()


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("sealed", [False, True])
def test_cli_allow_unknown_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  unknown: bool, sealed: bool) -> None:
    """任意指定だけで未確定行を追加し、交換複製と教師隔離を維持する。"""
    root = tmp_path / "records"
    record = build_record(root / "attempts" / "synth" / "a1")
    if unknown:
        with np.load(record / "lean.npz") as source:
            arrays = {k: source[k] for k in source.files}
        arrays["won"][:] = np.nan
        np.savez(record / "lean.npz", **arrays)
    (root / "receipts").mkdir()
    (root / "receipts" / "synth.json").write_text(
        json.dumps({"status": "complete", "attempt": "a1"}), encoding="utf-8")
    outputs = []
    for allow in (False, True):
        out = tmp_path / str(allow)
        args = ["cli", "--video", "synth", "--out", str(out), "--b2-root", str(root)]
        args += ["--allow-unknown-winner"] if allow else []
        args += ["--seal-labels"] if sealed else []
        monkeypatch.setattr(sys, "argv", args)
        cli.main()
        outputs.append(pd.read_parquet(out / "synth.parquet"))
    old, new = outputs
    assert "winner_unknown" not in old and len(new) > 0
    assert new.winner_unknown.eq(int(unknown)).all()
    assert set(new.orientation) == {0, 1}
    assert new.weight.sum() == pytest.approx(1.0)
    if unknown:
        assert old.empty
        labels = pd.read_parquet(out / "synth.sealed_label.parquet") if sealed else new
        assert labels.won_p1.isna().all()  # 合成教師だけを確認する。
    else:
        pd.testing.assert_frame_equal(old, new.drop(columns="winner_unknown"), check_exact=True)
    manifest = json.loads((out / "synth.manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_columns"] == len(new.columns)
    audit_path = out / "synth.sealed_label.json" if sealed else out / "synth.manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))["label_death_audit"]
    assert audit["total_games"] == 1 and audit["excluded_unknown_winner_games"] == int(unknown)
    assert audit["games"] == int(not unknown)
    if sealed:
        assert "won_p1" not in new and "label_death_audit" not in manifest


def test_unknown_audit_excludes_dead_game() -> None:
    """窒息が判定可能でも未確定教師は監査母数に入れない。"""
    rows = [{"game_idx": g, "frame_idx": 1, "won_p1": w} for g, w in ((1, 1.0), (2, np.nan))]
    deaths = {str(g): {"p1": {"dead": 0}, "p2": {"dead": 1}} for g in (1, 2)}
    assert cli.evaluation_audit(rows, deaths, True) == {
        "games": 1, "judged": 1, "mismatch": 0, "undetermined": 0,
        "total_games": 2, "excluded_unknown_winner_games": 1}


def test_unknown_pass_preserves_known_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知試合の前後を含め、既知行とその生成条件をそのまま維持する。"""
    calls = []

    def fake_generate(*args: object, **kwargs: object) -> tuple[list[dict], dict]:
        calls.append(kwargs)
        games = (1, 3) if kwargs["require_winner"] else (2,)
        rows = [{"game_idx": g, "frame_idx": g, "won_p1": np.nan if g == 2 else 1.0} for g in games]
        return rows, {"rows": len(rows), "skipped_no_winner_frames": 10, "leak_violations": 0,
                      "stable_death": {}, "excluded_outside_match_rows": 2,
                      "excluded_outside_match_rows_by_game": {"1": 1, "2": 2, "3": 1}}

    monkeypatch.setattr(cli, "generate_rows", fake_generate)
    mapping = {g: {"start_sec": g, "end_sec": g + 1} for g in (1, 2, 3)}
    rows, info = cli.evaluation_rows(Path("unused"), "synth", TimingTable(),
                                     {1: 1, 2: None, 3: 1}, mapping, allow_unknown_winner=True)
    assert [r["game_idx"] for r in rows] == [1, 2, 3]
    assert calls[0]["require_winner"] and calls[0]["match_mapping"] == mapping
    assert not calls[1]["require_winner"] and calls[1]["match_mapping"] == {2: mapping[2]}
    assert info["rows"] == 3 and info["excluded_outside_match_rows"] == 4
