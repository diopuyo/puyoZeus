"""段1 実行器の CLI (合成 parquet): plan のみ・部分実行 → score-only の結合・前提検査違反で停止。"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.b3_train_synth import make_rows

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path, n_videos=10):
    df, _ = make_rows(n_videos=n_videos, games_per_video=3, rows_per_game=24)
    rows = tmp_path / "rows"
    rows.mkdir()
    targets = []
    for vid, sub in df.groupby("video_id"):
        sub.reset_index(drop=True).to_parquet(rows / f"{vid}.parquet", index=False)
        (rows / f"{vid}.done").write_text("done\n")
        targets.append({"video_id": vid, "youtube_id": "yt" + vid[-2:]})
    (tmp_path / "targets.json").write_text(json.dumps(targets), encoding="utf-8")
    return rows


def _run(args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-m", "scripts.b3_train_stage1", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=900)


def test_plan_only_does_not_train_and_writes_assignment(tmp_path):
    rows = _setup(tmp_path)
    out = tmp_path / "out"
    r = _run(["--rows-dir", str(rows), "--targets", str(tmp_path / "targets.json"), "--out", str(out), "--no-min-groups"])
    assert r.returncode == 0, r.stderr[-800:]
    assert (out / "assignment.tsv").exists() and (out / "precheck.json").exists()
    assert not (out / "RECEIPT.json").exists() and not (out / "parts").exists()
    pre = json.loads((out / "precheck.json").read_text(encoding="utf-8"))
    assert pre["violations_total"] == 0 and pre["groups"] == 10 and len(pre["assignment_sha256"]) == 64


def test_min_groups_enforced_by_default(tmp_path):
    rows = _setup(tmp_path)
    out = tmp_path / "out"
    r = _run(["--rows-dir", str(rows), "--targets", str(tmp_path / "targets.json"), "--out", str(out), "--run",
              "--grid", "smoke"])
    assert r.returncode != 0 and "前提検査" in (r.stdout + r.stderr)          # 10 group < 50 → 停止
    assert not (out / "RECEIPT.json").exists()


@pytest.mark.parametrize("monotonic", ["0", "1"])
def test_partial_folds_then_score_only(tmp_path, monkeypatch, monotonic: str) -> None:
    from src.b3_train_model import F9_MONOTONIC
    monkeypatch.setenv("B3_MONOTONIC", monotonic)
    rows = _setup(tmp_path)
    out = tmp_path / "out"
    base = ["--rows-dir", str(rows), "--targets", str(tmp_path / "targets.json"), "--out", str(out), "--no-min-groups",
            "--run", "--grid", "smoke", "--seeds", "1003"]
    for folds in (["0", "1", "2"], ["3", "4"]):
        r = _run(base + ["--folds", *folds])
        assert r.returncode == 0, r.stderr[-1500:]
        assert not (out / "RECEIPT.json").exists()                          # 部分実行は採点しない
    # 採点時の環境で制約記録を再生成せず、partsの実際の学習条件を引き継ぐ。
    monkeypatch.setenv("B3_MONOTONIC", "0")
    r = _run(base + ["--score-only"])
    assert r.returncode == 0, r.stderr[-1500:]
    rec = json.loads((out / "RECEIPT.json").read_text(encoding="utf-8"))
    expected = F9_MONOTONIC if monotonic == "1" else {}
    for info in rec["training"]["cand"]:
        assert info["monotonic_cst"] == dict(static=expected, exchange=expected)
        part = out / "parts" / f"cand_seed1003_fold{info['outer_fold']}.info.json"
        assert json.loads(part.read_text())[0]["monotonic_cst"] == info["monotonic_cst"]
    assert rec["verdict"] in ("FAIL", "HOLD") and "leak_prechecks" in rec["not_evaluated"]
    assert len(rec["training"]["cand"]) == 5 and len(rec["training"]["ref"]) == 5
    assert all(i["hgb_threads"] == 1 and i["blas_threads"] == 1 for i in rec["training"]["cand"])
    assert all("inner_ll_gap" in i for i in rec["training"]["cand"])
    sc = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    assert sc["overall"]["population"]["games"] == 30
    oof = pd.read_parquet(out / "oof_cand.parquet")
    assert np.isfinite(oof["p_1003"]).all() and len(oof) == 30 * 24 * 2


def test_score_only_refuses_missing_part(tmp_path):
    rows = _setup(tmp_path)
    out = tmp_path / "out"
    base = ["--rows-dir", str(rows), "--targets", str(tmp_path / "targets.json"), "--out", str(out), "--no-min-groups",
            "--run", "--grid", "smoke", "--seeds", "1003"]
    assert _run(base + ["--folds", "0"]).returncode == 0
    r = _run(base + ["--score-only"])
    assert r.returncode != 0 and "部分結果が欠落" in (r.stdout + r.stderr)


def test_shared_selection_separate_seed_jobs(tmp_path, monkeypatch) -> None:
    rows = _setup(tmp_path)
    out = tmp_path / "out"
    base = ["--rows-dir", str(rows), "--targets", str(tmp_path / "targets.json"), "--out", str(out),
            "--no-min-groups", "--run", "--grid", "smoke", "--folds", "0", "--share-inner-selection"]
    monkeypatch.setenv("B3_HGB_THREADS", "8")
    missing = _run(base + ["--seeds", "1004"])
    assert missing.returncode != 0 and "FileNotFoundError" in missing.stderr
    assert not list((out / "parts").glob("*.parquet"))
    for seed in (1003, 1004):
        result = _run(base + ["--seeds", str(seed)])
        assert result.returncode == 0, result.stderr[-1500:]
    parts = out / "parts"
    choice = json.loads((parts / "inner_selection_fold0.json").read_text(encoding="utf-8"))
    assert len(choice["table"]) == 2 and choice["hgb_threads"] == 8
    for seed in (1003, 1004):
        info = json.loads((parts / f"cand_seed{seed}_fold0.info.json").read_text(encoding="utf-8"))[0]
        assert info["chosen"] == choice["chosen"]["name"] and info["inner_table"] == choice["table"]
        assert info["selection_seed"] == 1003 and info["hgb_threads"] == 8
    assert not (out / "scores.json").exists()
