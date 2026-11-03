"""段2実行前に入力存在・採点関数・モデル・資源条件を確認する。"""
from __future__ import annotations

import ast
import gzip
import json
import os

from scripts.gfe_phase_stage2_replay_20261002 import (
    OUT, SOURCES, record_path, baseline, EVALSET, PREFIRE,
)
from scripts.gfe_phase_stage2_score_20261002 import metric_functions
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json, ROOT, SHARED, REFERENCE, run_path
from src.exchange_event_evaluator import FileExchangeModels, file_sha256
import numpy as np
import pandas as pd


def evaluation_separation() -> dict:
    """別名video番号を元の動画IDへ戻し、評価4動画との混入が無いことを確認する。"""
    names = pd.read_csv(run_path(REFERENCE, "base")/"rows.csv", usecols=["video_id"]).video_id.unique()
    tiers = pd.read_csv(SHARED/"data/video_tier_index_2026-08-07.tsv", sep="\t", dtype=str).set_index("video_name")
    selected = tiers.loc[names, "video_id"]
    assert selected.notna().all()
    training = set(selected)
    evaluation = {}
    for source in SOURCES:
        with gzip.open(record_path(source), "rt") as stream:
            header = json.loads(next(stream))["video_id"]
            evaluation[source] = header.split("_first_", 1)[0] if "_first_" in header else header.removeprefix("video_zenchi_")
            assert len(evaluation[source]) == 11, header
    overlap = training & set(evaluation.values())
    assert not overlap, overlap
    return dict(training_videos=len(names), evaluation_videos=len(set(evaluation.values())),
                evaluation_ids=evaluation, overlapping_ids=sorted(overlap))


def main() -> None:
    """他worktreeには書かず、前提が欠けていれば本処理前に停止する。"""
    os.nice(max(0, 19-os.nice(0)))
    memory_guard()
    assets = {}
    for source in SOURCES:
        for path in (record_path(source), baseline(source)/"display.npz", baseline(source)/"events.jsonl"):
            assert path.is_file(), path
            assets[str(path)] = dict(bytes=path.stat().st_size)
    for relative, count in (("labels.json", 57), ("set2/labels.json", 58)):
        data = json.loads((EVALSET/relative).read_text())
        assert len(data["games"]) == count
    metric = metric_functions()
    test = metric.paired(np.array([.4, .5]), np.array([.5, .6]))
    assert np.allclose(test["ci95"], [-.1, -.1])
    model = FileExchangeModels.load(ROOT/"models/exchange_event_v3")
    assert model.count_features and not model.user_phase
    names = ("_compute_row", "_asof_attach_opponent", "_attach_opponent_diff_columns",
             "_add_pair_interaction_columns", "_attach_ojama_truth_own_columns",
             "_attach_ojama_margin_column", "_attach_ojama_net_balance_synced_column",
             "_all_clear_state_for_group", "_apply_all_clear_truth")
    trees = [ast.parse((root/"scripts/build_labeled_win_from_npz.py").read_text()) for root in (ROOT, SHARED)]
    functions = [{n.name: ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef)} for t in trees]
    source_equal = {name: functions[0][name] == functions[1][name] for name in names}
    assert all(source_equal.values()), source_equal
    result = dict(passed=True, inputs=assets, sources=len(SOURCES), source_functions_equal=source_equal,
        evaluation_separation=evaluation_separation(),
        source_sha256={str(root): file_sha256(root/"scripts/build_labeled_win_from_npz.py") for root in (ROOT, SHARED)},
        original_model_sha256={str(p.relative_to(ROOT)): file_sha256(p)
                               for p in (ROOT/"models/exchange_event_v1").rglob("*") if p.is_file()},
        production_config_sha256=file_sha256(ROOT/"src/production_config.py"))
    save_json(OUT/"PREFLIGHT.json", result)
    print(json.dumps(dict(passed=True, sources=len(SOURCES), source_functions_equal=source_equal)), flush=True)


if __name__ == "__main__":
    main()
