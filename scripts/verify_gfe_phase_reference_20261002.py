"""R1/R2の行・fold・設定・全採点表を保存原票から再検証する。"""
from __future__ import annotations

import json
import os
import time

from scripts import gfe_phase_stage1_20261002 as stage
from scripts import gfe_phase_reference_20261002 as reference
from scripts import gfe_phase_stage1_report_20261002 as report
import numpy as np
import pandas as pd

ATOL = 1e-12


def verify_scopes(frame: pd.DataFrame, saved: dict) -> None:
    """全体と3種類の段階別を原予測から再計算する。"""
    report.VARIANTS = reference.VARIANTS
    for key, expected in saved["scopes"].items():
        if key == "overall":
            part = frame
        else:
            column, phase = key.rsplit("_", 1)
            part = frame[frame[column] == stage.PHASES.index(phase)]
        actual = report.scope_report(part)
        for count in ("rows", "games", "videos", "predictions"):
            assert actual[count] == expected[count]
        for name in reference.VARIANTS:
            for metric in ("log_loss", "auc", "ece"):
                assert abs(actual["metrics"][name][metric]-expected["metrics"][name][metric]) < ATOL
            if name != "V0":
                for metric in ("delta", "ci95"):
                    np.testing.assert_allclose(actual["differences"][name][metric],
                        expected["differences"][name][metric], rtol=0, atol=ATOL)


def main() -> None:
    """参考比較から選別判定を作っていないことも確認する。"""
    os.nice(max(0, 19-os.nice(0)))
    stage.memory_guard()
    started = time.monotonic()
    frame = pd.read_csv(reference.OUT/"predictions.csv")
    old = pd.read_csv(stage.OUTPUT/"predictions.csv", usecols=[
        "row_id", "video_id", "label", "phase", "seed", "fold", "V0", "U", "current_fill", "game_id"])
    for key in old.columns:
        np.testing.assert_array_equal(frame[key], old[key])
    for variant in reference.VARIANTS[1:]:
        for seed in stage.SEEDS:
            for fold in stage.FOLDS:
                stem = reference.OUT/f"{variant}_seed_{seed}_fold_{fold}"
                meta = json.loads(stem.with_suffix(".json").read_text())
                assert meta["columns"] == reference.columns(variant)
                assert all(meta["parameters"][k] == v for k, v in stage.HGB_PARAMS.items())
                part = pd.read_csv(stem.with_suffix(".csv"))
                selected = frame[(frame.seed == seed) & (frame.fold == fold)]
                np.testing.assert_array_equal(part.row_id, selected.row_id)
                np.testing.assert_array_equal(part[variant], selected[variant])
    saved = json.loads((reference.OUT/"METRICS.json").read_text())
    assert saved["decisions"] is None and saved["purpose"] == "参考・結果を見た後に追加"
    verify_scopes(frame, saved)
    result = dict(passed=True, predictions=len(frame), folds=30, scopes=len(saved["scopes"]),
                  seconds=time.monotonic()-started)
    stage.save_json(reference.OUT/"VERIFICATION.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
