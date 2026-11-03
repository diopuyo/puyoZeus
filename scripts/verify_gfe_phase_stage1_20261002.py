"""段1成果物の行対応・設定・機械判定を学習なしで独立確認する。"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from scripts.gfe_phase_stage1_20261002 import (
    FOLDS, HGB_PARAMS, LR_PARAMS, OUTPUT, PHASES, SEEDS, STAGE_LIMIT,
    VARIANTS, column_names, save_json,
)
import numpy as np
import pandas as pd

ROUNDTRIP_ATOL = 1e-15


def verify_folds(frame: pd.DataFrame, report: dict[str, Any]) -> dict[str, Any]:
    """V0を完全保存し、全候補が同じ行・fold・パラメータであることを確認する。"""
    maximum = 0.0
    for seed in SEEDS:
        seed_rows = frame[frame.seed == seed]
        np.testing.assert_array_equal(np.sort(seed_rows.row_id), np.arange(report["rows"]))
        for fold in FOLDS:
            part = seed_rows[seed_rows.fold == fold]
            path = OUTPUT / f"V0_seed_{seed}_fold_{fold}.csv"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == report["baseline_sha256"][path.name]
            original = pd.read_csv(path)
            np.testing.assert_array_equal(part.row_id, original.row_id)
            np.testing.assert_array_equal(part.label, original.label)
            np.testing.assert_array_equal(part.video_id, original.video_id)
            error = float(np.max(np.abs(part.V0.to_numpy()-original.V0.to_numpy())))
            maximum = max(maximum, error)
            assert error <= ROUNDTRIP_ATOL
            for variant in VARIANTS[1:]:
                details = json.loads((OUTPUT / f"{variant}_seed_{seed}_fold_{fold}.json").read_text())
                assert details["columns"] == column_names(variant)
                expected = HGB_PARAMS if variant == "V3" else LR_PARAMS
                assert all(details["parameters"][k] == v for k, v in expected.items())
    return dict(baseline_hashes_unchanged=True, baseline_roundtrip_max_error=maximum,
                candidate_folds=len(SEEDS)*len(FOLDS)*len(VARIANTS[1:]))


def verify_decisions(report: dict[str, Any]) -> None:
    """Uの3段階を合成したLLと、閾値の不等号を独立照合する。"""
    scopes = report["scopes"]
    assert sum(scopes[f"U_{name}"]["rows"] for name in PHASES) == report["rows"]
    for variant in VARIANTS:
        weighted = sum(scopes[f"U_{name}"]["rows"] * scopes[f"U_{name}"]["metrics"][variant]["log_loss"]
                       for name in PHASES) / report["rows"]
        assert abs(weighted-scopes["overall"]["metrics"][variant]["log_loss"]) <= ROUNDTRIP_ATOL
        if variant == "V0":
            continue
        upper = scopes["overall"]["differences"][variant]["ci95"][1]
        means = [scopes[f"U_{name}"]["differences"][variant]["delta"] for name in PHASES]
        assert report["decisions"][variant]["passed"] == (upper < 0 and max(means) <= STAGE_LIMIT)


def main() -> None:
    """完了済み成果物だけを検査し、検査時間と件数を保存する。"""
    started = time.monotonic()
    report = json.loads((OUTPUT / "METRICS.json").read_text())
    assert report["status"] == "COMPLETE"
    frame = pd.read_csv(OUTPUT / "predictions.csv")
    assert len(frame) == len(SEEDS)*report["rows"]
    assert int(frame.video_id.nunique()) == report["videos"]
    assert int(frame.game_id.nunique()) == report["games"]
    assert np.isfinite(frame[list(VARIANTS)].to_numpy()).all()
    assert ((frame[list(VARIANTS)] >= 0) & (frame[list(VARIANTS)] <= 1)).all().all()
    with np.load(OUTPUT / "user_phase.npz") as saved:
        np.testing.assert_array_equal(frame.U, saved["phase"][frame.row_id.to_numpy()])
    results = verify_folds(frame, report)
    verify_decisions(report)
    results.update(passed=True, predictions=len(frame), seconds=time.monotonic()-started)
    save_json(OUTPUT / "VERIFICATION.json", results)
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
