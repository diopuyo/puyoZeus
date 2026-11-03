"""段2のモデル・母数・既定OFF・時間・変更禁止資産を独立検証する。"""
from __future__ import annotations

import json
import os
import time

from scripts.gfe_phase_stage2_replay_20261002 import OUT, SOURCES, baseline
from scripts.gfe_phase_stage2_score_20261002 import root, MODES, judge, metric_functions, latency
from scripts.gfe_phase_stage2_train_20261002 import model_directory
from scripts.gfe_phase_stage1_20261002 import ROOT, HGB_PARAMS, LR_PARAMS, save_json
from src.exchange_event_evaluator import FileExchangeModels, file_sha256


def unchanged() -> dict:
    """既存v1の全9ファイルと本番設定のハッシュが実行前と一致する。"""
    preflight = json.loads((OUT/"PREFLIGHT.json").read_text())
    for path, expected in preflight["original_model_sha256"].items():
        assert file_sha256(ROOT/path) == expected, path
    assert file_sha256(ROOT/"src/production_config.py") == preflight["production_config_sha256"]
    assert not preflight["evaluation_separation"]["overlapping_ids"]
    return dict(original_models=len(preflight["original_model_sha256"]), production_config=True)


def models() -> dict:
    """最終モデルの保存形式・列順・パラメータ・学習母数を照合する。"""
    result = {}
    for mode in MODES[1:]:
        directory = model_directory(mode)
        loaded = FileExchangeModels.load(ROOT/"models/exchange_event_v3", gfe_directory=directory)
        meta = json.loads((directory/"manifest.json").read_text())
        assert (meta["rows"], meta["games"], meta["videos"]) == (1035639, 7686, 144)
        expected = HGB_PARAMS if mode == "V3" else LR_PARAMS
        assert all(loaded.estimators["G_fe"].get_params()[k] == v for k, v in expected.items())
        assert loaded.user_phase and meta["adopted"] is False
        result[mode] = dict(estimator=type(loaded.estimators["G_fe"]).__name__, rows=meta["rows"])
    return result


def replays() -> dict:
    """全54再生の完了票と18本の本番バイト一致を確認する。"""
    frames = {}
    for source in SOURCES:
        for mode in MODES:
            directory = root(mode, source)
            meta = json.loads((directory/"DONE.json").read_text())
            assert meta["state"] == "completed"
            frames[f"{mode}/{source}"] = meta["display_frames"]
            assert all(p["num_threads"] == 1 for p in meta["threadpools"] if p["user_api"] == "blas")
        for name in ("display.npz", "events.jsonl"):
            assert (root("baseline", source)/name).read_bytes() == (baseline(source)/name).read_bytes()
    return dict(replays=len(frames), byte_identical_baselines=len(SOURCES), frames=frames)


def main() -> None:
    """保存票の判定と実時間配列から再検算した判定が一致することを確認する。"""
    os.nice(max(0, 19-os.nice(0)))
    started = time.monotonic()
    result = dict(unchanged=unchanged(), models=models(), replays=replays())
    report = json.loads((OUT/"METRICS.json").read_text())
    speeds = {m: latency(m) for m in MODES}
    assert speeds == report["latency"]
    expected = judge([report["set1"], report["set2"]], report["regression"], speeds, metric_functions())
    assert expected == report["decisions"]
    result.update(passed=True, seconds=time.monotonic()-started)
    save_json(OUT/"VERIFICATION.json", result)
    timing = json.loads((OUT/"TIMING.json").read_text())
    report["timing"] = dict(replay_and_score_seconds=timing["seconds"],
        training_seconds=sum(json.loads((OUT/f"TRAIN_{m}.json").read_text())["seconds"] for m in MODES[1:]))
    report["verification"] = dict(passed=True, seconds=result["seconds"])
    save_json(OUT/"METRICS.json", report)
    from scripts.gfe_phase_stage2_report_20261002 import write_summary
    write_summary(report)
    print(json.dumps(dict(passed=True, seconds=result["seconds"], replays=result["replays"]["replays"])), flush=True)


if __name__ == "__main__":
    main()
