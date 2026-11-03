"""E16の根因・同期追従・死亡後発火・同一CV母集団の証跡を集約する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.train_exchange_event_models_v3_20260928 import OUT as E15_TRAIN
from scripts.run_e16_replay_20260928 import ALL_SOURCES, directory
from src.exchange_event_landing import logit_mean

OUT = Path("logs/e16")
TIMES = (2612.883333333333, 2613.216666666667, 2613.516666666667, 2613.816666666667)
FIRE_START, FIRE_END, DEATH_START = 2699.4, 2699.6, 2698.5
IDENTITY = ["row_id", "video_id", "game", "exchange", "t_sec", "weight", "label", "sign"]


def read(path: Path) -> object:
    """UTF-8の保存値をそのまま読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def alignment() -> dict:
    """表示と独立に、保持解除と各側のcount実値を確認する。"""
    before = read(Path("logs/e15/on/review/count_trace.json"))
    after = read(OUT/"on/review/count_trace.json")
    samples = []
    for stamp in TIMES:
        row = dict(t_sec=stamp)
        for name, data in (("E15", before), ("E16", after)):
            trace = min(data, key=lambda r: abs(r["t_sec"]-stamp))
            counts = np.array(trace["counts"])
            margin = np.sign(counts[:, -1])*np.expm1(abs(counts[:, -1]))
            row[name] = dict(p1=trace["p1"], nf=np.expm1(counts[:, :-1]).tolist(), margin=margin.tolist())
        samples.append(row)
    with (OUT/"sync_trace.csv").open(encoding="utf-8") as stream:
        updates = [r for r in csv.DictReader(stream) if r["side"] == "1" and r["updated"] == "True"]
    return dict(counterfactual=read(OUT/"pair_counterfactual.json"), samples=samples, accepted_updates=updates)


def death_activation() -> dict:
    """A3の同じ発火通知が新たな参加連鎖として受理されたかを数える。"""
    counts = {}
    for name, root in (("E15", Path("logs/e15/on/review")), ("E16", OUT/"on/review")):
        records = [json.loads(s) for s in (root/"events.jsonl").read_text().splitlines()]
        chains = [c for r in records for c in r["chains"]
                  if c["side"] == "2P" and FIRE_START <= c["trigger_sec"] <= FIRE_END]
        counts[name] = [dict(trigger=c["trigger_sec"], predicted_score=c["predicted_final_score"],
                            predicted_count=c["predicted_chain_count"]) for c in chains]
    observations = read(OUT/"records/review.json")["detections"]
    first = next(r for r in observations if DEATH_START <= r["t_sec"] <= FIRE_END and "2P" in r["sides"])
    counts["first_confirmed_death"] = first
    return counts


def cv_identity() -> dict:
    """行・重み・動画単位foldがE15と一致することを全OOF行で検査する。"""
    assert (OUT/"train/rows.csv").read_bytes() == (E15_TRAIN/"rows.csv").read_bytes()
    paths = sorted((OUT/"train").glob("seed_*/fold_*/predictions.csv"))
    count = 0
    for path in paths:
        reference = E15_TRAIN/path.relative_to(OUT/"train")
        current, old = pd.read_csv(path), pd.read_csv(reference)
        pd.testing.assert_frame_equal(current[IDENTITY], old[IDENTITY], check_exact=True)
        assert not current.duplicated("row_id").any()
        count += len(current)
    return dict(identical=True, folds=len(paths), predictions=count, cohort=read(OUT/"train/COHORT.json"))


def layer_integrity() -> dict:
    """全保存フレームの合成値・現在層復帰が記録した理由と一致するか検査する。"""
    count, fallback, blended = 0, 0, 0
    for source in ALL_SOURCES:
        for line in (directory(source)/"events.layers.jsonl").read_text().splitlines():
            row = json.loads(line)
            current, prediction = row["p1_current"], row["p1_prediction"]
            expected = current
            if current is not None and prediction is not None and not row["prediction_reasons"]:
                expected = logit_mean(current, prediction)
                blended += 1
            elif current is not None and row["prediction_reasons"]:
                fallback += 1
            assert row["p1_layer_combined"] == expected
            count += 1
    return dict(checked=count, blended=blended, fallback_with_current=fallback, mismatches=0)


def main() -> None:
    """固定の合否結果とは別に根因の数値を保存し、METRICSへも参照を加える。"""
    result = dict(synchronization=alignment(), A3=death_activation(), cv_identity=cv_identity(),
        layers=layer_integrity(),
        landing=dict(response_used_by_gfe=False, input="仮想着弾盤面のD46、M0logit、埋まり/時間位相と交互作用",
            references=["src/exchange_event_landing.py:288", "src/exchange_event_landing.py:305",
                        "src/exchange_event_landing.py:320", "scripts/visualize_advantage_overlay.py:6330"],
            unchanged=True))
    save_json(OUT/"DIAGNOSTICS.json", result)
    metrics = read(OUT/"METRICS.json")
    metrics["diagnostics"] = result
    metrics["verification"] = dict(
        tests=(OUT/"final_tests.log").read_text().strip().splitlines()[-1],
        off_equivalence=json.loads((OUT/"off_equivalence.log").read_text().strip().splitlines()[-1])["equivalence"],
        code=read(OUT/"CODE_AUDIT.json"))
    save_json(OUT/"METRICS.json", metrics)
    print(json.dumps(dict(A3=result["A3"], cv=result["cv_identity"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
