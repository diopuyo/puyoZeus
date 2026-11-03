"""段1の結果を見た後に追加するR1/R2参考比較。合否には使わない。"""
from __future__ import annotations

import json
import os
import resource
import time
from typing import Any

from scripts import gfe_phase_stage1_20261002 as stage
from scripts.gfe_phase_stage1_candidates_20261002 import fold_inputs
from scripts import gfe_phase_stage1_report_20261002 as report
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = stage.ROOT / "logs/gfe_phase_reference_20261002"
VARIANTS = ("V0", "R1", "R2")


def columns(variant: str) -> list[str]:
    """R1はV0の59列、R2はV3の54列の末尾に経過秒6列を追加する。"""
    if variant == "R1":
        return stage.column_names("V0")
    return stage.column_names("V3") + [n for n in stage.G_COLUMNS if "elapsed" in n]


def features(variant: str, design: np.ndarray, probability: np.ndarray,
             sign: np.ndarray, phase: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """既存変換を共有し、列順以外の加工を加えない。"""
    base = stage.features("V0", design, probability, sign, phase, missing)
    if variant == "R1":
        return base
    extra = base[:, [i for i, n in enumerate(stage.G_COLUMNS) if "elapsed" in n]]
    return np.column_stack((stage.features("V3", design, probability, sign, phase, missing), extra))


def train(variant: str, rows: pd.DataFrame, design: np.ndarray, phases: dict,
          u: np.ndarray, missing: np.ndarray, seed: int, fold: int) -> pd.DataFrame:
    """段1と同じ内側OOF・外側予測を読み、固定HistGBCで学習する。"""
    stage.memory_guard()
    started = time.monotonic()
    train_ids, test, oof, outer, elapsed = fold_inputs(rows, phases["elapsed"], seed, fold)
    phase = np.column_stack((phases["fill"] if variant == "R1" else u, elapsed))
    sign, labels = rows.sign.to_numpy(), rows.label.to_numpy()
    x = features(variant, design[train_ids], oof[train_ids], sign[train_ids],
                 phase[train_ids], missing[train_ids])
    model = HistGradientBoostingClassifier(**stage.HGB_PARAMS)
    model.fit(x, np.where(sign[train_ids] > 0, labels[train_ids], 1-labels[train_ids]))
    del x
    x = features(variant, design[test], outer, sign[test], phase[test], missing[test])
    p = model.predict_proba(x)[:, 1]
    result = rows.iloc[test][["row_id", "video_id", "label", "phase"]].copy()
    result[variant] = np.where(sign[test] > 0, p, 1-p)
    result["seed"], result["fold"] = seed, fold
    path = OUT / f"{variant}_seed_{seed}_fold_{fold}"
    result.to_csv(path.with_suffix(".csv"), index=False)
    details = dict(columns=columns(variant), parameters=model.get_params(),
                   seconds=time.monotonic()-started)
    stage.save_json(path.with_suffix(".json"), details)
    print(f"{variant} seed={seed} fold={fold} seconds={details['seconds']:.2f}", flush=True)
    return result


def summarize(combined: pd.DataFrame, started: float) -> None:
    """段1と同じ採点表だけを出力し、選別判定は作らない。"""
    report.VARIANTS = VARIANTS
    scopes = {"overall": report.scope_report(combined)}
    for key in ("U", "current_fill", "phase"):
        for i, name in enumerate(stage.PHASES):
            scopes[f"{key}_{name}"] = report.scope_report(combined[combined[key] == i])
    result: dict[str, Any] = dict(status="COMPLETE", purpose="参考・結果を見た後に追加",
        scopes=scopes, elapsed_seconds=time.monotonic()-started, decisions=None)
    stage.save_json(OUT / "METRICS.json", result)
    lines = ["# G_fe参考比較", "", "参考・結果を見た後に追加。合否には使わない。", "",
             "R1=V0と同じ59列、R2=V3の54列+経過秒3列+m0×経過秒3列。",
             "HistGBC設定・行・fold・M0予測・bootstrapは段1と同じ。", "",
             "|区分|行/試合/動画|構成|LL|AUC|ECE|ΔLL|95%区間|",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for scope, values in scopes.items():
        for variant, m in values["metrics"].items():
            d = values["differences"].get(variant)
            diff = f"{d['delta']:+.9f}|[{d['ci95'][0]:+.9f}, {d['ci95'][1]:+.9f}]" if d else "—|—"
            lines.append(f"|{scope}|{values['rows']}/{values['games']}/{values['videos']}|{variant}|"
                         f"{m['log_loss']:.9f}|{m['auc']:.9f}|{m['ece']:.9f}|{diff}|")
    lines += ["", f"所要時間: {result['elapsed_seconds']:.2f}秒。",
              "", "V3入力の生成元監査: LEAK_AUDIT.mdを参照。"]
    (OUT / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    """nice19・メモリ上限5GiBで参考比較を順番に実行する。"""
    os.nice(max(0, 19-os.nice(0)))
    resource.setrlimit(resource.RLIMIT_AS, (stage.MAX_MEMORY, stage.MAX_MEMORY))
    stage.memory_guard()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    stage.save_json(OUT / "PROTOCOL.json", dict(purpose="参考・結果を見た後に追加",
        columns={v: columns(v) for v in VARIANTS[1:]}, parameters=stage.HGB_PARAMS))
    rows, design, phases = stage.datasets(stage.REFERENCE)
    with np.load(stage.OUTPUT / "user_phase.npz") as saved:
        u, missing = saved["phase"], saved["missing"]
    combined = pd.read_csv(stage.OUTPUT / "predictions.csv").drop(columns=list(stage.VARIANTS[1:]))
    for variant in VARIANTS[1:]:
        parts = [train(variant, rows, design, phases, u, missing, seed, fold)
                 for seed in stage.SEEDS for fold in stage.FOLDS]
        candidate = pd.concat(parts, ignore_index=True)
        keys = ["row_id", "video_id", "label", "phase", "seed", "fold"]
        np.testing.assert_array_equal(combined[keys], candidate[keys])
        combined[variant] = candidate[variant].to_numpy()
    combined.to_csv(OUT / "predictions.csv", index=False)
    summarize(combined, started)


if __name__ == "__main__":
    main()
