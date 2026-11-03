"""事前登録されたG_fe局面比較の段1。既存資産は読み取り専用。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import ast
import argparse
import json
import os
from pathlib import Path
import resource
import hashlib
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scripts.train_exchange_event_models_20260926 import (
    FOLDS, LR_PARAMS, REFERENCE, ROOT, SEEDS, SHARED, datasets, fit_g_fold,
    run_path, save_json,
)
from src.exchange_event_features import D_COLUMNS, G_COLUMNS, g_features
from src.exchange_event_user_phase import (
    CELLS, EARLY_MAX, LATE_MIN, USER_COLUMNS, user_phase, user_features,
)

OUTPUT = ROOT / "logs/gfe_phase_stage1_20261002"
EXEV = Path(_public_dev_text('<repo>'))
USER_DEF = EXEV / "logs/phase_definition_facts_2026-10-02/user_def"
SOURCE = SHARED / "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
PHASES = ("early", "middle", "late")
VARIANTS = ("V0", "V1", "V1b", "V2", "V3")
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=200, early_stopping=False, random_state=0)
GIB = 1024 ** 3
RESERVE, MAX_MEMORY = 6 * GIB, 5 * GIB
REPRO_ATOL = 1e-4
BOOTSTRAPS, BOOTSTRAP_SEED, N_BINS = 10000, 0, 10
STAGE_LIMIT = 0.002


def column_names(variant: str) -> list[str]:
    """基準段階を省かず、全3段階のダミーと交互作用を保持する。"""
    if variant == "V0":
        return list(G_COLUMNS)
    if variant in ("V1b", "V3"):
        return list(USER_COLUMNS)
    names = [name.replace("fill", "U") for name in G_COLUMNS]
    if variant != "V1":
        names = [name for name in names if "elapsed" not in name]
    names += ["opp_total_missing"]
    if variant == "V2":
        names += [f"{name}_x_U_{phase}" for phase in PHASES for name in D_COLUMNS]
    return names


def features(variant: str, design: np.ndarray, probability: np.ndarray,
             sign: np.ndarray, phase: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """既存59列変換を用い、事前登録された差分だけを適用する。"""
    if variant in ("V1b", "V3"):
        return user_features(design, probability, sign, phase[:, 0], missing)
    base = g_features(design, probability, sign, phase)
    if variant == "V0":
        return base
    if variant != "V1":
        keep = [i for i, name in enumerate(G_COLUMNS) if "elapsed" not in name]
        base = base[:, keep]
    pieces = [base, missing[:, None]]
    if variant == "V2":
        pieces.extend(design * (phase[:, 0] == stage)[:, None] for stage in range(len(PHASES)))
    return np.column_stack(pieces)


def memory_guard() -> None:
    """6GiBの余裕と次の学習の最大使用量を確保してから進める。"""
    entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(entries["MemAvailable"].split()[0]) * 1024
    if available < RESERVE + MAX_MEMORY:
        raise RuntimeError(f"メモリ不足のため停止: available={available}")


def load_user(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """集計コードと同じ元CSVのキー対応と段階を全行照合する。"""
    keys = ["video_id", "game_idx", "frame", "side"]
    raw = pd.read_csv(SOURCE, usecols=keys + ["board_puyo_total", "diff_board_puyo_total"])
    raw = raw.set_index(keys)
    assert raw.index.is_unique
    lookup = rows[["video_id", "frame", "source_side"]].rename(columns={"source_side": "side"})
    lookup["game_idx"] = rows.game_id.str.rsplit(":", n=1).str[-1].astype(int)
    index = pd.MultiIndex.from_frame(lookup[keys])
    assert index.isin(raw.index).all()
    aligned = raw.reindex(index)
    own, diff = (aligned[name].to_numpy() for name in ("board_puyo_total", "diff_board_puyo_total"))
    phase, missing = user_phase(own, diff)
    # 外部モジュールの起動処理・書き込みを避け、純粋関数だけを照合用に読む。
    tree = ast.parse((USER_DEF / "aggregate_user.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "phase"]
    namespace = dict(np=np, EARLY_MAX=EARLY_MAX, LATE_MIN=LATE_MIN)
    exec(compile(ast.Module(body=functions, type_ignores=[]), "aggregate_user.py", "exec"), namespace)
    reference = namespace["phase"](np.rint(own * CELLS), np.rint((own-diff) * CELLS))
    np.testing.assert_array_equal(reference[missing == 0], phase[missing == 0])
    np.testing.assert_array_equal(reference == -1, missing.astype(bool))
    assert int(missing.sum()) == 6689
    return phase, missing, dict(mismatches=0, missing=int(missing.sum()), rows=len(rows))


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """行重みの点推定。ECEは既存と同じ等幅区間で集計する。"""
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    ece = np.abs(np.bincount(bins, weights=p-y, minlength=N_BINS)).sum() / len(y)
    return dict(log_loss=float(log_loss(y, p)), auc=float(roc_auc_score(y, p)), ece=float(ece))


def reproduction(actual: dict[str, float], expected: dict[str, float]) -> dict[str, Any]:
    """統括が候補採点前に確定したLLの許容幅で再現を判定する。"""
    delta = {key: actual[key] - expected[key] for key in ("log_loss", "auc")}
    return dict(actual=actual, expected=expected, delta=delta, atol=REPRO_ATOL,
                passed=abs(delta["log_loss"]) < REPRO_ATOL)


def run_v0() -> pd.DataFrame:
    """既存trainerそのものを呼び出し、候補の前に15foldを再現する。"""
    parts = []
    for seed in SEEDS:
        for fold in FOLDS:
            memory_guard()
            started = time.monotonic()
            part = fit_g_fold(REFERENCE, seed, fold).rename(columns={"G_fe": "V0"})
            part["seed"], part["fold"] = seed, fold
            part.to_csv(OUTPUT / f"V0_seed_{seed}_fold_{fold}.csv", index=False)
            parts.append(part)
            print(f"V0 seed={seed} fold={fold} seconds={time.monotonic()-started:.2f}", flush=True)
    return pd.concat(parts, ignore_index=True)


def finish(report: dict[str, Any], baseline: pd.DataFrame, started: float) -> None:
    """再現合格時だけ候補へ進み、不一致時は停止報告を保存する。"""
    from scripts.gfe_phase_stage1_report_20261002 import evaluate, write_summary
    from scripts.gfe_phase_stage1_candidates_20261002 import train_candidate

    if report["status"] != "V0_PASS":
        write_summary(report)
        return
    rows, design, phases = datasets(REFERENCE)
    with np.load(OUTPUT / "user_phase.npz") as saved:
        u, missing = saved["phase"], saved["missing"]
    combined = baseline.copy()
    for variant in VARIANTS[1:]:
        parts = [train_candidate(variant, rows, design, u, missing, phases["elapsed"], seed, fold)
                 for seed in SEEDS for fold in FOLDS]
        candidate = pd.concat(parts, ignore_index=True)
        np.testing.assert_array_equal(combined[["row_id", "seed", "fold"]], candidate[["row_id", "seed", "fold"]])
        combined[variant] = candidate[variant].to_numpy()
    ids = combined.row_id.to_numpy()
    combined["U"], combined["current_fill"] = u[ids], phases["fill"][ids]
    combined["game_id"] = rows.game_id.to_numpy()[ids]
    combined.to_csv(OUTPUT / "predictions.csv", index=False)
    report.update(evaluate(combined))
    report.update(status="COMPLETE", elapsed_seconds=time.monotonic()-started)
    report["total_execution_seconds"] = report["elapsed_seconds"] + report.get("prior_v0_seconds", 0)
    report["candidate_execution"] = {v: "COMPLETE" for v in VARIANTS[1:]}
    report.pop("decision", None)
    write_summary(report)


def resume_registered() -> None:
    """停止履歴とV0予測を保存し、統括の再現成立判断に従って再開する。"""
    started = time.monotonic()
    for name in ("METRICS.json", "SUMMARY.md"):
        archived = OUTPUT / ("INITIAL_STOP_" + name)
        if not archived.exists():
            archived.write_bytes((OUTPUT / name).read_bytes())
    report = json.loads((OUTPUT / "INITIAL_STOP_METRICS.json").read_text())
    checks = {name: reproduction(c["actual"], c["expected"]) for name, c in report["reproduction"].items()}
    assert all(c["passed"] for c in checks.values())
    report.update(status="V0_PASS", reproduction=checks, prior_v0_seconds=report["elapsed_seconds"])
    paths = [OUTPUT / f"V0_seed_{s}_fold_{f}.csv" for s in SEEDS for f in FOLDS]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report["baseline_sha256"] = hashes
    report["resumption"] = "統括判断: |ΔLL| < 1e-4、今回の保存済みV0を基準とする"
    save_json(OUTPUT / "RESUMPTION.json", dict(rule=report["resumption"], baseline_sha256=hashes,
        lr=LR_PARAMS, hgb=HGB_PARAMS, columns={v: column_names(v) for v in VARIANTS}))
    parts = [pd.read_csv(path) for path in paths]
    baseline = pd.concat(parts, ignore_index=True)
    del parts
    finish(report, baseline, started)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def execute() -> None:
    """再現が不一致なら以降の候補を一切実行しない。"""
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save_json(OUTPUT / "PROTOCOL.json", dict(variants={v: column_names(v) for v in VARIANTS},
        lr=LR_PARAMS, hgb=HGB_PARAMS, scaling=False, reproduction_atol=REPRO_ATOL,
        seeds=list(SEEDS), folds=list(FOLDS), bootstrap_n=BOOTSTRAPS, bootstrap_seed=BOOTSTRAP_SEED))
    rows, _, _ = datasets(REFERENCE)
    u, missing, check = load_user(rows)
    np.savez(OUTPUT / "user_phase.npz", phase=u, missing=missing)
    save_json(OUTPUT / "USER_CHECK.json", check)
    result = run_v0()
    expected = json.loads((run_path(REFERENCE, "g") / "METRICS.json").read_text())["metrics"]
    checks = {}
    for name, part in [("overall", result)] + [(p, result[result.phase == i]) for i, p in enumerate(PHASES)]:
        checks[name] = reproduction(metric(part.label.to_numpy(), part.V0.to_numpy()), expected[name]["G_fe"])
    report = dict(status="V0_PASS" if all(c["passed"] for c in checks.values()) else "STOP_V0_MISMATCH",
        reproduction=checks, user_check=check, rows=len(rows), games=int(rows.game_id.nunique()),
        videos=int(rows.video_id.nunique()), predictions=len(result), elapsed_seconds=time.monotonic()-started)
    save_json(OUTPUT / "METRICS.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    finish(report, result, started)


def main() -> None:
    """nice19・最大5GiB・単一プロセスで実行する。"""
    os.nice(max(0, 19 - os.nice(0)))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    memory_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finish", action="store_true", help="保存済みV0判定から報告・候補学習へ進む")
    parser.add_argument("--resume-registered", action="store_true", help="統括判断を適用し保存済みV0から再開")
    args = parser.parse_args()
    if args.resume_registered:
        resume_registered()
    elif args.finish:
        report = json.loads((OUTPUT / "METRICS.json").read_text())
        parts = [pd.read_csv(OUTPUT / f"V0_seed_{s}_fold_{f}.csv") for s in SEEDS for f in FOLDS]
        finish(report, pd.concat(parts, ignore_index=True), time.monotonic()-report["elapsed_seconds"])
    else:
        execute()


if __name__ == "__main__":
    main()
