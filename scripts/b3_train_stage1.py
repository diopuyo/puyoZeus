"""B3 段1 (元動画 group 5fold × 3seed の外側 CV) の実行器。

**学習本番は統括の指示があるまで起動しない。** 既定は --plan (検査と割当だけ。学習しない)。
使い方 (WSL、BLAS=1):
  PYTHONPATH=. venv/bin/python -m scripts.b3_train_stage1 --rows-dir D:/.../rows_v1 --targets <targets.json> \
      --out D:/puyo_analyzer/b3_data/stage1_v1 [--run] [--seeds 1003 1004 1005] [--oracle-ll X]
出力 (--out): assignment.tsv / precheck.json / oof_{cand,ref}.parquet / scores.json / RECEIPT.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.b3_eval_cv import candidate_trainer, run_cv, stage1_scores
from src.b3_eval_gates import evaluate_stage1
from src.b3_train_config import MIN_GROUPS_TOTAL, OUTER_SEEDS
from src.b3_train_data import list_complete_parquets, load_group_map, load_training_rows, precheck_rows
from src.b3_column_spec import COLUMN_NAMES, EXCLUDED_FAST_COLUMNS
from src.b3_train_folds import (fold_summary, games_per_group_from_rows, outer_assignment, save_json,
                                write_assignment_tsv)
from src.b3_train_model import (DEFAULT_GRID, HgbConfig, feature_columns, fit_two_state, save_model,
                                model_monotonic_constraints)
from src.b3_train_rcv import rcv_trainer, validate_m0_sources

SMOKE_GRID = (HgbConfig("smoke_a", 8, 0.2, 7, 20), HgbConfig("smoke_b", 16, 0.2, 15, 20))
REPOST_MERGES = (("iZZyNGtJ3G8", "zB8d1FuCStU"),)   # c110/c112 (再投稿の疑い、事前登録 §2.2。照合後に統括が確定)


def _save_parts(parts: Path, name: str, df: pd.DataFrame, run, folds) -> None:
    """fold ごとに行キーと p を保存 (行順に依存しない結合のため row_key で持つ)。"""
    parts.mkdir(parents=True, exist_ok=True)
    for seed, p in run.preds.items():
        for k in (sorted(set(run.fold_of_row)) if folds is None else folds):
            m = run.fold_of_row == k
            pd.DataFrame({"row_key": df["row_key"].to_numpy()[m], "p": p[m]}).to_parquet(
                parts / f"{name}_seed{seed}_fold{k}.parquet", index=False)
            info = [i for i in run.infos if i["seed"] == seed and i["outer_fold"] == k]
            save_json(parts / f"{name}_seed{seed}_fold{k}.info.json", info)


def _load_parts(parts: Path, name: str, df: pd.DataFrame, seeds) -> "CvRun":
    from src.b3_eval_cv import CvRun
    from src.b3_train_config import N_OUTER_FOLDS
    run = CvRun(name)
    pos = pd.Series(np.arange(len(df)), index=df["row_key"].to_numpy())
    for seed in seeds:
        p = np.full(len(df), np.nan)
        for k in range(N_OUTER_FOLDS):
            f = parts / f"{name}_seed{seed}_fold{k}.parquet"
            if not f.exists():
                raise SystemExit(f"部分結果が欠落: {f}")
            part = pd.read_parquet(f)
            p[pos.loc[part["row_key"]].to_numpy()] = part["p"].to_numpy()
            info_path = f.with_suffix(".info.json")
            if info_path.exists():
                run.infos.extend(json.loads(info_path.read_text(encoding="utf-8")))
        if np.isnan(p).any():
            raise SystemExit(f"{name} seed={seed}: OOF に未予測行")
        run.preds[seed] = p
    return run


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-dir", type=Path, required=True)
    ap.add_argument("--m0-oof-dir", type=Path, help="b3_m0_per_foldの完成成果物。省略時は従来のM0なし近似")
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos", nargs="*", help="省略時は完了マーカーのある全動画")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(OUTER_SEEDS))
    ap.add_argument("--oracle-ll", type=float, default=None)
    ap.add_argument("--include-mask-codes", action="store_true")
    ap.add_argument("--share-inner-selection", action="store_true",
                    help="各 fold の seed 1003 の内側選択を parts に保存し、他の seed で再利用")
    ap.add_argument("--run", action="store_true", help="指定しなければ検査と割当だけ (学習しない)")
    ap.add_argument("--folds", nargs="*", type=int, help="一部の外側 fold だけ学習 (並列ジョブ用)。部分結果は parts/ に保存し採点しない")
    ap.add_argument("--grid", choices=("default", "smoke"), default="default", help="smoke は動作確認用の極小格子 (本番に使わない)")
    ap.add_argument("--score-only", action="store_true", help="parts/ の全 fold 結果を結合して採点だけ行う")
    ap.add_argument("--no-min-groups", action="store_true", help="試走用: group 数下限を強制しない")
    ap.add_argument("--save-final", type=Path, help="登録済み設定で全学習行のモデルを保存する専用実行")
    ap.add_argument("--final-config", choices=[c.name for c in DEFAULT_GRID])
    ap.add_argument("--final-seed", type=int, default=1003)
    return ap.parse_args()


def _score_parts(a: argparse.Namespace, df: pd.DataFrame, gmap: dict[str, str], parts: Path, t0: float) -> None:
    """保存済み予測を採点し、学習時の記録を受領票へ引き継ぐ。"""
    cand, ref = _load_parts(parts, "cand", df, a.seeds), _load_parts(parts, "ref", df, a.seeds)
    scores = stage1_scores(df, cand, ref, gmap)
    receipt = evaluate_stage1(scores, oracle_ll=a.oracle_ll, prechecks=None)
    for name, run in (("cand", cand), ("ref", ref)):
        out = pd.DataFrame({"row_key": df["row_key"].to_numpy(), **{f"p_{s}": p for s, p in run.preds.items()}})
        out.to_parquet(a.out / f"oof_{name}.parquet", index=False)
    save_json(a.out / "scores.json", scores)
    save_json(a.out / "RECEIPT.json", {**receipt, "elapsed_sec": time.time() - t0, "seeds": a.seeds,
                                      "training": {"cand": cand.infos, "ref": ref.infos}})
    print(json.dumps({"verdict": receipt["verdict"], "unmet": receipt["unmet"],
                      "not_evaluated": receipt["not_evaluated"]}, ensure_ascii=False))


def _save_final(a: argparse.Namespace, df: pd.DataFrame, cols: list[str], tsv_sha: str) -> None:
    """最終設定を明示した専用実行でだけ全学習行のモデルを保存する。"""
    if not a.final_config or a.score_only or a.folds is not None:
        raise SystemExit("最終保存には --final-config が必須。部分CV・採点と同時実行不可")
    cfg = next(c for c in DEFAULT_GRID if c.name == a.final_config)
    if a.save_final.exists():
        raise FileExistsError(a.save_final)
    model = fit_two_state(df, cols, cfg, a.final_seed)
    save_model(model, a.save_final)
    save_json(a.save_final.with_suffix(".json"), dict(config=a.final_config, seed=a.final_seed,
              rows=len(df), features=cols, assignment_sha256=tsv_sha,
              monotonic_cst=model_monotonic_constraints(model)))


def main() -> None:
    a = _parse_args()
    t0 = time.time()
    paths = list_complete_parquets(a.rows_dir, a.videos)
    validate_m0_sources(a.m0_oof_dir, paths, a.folds)
    gmap_all = load_group_map(a.targets, REPOST_MERGES)
    cols = feature_columns(COLUMN_NAMES, include_mask_codes=a.include_mask_codes)
    print(json.dumps({"excluded_fast_count": len(EXCLUDED_FAST_COLUMNS),
                      "excluded_fast_columns": EXCLUDED_FAST_COLUMNS}, ensure_ascii=False))
    df = load_training_rows(paths, cols)
    gmap = {v: gmap_all[v] for v in df["video_id"].unique()}
    pre = precheck_rows(df, gmap, MIN_GROUPS_TOTAL, enforce_min_groups=not a.no_min_groups)
    gpg = games_per_group_from_rows(df, gmap)
    outer = outer_assignment(gpg)
    members: dict[str, list[str]] = {}
    for v, g in gmap.items():
        members.setdefault(g, []).append(v)
    a.out.mkdir(parents=True, exist_ok=True)
    tsv_sha = write_assignment_tsv(a.out / "assignment.tsv", gpg, outer, members)
    save_json(a.out / "precheck.json", {**pre, "assignment_sha256": tsv_sha, "folds": fold_summary(gpg, outer)})
    print(json.dumps({"precheck": pre, "assignment_sha256": tsv_sha, "folds": fold_summary(gpg, outer)}, ensure_ascii=False))
    if not a.run:
        print("--plan のみ (学習は --run 指定時)")
        return
    if pre["violations_total"]:
        raise SystemExit("前提検査に違反あり。精度採点に進まない (事前登録 §3.1)")
    if a.save_final:
        _save_final(a, df, cols, tsv_sha)
        return
    parts = a.out / "parts"
    if not a.score_only:
        grid = SMOKE_GRID if a.grid == "smoke" else DEFAULT_GRID
        candidate = candidate_trainer(grid, shared_selection_parts=parts if a.share_inner_selection else None)
        for name, trainer in (("cand", candidate), ("ref", rcv_trainer(m0_oof_dir=a.m0_oof_dir))):
            run = run_cv(name, df, trainer, gmap, a.seeds, cols, progress=print, folds=a.folds)
            _save_parts(parts, name, df, run, a.folds)
        if a.folds is not None:
            print("部分実行: 採点は --score-only で全 fold 揃ってから")
            return
    _score_parts(a, df, gmap, parts, t0)


if __name__ == "__main__":
    main()
